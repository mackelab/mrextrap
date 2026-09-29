"""WP8: training of the UNet and diffusion priors on synthetic latents (CPU, tiny configs)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from omegaconf import OmegaConf

from mrextrap.data.synthetic import make_synthetic_latent_dict
from mrextrap.io import load_state_dict_any, save_latent_dict
from mrextrap.training.diffusion_prior import (
    build_denoiser,
    load_diffusion_prior,
    train_diffusion_prior,
)
from mrextrap.training.unet_prior import (
    augment_data,
    build_unet_prior,
    legacy_epoch_label,
    load_unet_prior,
    train_unet_prior,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _write_latents(tmp_path: Path) -> dict:
    paths = {}
    for i, split in enumerate(("train", "val", "test")):
        d = make_synthetic_latent_dict(
            5, seq_len_range=(3, 4), seed=i, dataset_type=split, include_betas=True
        )
        p = tmp_path / "latents" / f"{split}_latent_betas_l1.pt"
        save_latent_dict(d, p)
        paths[split] = str(p)
    return {
        "latents_path_train": paths["train"],
        "latents_path_val": paths["val"],
        "latents_path_test": paths["test"],
    }


@pytest.fixture
def unet_cfg(tiny_unet_cfg, tmp_path):
    cfg = tiny_unet_cfg
    cfg.dataset = _write_latents(tmp_path)
    cfg.output_dir = str(tmp_path / "unet_prior")
    cfg.training.num_epochs = 3
    return cfg


@pytest.fixture
def diffusion_cfg(tiny_unet_cfg, tmp_path):
    cfg = tiny_unet_cfg
    cfg.dataset = _write_latents(tmp_path)
    cfg.output_dir = str(tmp_path / "diffusion_prior")
    cfg.training.num_epochs = 3
    cfg.training.lr = 3e-4
    cfg.training.noise_schedule = "linear"
    cfg.training.ema_decay = 0.99
    cfg.training.NT = 20
    for k in ("T1", "T2", "augment_data", "use_wandb"):
        del cfg.training[k]
    for k in ("num_classes", "emb_bias"):
        del cfg.model[k]
    return cfg


def test_legacy_epoch_label():
    assert legacy_epoch_label(1989) == "epoch_2000"  # Q4
    assert legacy_epoch_label(1999) == "epoch_2020"
    assert legacy_epoch_label(9) == "epoch_20"
    assert legacy_epoch_label(19) == "epoch_40"
    assert legacy_epoch_label(29) == "epoch_40"


def test_augment_keeps_x_y_aligned():
    torch.manual_seed(1)
    x = torch.randn(2, 4, 3, 4, 5)
    for _ in range(20):
        ax, ay = augment_data(x, x.clone())
        assert torch.equal(ax, ay)
        assert sorted(ax.shape[2:]) == [3, 4, 5]


def test_train_unet_prior_roundtrip(unet_cfg):
    out = train_unet_prior(unet_cfg)
    log = json.loads((out / "train_log.json").read_text())
    assert all(torch.isfinite(torch.tensor(e["loss"])) for e in log if "loss" in e)
    assert any("val_loss" in e for e in log)
    assert (out / "config.yaml").is_file()
    stats = torch.load(out / "normalization.pt")
    assert set(stats) == {
        "latent_means",
        "latent_stds",
        "beta_means",
        "beta_stds",
        "age_mu",
        "age_sigma",
    }
    ckpt = out / "checkpoints" / legacy_epoch_label(2)  # only the final epoch is saved (3 < 10)
    assert (ckpt / "model.safetensors").is_file()
    assert (ckpt / "state").is_dir()

    model = load_unet_prior(unet_cfg, ckpt)
    model2 = load_unet_prior(unet_cfg, ckpt)
    z = torch.randn(2, 4, 4, 4, 4)
    y = torch.rand(2, 1)
    with torch.no_grad():
        a, b = model(z, y=y), model2(z, y=y)
    assert torch.equal(a.sample_mean, b.sample_mean)
    sd = load_state_dict_any(ckpt / "model.safetensors")
    assert set(sd) == set(build_unet_prior(unet_cfg.model).state_dict())


def test_train_unet_prior_with_augmentation(unet_cfg):
    unet_cfg.training.augment_data = True
    unet_cfg.training.num_epochs = 1
    out = train_unet_prior(unet_cfg)
    assert (out / "checkpoints" / "epoch_20" / "model.safetensors").is_file()


def test_train_diffusion_prior_ema_roundtrip(diffusion_cfg):
    out = train_diffusion_prior(diffusion_cfg)
    log = json.loads((out / "train_log.json").read_text())
    assert all(torch.isfinite(torch.tensor(e["loss"])) for e in log)
    ckpt = out / "checkpoints" / legacy_epoch_label(2)
    assert (ckpt / "ema_denoiser.safetensors").is_file()
    assert (ckpt / "state").is_dir()

    sd = load_state_dict_any(ckpt / "ema_denoiser.safetensors")
    assert set(sd) == set(build_denoiser(diffusion_cfg.model).state_dict())
    d1 = load_diffusion_prior(diffusion_cfg, ckpt)
    d2 = load_diffusion_prior(diffusion_cfg, ckpt)
    x = torch.randn(2, 8, 4, 4, 4)
    with torch.no_grad():
        o1 = d1(x, timestep=torch.tensor([3, 7]), y=torch.rand(2, 1)).sample
        o2 = d2(x, timestep=torch.tensor([3, 7]), y=torch.rand(2, 1)).sample
    assert o1.shape == (2, 4, 4, 4, 4)
    assert torch.isfinite(o1).all() and o1.shape == o2.shape

    # EMA weights differ from the raw (accelerate state) weights
    raw = load_state_dict_any(ckpt / "state" / "model.safetensors")
    assert any(not torch.equal(raw[k], sd[k]) for k in sd if sd[k].is_floating_point())


def test_diffusion_optimizer_betas_preserved(diffusion_cfg, monkeypatch):
    """Q5: AdamW betas (0.95, 0.999) and hard-coded grad clip 2.0."""
    seen = {}
    real = torch.optim.AdamW

    def spy(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)

    monkeypatch.setattr(torch.optim, "AdamW", spy)
    diffusion_cfg.training.num_epochs = 1
    train_diffusion_prior(diffusion_cfg)
    assert seen["betas"] == (0.95, 0.999) and seen["weight_decay"] == 1e-6


# ---- paper hyperparameters: values of the W&B config.yaml of the paper runs -------------------
# run-20250104_003705-737ieovl  (UNet)   run-20240623_233801-bfxpfvzd  (diffusion)
UNET_WANDB = {
    "training": {
        "lr": 0.0001,
        "seed": 42,
        "num_epochs": 2000,
        "batch_size": 160,
        "gradient_accumulation_steps": 1,
        "num_warmup_steps": 4000,
        "mixed_precision": "bf16",
        "num_workers": 4,
        "lr_scheduler": "cosine",
        "grad_clip_val": 1.0,
        "T1": 1.0,
        "T2": 0.001,
        "torch_compile": False,
        "use_wandb": True,
        "augment_data": True,
    },
    "model": {
        "image_size": [4, 20, 24, 20],
        "in_channels": 4,
        "model_channels": 64,
        "out_channels": 4,
        "num_res_blocks": 2,
        "attention_resolutions": [],
        "channel_mult": [1, 2],
        "num_classes": 1,
        "emb_bias": True,
    },
    "exp_name": "train_unet_mean_var_with_gauss_loss_lowlr_aug_warmup4000",
}
DIFFUSION_WANDB = {
    "training": {
        "lr": 0.0003,
        "seed": 42,
        "num_epochs": 2000,
        "batch_size": 256,
        "gradient_accumulation_steps": 1,
        "num_warmup_steps": 3000,
        "mixed_precision": "bf16",
        "torch_compile": True,
        "num_workers": 4,
        "lr_scheduler": "cosine",
        "noise_schedule": "linear",
        "ema_decay": 0.99,
        "NT": 500,
        "grad_clip_val": 1.0,
    },
    "model": {
        "image_size": [4, 20, 24, 20],
        "in_channels": 4,
        "model_channels": 64,
        "out_channels": 4,
        "num_res_blocks": 2,
        "attention_resolutions": [],
        "channel_mult": [1, 2],
    },
    "exp_name": "train_beta_diffusion_smaller64highlr3e-4longtrainlinear",
}


@pytest.mark.parametrize(
    ("name", "expected", "allowed_diff"),
    [
        ("unet_prior.yaml", UNET_WANDB, {("training", "use_wandb")}),  # released default: false
        ("diffusion_prior.yaml", DIFFUSION_WANDB, set()),
    ],
)
def test_prior_configs_match_wandb_paper_runs(name, expected, allowed_diff):
    cfg = yaml.safe_load((CONFIGS / name).read_text())
    for section in ("training", "model"):
        for key, value in expected[section].items():
            if (section, key) in allowed_diff:
                continue
            assert cfg[section][key] == value, f"{name}: {section}.{key}"
        assert set(cfg[section]) == set(expected[section])
    assert cfg["exp_name"] == expected["exp_name"]
    assert cfg["model"]["model_channels"] == 64  # not the WIDER 128
    assert cfg["model"]["channel_mult"] == [1, 2]
    assert OmegaConf.load(CONFIGS / name) is not None
