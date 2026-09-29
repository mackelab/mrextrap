"""Tests for config merging, latent IO and the CLI shell (WP1)."""

import pytest
import torch
from omegaconf import OmegaConf

from mrextrap.cli import SUBCOMMANDS, main
from mrextrap.config import load_config
from mrextrap.io import (
    load_latent_dict,
    load_state_dict_any,
    save_latent_dict,
    save_safetensors,
    validate_latent_dict,
)

# Paper values, hard-coded because the legacy source configs are not available in CI.
UNET_TRAINING = {
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
    "use_wandb": False,
    "augment_data": True,
}
UNET_MODEL = {
    "image_size": [4, 20, 24, 20],
    "in_channels": 4,
    "model_channels": 64,
    "out_channels": 4,
    "num_res_blocks": 2,
    "attention_resolutions": [],
    "channel_mult": [1, 2],
    "num_classes": 1,
    "emb_bias": True,
}
DIFF_TRAINING = {
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
}
DIFF_MODEL = {
    "image_size": [4, 20, 24, 20],
    "in_channels": 4,
    "model_channels": 64,
    "out_channels": 4,
    "num_res_blocks": 2,
    "attention_resolutions": [],
    "channel_mult": [1, 2],
}


def _plain(node):
    return OmegaConf.to_container(node, resolve=True)


def test_paper_configs_match_expected():
    unet = load_config("configs/unet_prior.yaml")
    assert _plain(unet.training) == UNET_TRAINING
    assert _plain(unet.model) == UNET_MODEL
    assert unet.exp_name == "train_unet_mean_var_with_gauss_loss_lowlr_aug_warmup4000"
    diff = load_config("configs/diffusion_prior.yaml")
    assert _plain(diff.training) == DIFF_TRAINING
    assert _plain(diff.model) == DIFF_MODEL
    assert diff.exp_name == "train_beta_diffusion_smaller64highlr3e-4longtrainlinear"


def test_merge_and_dotlist():
    cfg = load_config("configs/unet_prior.yaml", ["training.num_epochs=3", "paths.run_dir=/x"])
    assert cfg.training.num_epochs == 3
    assert cfg.dataset.latents_path_train == "/x/latents/train_latent_betas_l1.pt"


def test_smoke_config_finds_parent_paths():
    cfg = load_config("configs/smoke/unet_prior.yaml")
    assert cfg.paths.run_dir


def _latent_dict(n_subj=2, with_betas=True):
    zs = [torch.randn(3 + i, 4, 4, 4, 4) for i in range(n_subj)]
    d = {
        "zs": zs,
        "ages": [torch.arange(z.shape[0], dtype=torch.float32)[None] for z in zs],
        "ages_abs": [70 + torch.arange(z.shape[0], dtype=torch.float32)[None] for z in zs],
        "subject_ids": [[f"S{i}"] * z.shape[0] for i, z in enumerate(zs)],
        "dataset_type": "train",
    }
    if with_betas:
        d["betas"] = [torch.randn_like(z) for z in zs]
    return d


def test_latent_dict_round_trip_and_validator(tmp_path):
    d = _latent_dict()
    save_latent_dict(d, tmp_path / "sub" / "l.pt")
    back = load_latent_dict(tmp_path / "sub" / "l.pt")
    validate_latent_dict(back, require_betas=True)
    assert torch.equal(back["zs"][1], d["zs"][1])
    assert back["subject_ids"] == d["subject_ids"]


def test_validator_rejects_bad_dicts():
    d = _latent_dict(with_betas=False)
    with pytest.raises(ValueError, match="betas"):
        validate_latent_dict(d, require_betas=True)
    del d["ages"]
    with pytest.raises(ValueError, match="ages"):
        validate_latent_dict(d)
    d = _latent_dict()
    d["betas"][0] = d["betas"][0][:-1]
    with pytest.raises(ValueError, match="betas"):
        validate_latent_dict(d)


@pytest.mark.parametrize("suffix", [".safetensors", ".bin", ".pt"])
def test_load_state_dict_any_strips_prefixes(tmp_path, suffix):
    sd = {
        "_orig_mod.a.weight": torch.ones(2),
        "module.b.bias": torch.zeros(3),
        "module._orig_mod.c": torch.ones(1),
        "d": torch.ones(1),
    }
    path = tmp_path / f"m{suffix}"
    if suffix == ".safetensors":
        save_safetensors(sd, path)
    else:
        torch.save(sd, path)
    out = load_state_dict_any(path)
    assert sorted(out) == ["a.weight", "b.bias", "c", "d"]


def test_cli_help_lists_all_subcommands(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for name in SUBCOMMANDS:
        assert name in out
    assert len(SUBCOMMANDS) == 15


def test_load_latent_dict_accepts_legacy_numpy_str_ids(tmp_path):
    """Paper-era latent files store subject ids as numpy.str_ (rejected by weights_only=True)."""
    import numpy as np
    import pytest
    import torch

    from mrextrap.io import load_latent_dict

    d = {
        "zs": [torch.zeros(2, 1)],
        "ages": [torch.zeros(2)],
        "ages_abs": [torch.zeros(2)],
        "subject_ids": [np.array(["a", "a"])[0]],
    }
    torch.save(d, tmp_path / "legacy.pt")
    with pytest.warns(UserWarning, match="weights_only=False"):
        out = load_latent_dict(tmp_path / "legacy.pt")
    assert str(out["subject_ids"][0]) == "a"
