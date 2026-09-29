"""AE training, checkpointing and encoding on the synthetic 32^3 cohort (CPU, no LPIPS weights)."""

from __future__ import annotations

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from mrextrap.data.datasets import get_longitudinal_dataloaders
from mrextrap.encoding import encode_split, encode_splits, load_autoencoder
from mrextrap.io import load_latent_dict, load_state_dict_any, validate_latent_dict
from mrextrap.training.autoencoder import (
    adv_weight,
    build_vae,
    kl_weight,
    legacy_ae_epoch_label,
    run_training,
    train_autoencoder,
)


@pytest.fixture
def ae_cfg(tiny_ae_cfg, synthetic_cohort, tmp_path):
    cfg = tiny_ae_cfg.copy()
    cfg.dataset = {
        "train_csv": str(synthetic_cohort.csvs["train"]),
        "val_csv": str(synthetic_cohort.csvs["val"]),
        "test_csv": str(synthetic_cohort.csvs["test"]),
        "cropping_offset": [0, 0, 0],
        "crop_size": [32, 32, 32],
    }
    cfg.training.num_warmup_steps = 2
    cfg.output_dir = str(tmp_path / "autoencoder")
    cfg.paths = {"run_dir": str(tmp_path), "latents_dir": str(tmp_path / "latents")}
    return cfg


def test_schedule_helpers(tiny_ae_cfg):
    cfg = tiny_ae_cfg.copy()
    cfg.training.num_epochs = 10
    cfg.training.warmup_epochs = 1
    spe = 4
    assert kl_weight(0, cfg, spe) == 0.0
    assert kl_weight(2, cfg, spe) == pytest.approx(0.5 * cfg.training.kl_beta)
    assert kl_weight(100, cfg, spe) == cfg.training.kl_beta
    cfg.training.finetune = True
    assert kl_weight(0, cfg, spe) == cfg.training.kl_beta
    # Q3: ramp counts steps from training start, so it is already 1 when the discriminator starts
    cfg.training.finetune = False
    start = int(0.2 * cfg.training.num_epochs)
    assert adv_weight(start * spe, start, cfg, spe) == 1.0
    assert adv_weight(0, start - 1, cfg, spe) == 1.0
    assert legacy_ae_epoch_label(122) == legacy_ae_epoch_label(123) == 124


def test_two_step_training_and_checkpoint_roundtrip(ae_cfg):
    ae_cfg.training.max_steps_per_epoch = 2
    ae_cfg.training.disc_epoch_start = 0  # exercise the discriminator path as well
    out_dir, history = run_training(ae_cfg)
    steps = [h for h in history if "total_loss" in h]
    assert len(steps) == 2
    assert all(torch.isfinite(torch.tensor(h["total_loss"])) for h in steps)
    assert steps[0]["loss_disc"] > 0 and steps[1]["adv_loss"] != 0.0  # ramp is 0 at step 0
    assert (out_dir / "config.yaml").is_file()
    assert (out_dir / "checkpoints" / "epoch_2" / "model.safetensors").is_file()

    # exported weights: plain VAE keys, reload reproduces the parameters
    sd = load_state_dict_any(out_dir / "model.safetensors")
    assert not any(k.startswith(("module.", "_orig_mod.")) for k in sd)
    vae = build_vae(ae_cfg.model)
    vae.load_state_dict(sd)
    vae2 = load_autoencoder(out_dir)
    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        assert torch.equal(vae.eval().encode(x)[0], vae2.encode(x)[0])
    # the accelerate state copy holds the same weights
    sd_ckpt = load_state_dict_any(out_dir / "checkpoints" / "epoch_2" / "model.safetensors")
    assert all(torch.equal(sd[k], sd_ckpt[k]) for k in sd)


def test_loss_decreases_on_one_batch(ae_cfg):
    cfg = ae_cfg
    n_train = len(pd.read_csv(cfg.dataset.train_csv))
    cfg.training.batch_size = n_train  # the whole train split is one batch
    cfg.training.num_epochs = 20
    cfg.training.disc_epoch_start = 100  # generator-only, deterministic-ish objective
    cfg.training.lr = 3e-3
    cfg.training.num_warmup_steps = 0
    _, history = run_training(cfg)
    rec = [h["rec_loss"] for h in history if "rec_loss" in h]
    assert len(rec) == 20 and all(torch.isfinite(torch.tensor(rec)))
    assert sum(rec[-3:]) / 3 < sum(rec[:3]) / 3


def test_train_autoencoder_returns_output_dir(ae_cfg):
    ae_cfg.training.max_steps_per_epoch = 1
    out_dir = train_autoencoder(ae_cfg)
    assert out_dir == type(out_dir)(ae_cfg.output_dir)
    assert (out_dir / "history.jsonl").is_file()


def test_encode_split_schema_and_single_scan_skipped(ae_cfg, synthetic_cohort):
    vae = build_vae(ae_cfg.model).eval()
    cfg = ae_cfg.copy()
    cfg.training.batch_size = 1
    # add a subject with a single scan to the train split
    train_csv = synthetic_cohort.csvs["train"]
    df = pd.read_csv(train_csv)
    extra = df.iloc[[0]].copy()
    extra["subject_id"] = "SINGLE_SCAN"
    pd.concat([df, extra]).to_csv(train_csv, index=False)
    loader, _, _ = get_longitudinal_dataloaders(cfg, num_workers=0)
    d = encode_split(vae, loader, "cpu", dataset_type="train")
    validate_latent_dict(d)
    assert d["dataset_type"] == "train"
    assert "SINGLE_SCAN" not in {s for ids in d["subject_ids"] for s in ids}
    n_multi = (df.groupby("subject_id").size() > 1).sum()
    assert len(d["zs"]) == n_multi
    z = d["zs"][0]
    assert z.shape[1:] == (4, 4, 4, 4) and z.dtype == torch.float32
    assert d["ages"][0].shape == (1, z.shape[0])


def test_encode_matches_direct_fp32_encode(ae_cfg):
    """Q11: encoding equals plain fp32 vae.encode(x)[0] of each scan."""
    vae = build_vae(ae_cfg.model).eval()
    cfg = ae_cfg.copy()
    cfg.training.batch_size = 1
    loader, _, _ = get_longitudinal_dataloaders(cfg, num_workers=0)
    d = encode_split(vae, loader, "cpu")
    batch = next(b for b in loader if b["vols"].shape[1] > 1)
    with torch.no_grad():
        ref = torch.cat([vae.encode(v.unsqueeze(0))[0] for v in batch["vols"][0]])
    assert torch.allclose(d["zs"][0], ref, atol=1e-6)


def test_encode_splits_from_config(ae_cfg):
    out_dir = train_autoencoder(OmegaConf.merge(ae_cfg, {"training": {"max_steps_per_epoch": 1}}))
    cfg = ae_cfg.copy()
    cfg.encoding = {"splits": ["val", "test"], "checkpoint": str(out_dir), "device": "cpu"}
    written = encode_splits(cfg)
    assert set(written) == {"val", "test"}
    for split, path in written.items():
        d = load_latent_dict(path)
        validate_latent_dict(d)
        assert d["dataset_type"] == split


def test_reconstructed_paper_config_loads():
    from pathlib import Path

    cfg = OmegaConf.load(Path(__file__).parents[1] / "configs" / "autoencoder.yaml")
    assert cfg.exp_name == "train_autoencoder_older8"
    assert cfg.model.ch_mult == [1, 2, 2, 4] and cfg.model.n_channels == 32
    assert cfg.training.lr == 1e-4 and cfg.training.num_warmup_steps == 10000
