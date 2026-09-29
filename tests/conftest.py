"""Shared pytest fixtures: synthetic data and tiny model configs (CPU, offline, fast)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from mrextrap.data.synthetic import (
    SPLITS,
    make_fake_vols_csv,
    make_synthetic_cohort,
    make_synthetic_latent_dict,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def device() -> str:
    return "cpu"


@dataclass
class SyntheticCohort:
    """``run_dir/images/<sid>/scanNN.nii.gz`` (32^3) and ``run_dir/splits/{split}.csv``."""

    run_dir: Path
    csvs: dict[str, Path]
    shape: tuple[int, int, int] = (32, 32, 32)

    @property
    def splits_dir(self) -> Path:
        return self.run_dir / "splits"


@pytest.fixture
def synthetic_cohort(tmp_path) -> SyntheticCohort:
    run_dir = tmp_path / "run_dir"
    csvs = make_synthetic_cohort(run_dir, n_subjects=6, scans_per_subject=(2, 5), shape=(32,) * 3)
    return SyntheticCohort(run_dir=run_dir, csvs=csvs)


@pytest.fixture
def synthetic_latent_dict() -> dict[str, dict]:
    """split -> latent dict (latent 4x4x4x4). Every subject has 6-8 scans spanning >= 5 years,
    so multi-volume filtering keeps every subject (6 train+val+test subjects each >= 6)."""
    n = {"train": 8, "val": 6, "test": 6}
    return {
        s: make_synthetic_latent_dict(
            n[s], seq_len_range=(6, 8), latent_shape=(4, 4, 4, 4), seed=i, dataset_type=s
        )
        for i, s in enumerate(SPLITS)
    }


@pytest.fixture
def fake_vols_csv(tmp_path) -> Path:
    return make_fake_vols_csv(tmp_path / "vols.csv", n_rows=4, seed=0)


@pytest.fixture
def tiny_ae_cfg() -> DictConfig:
    """Tiny CPU autoencoder config for 32^3 inputs (latent 4x4^3); mirrors configs/smoke."""
    return OmegaConf.create(
        {
            "training": {
                "lr": 1e-4,
                "seed": 0,
                "num_epochs": 1,
                "batch_size": 2,
                "gradient_accumulation_steps": 1,
                "mixed_precision": "no",
                "torch_compile": False,
                "num_workers": 0,
                "disc_beta": 0.005,
                "lpips_beta": 0.0,
                "kl_beta": 1e-5,
                "lr_scheduler": "cosine",
                "grad_clip_val": 0.5,
                "finetune": False,
                "warmup_epochs": 1,
                "use_wandb": False,
                "cpu": True,
            },
            "model": {
                "n_channels": 16,  # GroupNorm(16 groups) needs >= 16 channels
                "z_channels": 4,
                "ch_mult": [1, 2, 2, 4],
                "num_res_blocks": 1,
                "resolution": [32, 32, 32],
            },
            "exp_name": "tiny_autoencoder",
        }
    )


@pytest.fixture
def tiny_unet_cfg() -> DictConfig:
    """Tiny prior-UNet config for 4x4x4x4 latents; mirrors configs/smoke/unet_prior.yaml."""
    return OmegaConf.create(
        {
            "training": {
                "lr": 1e-4,
                "seed": 0,
                "num_epochs": 2,
                "batch_size": 4,
                "gradient_accumulation_steps": 1,
                "num_warmup_steps": 2,
                "mixed_precision": "no",
                "num_workers": 0,
                "lr_scheduler": "cosine",
                "grad_clip_val": 1.0,
                "T1": 1.0,
                "T2": 0.001,
                "torch_compile": False,
                "use_wandb": False,
                "augment_data": False,
                "cpu": True,
            },
            "model": {
                "image_size": [4, 4, 4, 4],
                "in_channels": 4,
                "model_channels": 32,  # GroupNorm(32 groups) needs a multiple of 32
                "out_channels": 4,
                "num_res_blocks": 1,
                "attention_resolutions": [],
                "channel_mult": [1],
                "num_classes": 1,
                "emb_bias": True,
            },
            "exp_name": "tiny_unet_prior",
        }
    )
