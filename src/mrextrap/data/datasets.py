"""Image, longitudinal and latent-beta datasets plus their dataloader builders."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from mrextrap.data.images import DEFAULT_CROP_SIZE, DEFAULT_CROPPING_OFFSET, build_t1_transform
from mrextrap.io import load_latent_dict

LEGACY_PATH_PREFIX = "/mnt_mount/ADNI"


def sort_df_by_age_from_baseline(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)

        df_sorted = (
            df.groupby("subject_id", sort=False)
            .apply(lambda x: x.sort_values(by="age_from_baseline"))
            .reset_index(drop=True)
        )
    return df_sorted


class AutoencoderDataset(Dataset):
    """Cross-sectional dataset: one preprocessed T1 volume per row of ``df["filename"]``."""

    def __init__(
        self,
        df,
        normalize="minmax0mean",
        cropping_offset=DEFAULT_CROPPING_OFFSET,
        normalize_to_1=True,
        crop_size=DEFAULT_CROP_SIZE,
    ):
        super().__init__()
        if cropping_offset is None:
            cropping_offset = DEFAULT_CROPPING_OFFSET
        self.t1_paths = df["filename"].values
        self.normalize = normalize
        self.transforms_t1 = build_t1_transform(
            cropping_offset, crop_size, normalize=normalize, normalize_to_1=normalize_to_1
        )

    def __len__(self):
        return len(self.t1_paths)

    def __getitem__(self, idx):
        return self.transforms_t1(self.t1_paths[idx])


class LongitudinalDataset(Dataset):
    """Returns the entire series of volumes of one subject (sorted as given in the dataframe)."""

    def __init__(
        self,
        df_dict_per_subject,
        normalize="minmax0mean",
        cropping_offset=DEFAULT_CROPPING_OFFSET,
        normalize_to_1=True,
        return_full_subject=True,
        crop_size=DEFAULT_CROP_SIZE,
    ):
        """
        Args:
            df_dict_per_subject: ``{subject_id: DataFrame}`` (rows already sorted by age).
            return_full_subject: only True is implemented (as in the paper code).
        """
        super().__init__()
        if cropping_offset is None:
            cropping_offset = DEFAULT_CROPPING_OFFSET
        if not return_full_subject:
            raise NotImplementedError("return_full_subject=False not implemented yet")
        self.return_full_subject = return_full_subject
        # NOTE(paper): subject order is np.unique (sorted), not dataframe order.
        self.unique_subjects = np.unique(list(df_dict_per_subject.keys()))
        self.normalize = normalize
        self.df_dict_per_subject = df_dict_per_subject
        self.transforms_t1 = build_t1_transform(
            cropping_offset, crop_size, normalize=normalize, normalize_to_1=normalize_to_1
        )

    def __len__(self):
        return len(self.unique_subjects)

    def __getitem__(self, idx):
        """Return ``{"vols": [S,1,D,H,W], "ages": [S], "ages_abs": [S], "subject_id": str}``."""
        subject = self.unique_subjects[idx]
        df = self.df_dict_per_subject[subject]
        vols, ages, ages_abs = [], [], []
        for i in range(len(df)):
            vols.append(self.transforms_t1(df.iloc[i]["filename"]))
            ages.append(df.iloc[i]["age_from_baseline"])
            ages_abs.append(df.iloc[i]["AGE_filled"])
        return {
            "vols": torch.stack(vols, dim=0),
            "ages": torch.tensor(ages),
            "ages_abs": torch.tensor(ages_abs),
            "subject_id": subject,
        }


def _read_split_csvs(cfg):
    dfs = [
        pd.read_csv(cfg.dataset.train_csv),
        pd.read_csv(cfg.dataset.val_csv),
        pd.read_csv(cfg.dataset.test_csv),
    ]
    base_t1_path = cfg.dataset.get("base_t1_path", None)
    if base_t1_path is not None:
        prefix = cfg.dataset.get("legacy_path_prefix", LEGACY_PATH_PREFIX)
        for df in dfs:
            df["filename"] = df["filename"].apply(lambda x: x.replace(prefix, base_t1_path))
    return dfs


def _crop_kwargs(cfg):
    ds = cfg.dataset
    offset = ds.get("cropping_offset", None)
    size = ds.get("crop_size", None)
    return {
        "cropping_offset": list(offset) if offset is not None else DEFAULT_CROPPING_OFFSET,
        "crop_size": list(size) if size is not None else DEFAULT_CROP_SIZE,
    }


def _loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_autoencoder_dataloaders(cfg, num_workers=4):
    """Train (shuffled), val and test loaders over individual volumes."""
    dfs = _read_split_csvs(cfg)
    kw = _crop_kwargs(cfg)
    bs = cfg.training.batch_size
    datasets = [AutoencoderDataset(df, **kw) for df in dfs]
    return (
        _loader(datasets[0], bs, True, num_workers),
        _loader(datasets[1], bs, False, num_workers),
        _loader(datasets[2], bs, False, num_workers),
    )


def get_longitudinal_dataloaders(cfg, num_workers=4, shuffle_train=False):
    """Train, val and test loaders returning whole subject series (sorted by age_from_baseline).

    NOTE(paper): the original ``get_autoencoder_dataloaders_longitudinal`` defaulted to
    ``shuffle_train=True``; the release defaults to False (encoding/eval never need shuffling).
    """
    dfs = _read_split_csvs(cfg)
    kw = _crop_kwargs(cfg)
    bs = cfg.training.batch_size
    datasets = []
    for df in dfs:
        per_subject = {s: df[df.subject_id == s] for s in df.subject_id.unique()}
        per_subject = {s: sort_df_by_age_from_baseline(d) for s, d in per_subject.items()}
        datasets.append(LongitudinalDataset(per_subject, **kw))
    return (
        _loader(datasets[0], bs, shuffle_train, num_workers),
        _loader(datasets[1], bs, False, num_workers),
        _loader(datasets[2], bs, False, num_workers),
    )


class LatentBetaDataset(Dataset):
    """Flattened (per-scan) latents with their per-scan betas (non-Bayesian path only).

    Reads a ``{split}_latent_betas_l1.pt`` dict (see ``mrextrap.io``). Latents and betas are
    standardised channel-wise; statistics are computed on the given data unless supplied
    (val/test must reuse the train statistics).
    """

    def __init__(
        self,
        latents_path,
        latent_means=None,
        latent_stds=None,
        beta_means=None,
        beta_stds=None,
        clip_params=False,
    ):
        if "bayesian" in str(latents_path):
            raise NotImplementedError("The Bayesian beta variant is not part of the release.")
        self.latents = load_latent_dict(latents_path)

        self.zs = self.latents["zs"]
        self.ages = [age[0] for age in self.latents["ages"]]
        self.ages_abs = [age[0] for age in self.latents["ages_abs"]]
        # NOTE(paper): legacy files store ONE id per subject (``[sid]``), repeated per scan here.
        # Files with one id per scan (``[sid] * N``) are accepted as well.
        self.subject_ids = [
            subject_id_list * len(z) if len(subject_id_list) == 1 else list(subject_id_list)
            for z, subject_id_list in zip(self.zs, self.latents["subject_ids"])
        ]
        self.betas = self.latents["betas"]

        # remove sequences containing NaNs in any of the tensors
        any_nan_batches = [
            bool(
                torch.isnan(z).any()
                | torch.isnan(a).any()
                | torch.isnan(aa).any()
                | torch.isnan(b).any()
            )
            for z, a, aa, b in zip(self.zs, self.ages, self.ages_abs, self.betas)
        ]

        def keep(items):
            return [x for x, nan in zip(items, any_nan_batches) if not nan]

        self.zs = keep(self.zs)
        self.ages = keep(self.ages)
        self.ages_abs = keep(self.ages_abs)
        self.betas = keep(self.betas)
        # NOTE(paper): (Q13) ``subject_ids`` is only filtered for NaN sequences in the Bayesian
        # branch of the original code, which is not part of the release. Here it is therefore not
        # filtered, so the length assertion below fires if any NaN sequence was removed.

        self.zs = torch.cat(self.zs)
        self.ages = torch.cat(self.ages)
        self.ages_abs = torch.cat(self.ages_abs)
        self.betas = torch.cat(self.betas)
        self.clip_params = clip_params

        self.subject_ids = [sid for sid_list in self.subject_ids for sid in sid_list]

        if latent_means is not None and latent_stds is not None:
            self.latent_means = latent_means
            self.latent_stds = latent_stds
        else:
            # NOTE(paper): channel-wise mean and unbiased std over all (scan, x, y, z) positions.
            self.latent_means = self.zs.mean(dim=(0, 2, 3, 4)).reshape(-1, 1, 1, 1)
            self.latent_stds = self.zs.std(dim=(0, 2, 3, 4)).reshape(-1, 1, 1, 1)
        self.zs = (self.zs - self.latent_means) / self.latent_stds

        if beta_means is not None and beta_stds is not None:
            self.beta_means = beta_means
            self.beta_stds = beta_stds
        else:
            self.beta_means = self.betas.mean(dim=(0, 2, 3, 4)).reshape(-1, 1, 1, 1)
            self.beta_stds = self.betas.std(dim=(0, 2, 3, 4)).reshape(-1, 1, 1, 1)
        self.betas = (self.betas - self.beta_means) / self.beta_stds

        assert (
            len(self.zs)
            == len(self.ages)
            == len(self.ages_abs)
            == len(self.betas)
            == len(self.subject_ids)
        ), (
            "Lengths of zs, ages, ages_abs, betas, and subject_ids must match, got "
            f"{len(self.zs)}, {len(self.ages)}, {len(self.ages_abs)}, {len(self.betas)}, "
            f"{len(self.subject_ids)}"
        )

    def __len__(self):
        return self.zs.shape[0]

    def normalization_stats(self) -> dict:
        """Statistics for ``priors/normalization.pt``; ``age_*`` are over absolute ages."""
        return {
            "latent_means": self.latent_means,
            "latent_stds": self.latent_stds,
            "beta_means": self.beta_means,
            "beta_stds": self.beta_stds,
            "age_mu": self.ages_abs.mean(),
            "age_sigma": self.ages_abs.std(),
        }

    def standardize_latents(self, z):
        return (z - self.latent_means) / self.latent_stds

    def destandardize_latents(self, z):
        return z * self.latent_stds + self.latent_means

    def standardize_betas(self, beta):
        return (beta - self.beta_means) / self.beta_stds

    def destandardize_betas(self, beta):
        return beta * self.beta_stds + self.beta_means

    def __getitem__(self, idx):
        return {
            "subject_id": self.subject_ids[idx],
            "zs": self.zs[idx],
            "ages": self.ages[idx],
            "ages_abs": self.ages_abs[idx],
            "betas": self.betas[idx],
        }


def get_latent_dataloaders(cfg, num_workers=4):
    """Train/val/test loaders over ``LatentBetaDataset``; val/test reuse the train statistics."""
    train_dataset = LatentBetaDataset(Path(cfg.dataset.latents_path_train))
    stats = {
        "latent_means": train_dataset.latent_means,
        "latent_stds": train_dataset.latent_stds,
        "beta_means": train_dataset.beta_means,
        "beta_stds": train_dataset.beta_stds,
    }
    val_dataset = LatentBetaDataset(Path(cfg.dataset.latents_path_val), **stats)
    test_dataset = LatentBetaDataset(Path(cfg.dataset.latents_path_test), **stats)
    bs = cfg.training.batch_size
    return (
        _loader(train_dataset, bs, True, num_workers),
        _loader(val_dataset, bs, False, num_workers),
        _loader(test_dataset, bs, False, num_workers),
    )


LongitudinalLatentRegressionDataset = LatentBetaDataset  # legacy name
