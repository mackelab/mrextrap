"""Synthetic phantoms, cohorts and latent dictionaries for tests and the CPU smoke run.

Nothing here derives from real subject data. All generators are deterministic given their seeds.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

SPLITS = ("train", "val", "test")
BASELINE_AGE_REF = 60.0  # ventricle radius equals its base radius at this age

# Real SynthSeg label names / FreeSurfer ids (32 structures + background is not written).
# Region aggregation (Q9) is substring based, so names must match the real CSV columns.
_SYNTHSEG_LABELS: tuple[tuple[int, str], ...] = (
    (2, "left cerebral white matter"),
    (3, "left cerebral cortex"),
    (4, "left lateral ventricle"),
    (5, "left inferior lateral ventricle"),
    (7, "left cerebellum white matter"),
    (8, "left cerebellum cortex"),
    (10, "left thalamus"),
    (11, "left caudate"),
    (12, "left putamen"),
    (13, "left pallidum"),
    (14, "3rd ventricle"),
    (15, "4th ventricle"),
    (16, "brain-stem"),
    (17, "left hippocampus"),
    (18, "left amygdala"),
    (26, "left accumbens area"),
    (28, "left ventral DC"),
    (41, "right cerebral white matter"),
    (42, "right cerebral cortex"),
    (43, "right lateral ventricle"),
    (44, "right inferior lateral ventricle"),
    (46, "right cerebellum white matter"),
    (47, "right cerebellum cortex"),
    (49, "right thalamus"),
    (50, "right caudate"),
    (51, "right putamen"),
    (52, "right pallidum"),
    (53, "right hippocampus"),
    (54, "right amygdala"),
    (58, "right accumbens area"),
    (60, "right ventral DC"),
)
SYNTHSEG_LABEL_IDS = tuple(i for i, _ in _SYNTHSEG_LABELS)
SYNTHSEG_LABEL_NAMES = tuple(n for _, n in _SYNTHSEG_LABELS)


def _ellipsoid(shape, center, radii) -> np.ndarray:
    grids = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in shape], indexing="ij")
    d = sum(((g - c) / r) ** 2 for g, c, r in zip(grids, center, radii, strict=True))
    return d <= 1.0


def ventricle_radius_fraction(age: float, atrophy_rate: float, base: float = 0.10) -> float:
    """Ventricle radius (fraction of the volume side) grows linearly with age."""
    return base + atrophy_rate * (age - BASELINE_AGE_REF)


def make_phantom_volume(
    shape=(32, 32, 32), age: float = 70.0, subject_seed: int = 0, atrophy_rate: float = 0.002
) -> np.ndarray:
    """Brain ellipsoid with a central "ventricle" ellipsoid whose radius grows linearly with age.

    Geometry (centre jitter, tissue intensities) depends only on ``subject_seed``; only the
    ventricle radius depends on ``age`` (rate ``atrophy_rate`` per year, as a fraction of the
    side length). Additive noise is seeded by (subject_seed, age), so the result is deterministic.
    Returns float32 with background 0 and intensities up to ~1000 (MPRAGE-like, non-negative),
    which is the range ``clip_and_normalize_image`` expects.
    """
    shape = tuple(int(s) for s in shape)
    rng = np.random.RandomState(subject_seed)
    jitter = rng.uniform(-0.03, 0.03, size=3) * np.array(shape)
    wm_int, csf_int, gm_int = rng.uniform(750, 850), rng.uniform(80, 120), rng.uniform(450, 550)
    centre = np.array(shape) / 2.0 + jitter
    brain_radii = np.array(shape) * np.array([0.42, 0.46, 0.40])
    brain = _ellipsoid(shape, centre, brain_radii)
    core = _ellipsoid(shape, centre, brain_radii * 0.85)  # white matter core, grey matter shell
    vent_r = max(ventricle_radius_fraction(age, atrophy_rate), 0.01)
    vent = _ellipsoid(shape, centre, np.array(shape) * np.array([vent_r, vent_r * 1.6, vent_r]))
    vol = np.zeros(shape, dtype=np.float32)
    vol[brain] = gm_int
    vol[core] = wm_int
    vol[vent & brain] = csf_int
    noise_seed = zlib.crc32(f"{subject_seed}:{age:.4f}".encode())
    noise = np.random.RandomState(noise_seed).normal(0, 10, size=shape).astype(np.float32)
    vol = np.where(vol > 0, np.clip(vol + noise, 1, None), 0).astype(np.float32)
    return vol


def ventricle_volume(volume: np.ndarray) -> int:
    """Number of voxels in the CSF intensity band (< 200, > 0); used by tests."""
    return int(((volume > 0) & (volume < 200)).sum())


def _diagnosis(rng: np.random.RandomState, n_scans: int) -> list[str]:
    final = rng.choice(["CN", "MCI", "Dementia"])
    order = ["CN", "MCI", "Dementia"]
    k = order.index(final)
    switch = rng.randint(1, n_scans + 1)
    return [order[k] if j >= switch else order[max(k - 1, 0)] for j in range(n_scans)]


def make_synthetic_cohort(
    root,
    n_subjects: int = 6,
    scans_per_subject=(2, 5),
    shape=(32, 32, 32),
    seed: int = 0,
) -> dict[str, Path]:
    """Write phantom NIfTIs under ``root/images`` and split CSVs under ``root/splits``.

    ``scans_per_subject`` is an int or an inclusive ``(min, max)`` range. Subjects are split by
    subject into train/val/test (~50/25/25 %, each split non-empty for n_subjects >= 3).
    CSV columns follow the split schema: subject_id, session_id, filename, AGE_filled,
    age_from_baseline, DX_filled (rows sorted by subject then age). Returns split -> csv path.
    """
    root = Path(root)
    img_dir, split_dir = root / "images", root / "splits"
    img_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    lo, hi = (
        (scans_per_subject, scans_per_subject)
        if isinstance(scans_per_subject, int)
        else scans_per_subject
    )
    rng = np.random.RandomState(seed)
    n_test = max(1, round(0.25 * n_subjects))
    n_val = max(1, round(0.25 * n_subjects))
    split_of = ["test"] * n_test + ["val"] * n_val
    split_of += ["train"] * (n_subjects - len(split_of))
    rows: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for i in range(n_subjects):
        sid = f"SYN_{i:03d}"
        n_scans = int(rng.randint(lo, hi + 1))
        base_age = rng.uniform(60, 85)
        rate = rng.uniform(0.001, 0.004)
        deltas = np.concatenate([[0.0], np.cumsum(rng.uniform(0.8, 1.4, size=n_scans - 1))])
        dx = _diagnosis(rng, n_scans)
        (img_dir / sid).mkdir(exist_ok=True)
        for j, dt in enumerate(deltas):
            age = float(base_age + dt)
            vol = make_phantom_volume(shape, age, subject_seed=seed * 1000 + i, atrophy_rate=rate)
            path = img_dir / sid / f"scan{j:02d}.nii.gz"
            nib.save(nib.Nifti1Image(vol, np.eye(4)), str(path))
            rows[split_of[i]].append(
                {
                    "subject_id": sid,
                    "session_id": f"ses{j:02d}",
                    "filename": str(path),
                    "AGE_filled": age,
                    "age_from_baseline": float(dt),
                    "DX_filled": dx[j],
                }
            )
    out = {}
    for split in SPLITS:
        out[split] = split_dir / f"{split}.csv"
        pd.DataFrame(rows[split]).to_csv(out[split], index=False)
    return out


def make_synthetic_latent_dict(
    n_subjects: int = 6,
    seq_len_range=(5, 8),
    latent_shape=(4, 4, 4, 4),
    seed: int = 0,
    dataset_type: str = "train",
    include_betas: bool = False,
    noise: float = 0.01,
) -> dict:
    """Legacy-schema latent dict with exactly linear-in-age latents (plus small noise).

    z_t = z_0 + beta * age_from_baseline_t + noise. Keys: zs (list of (N,C,D,H,W)),
    ages (list of (1,N), years since first scan), ages_abs (list of (1,N)), subject_ids
    (list of list[str]), dataset_type; with ``include_betas`` also betas (list of (N,C,D,H,W),
    the true slope repeated per scan). Spacing is 1.0-1.3 y, so >= 6 scans span >= 5 years.
    The ground-truth slopes are additionally stored under ``true_betas`` (list of (C,D,H,W)).
    """
    g = torch.Generator().manual_seed(seed)
    lo, hi = seq_len_range
    zs, ages, ages_abs, sids, betas, true_betas = [], [], [], [], [], []
    for i in range(n_subjects):
        n = int(torch.randint(lo, hi + 1, (1,), generator=g))
        z0 = torch.randn(*latent_shape, generator=g)
        beta = 0.1 * torch.randn(*latent_shape, generator=g)
        gaps = 1.0 + 0.3 * torch.rand(n - 1, generator=g)
        age = torch.cat([torch.zeros(1), torch.cumsum(gaps, 0)])
        base = 60.0 + 25.0 * float(torch.rand(1, generator=g))
        z = z0[None] + beta[None] * age.view(-1, 1, 1, 1, 1)
        z = z + noise * torch.randn(z.shape, generator=g)
        zs.append(z.float())
        ages.append(age[None].float())
        ages_abs.append((age + base)[None].float())
        sids.append([f"SYN_{dataset_type}_{i:03d}"] * n)
        true_betas.append(beta)
        betas.append(beta[None].expand(n, *latent_shape).clone())
    d = {
        "zs": zs,
        "ages": ages,
        "ages_abs": ages_abs,
        "subject_ids": sids,
        "dataset_type": dataset_type,
        "true_betas": true_betas,
    }
    if include_betas:
        d["betas"] = betas
    return d


def make_fake_vols_csv(path, n_rows: int = 4, seed: int = 0) -> Path:
    """Write a SynthSeg-style ``vols.csv``: header of names, then a row of label ids, then data.

    NOTE(paper): the real file has two header rows (names, label ids) plus an unnamed index
    column, hence legacy ``read_csv(...)[1:]`` and ``drop(columns=["Unnamed: 0"])``.
    Volumes are in mm^3, rows are ``tmpNNN_final``; ventricles grow with the row index.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    base = rng.uniform(2000, 12000, size=len(SYNTHSEG_LABEL_NAMES))
    for k, name in enumerate(SYNTHSEG_LABEL_NAMES):
        if "ventricle" in name:
            base[k] *= 0.5
    data = []
    for r in range(n_rows):
        scale = np.array(
            [1 + 0.05 * r if "ventricle" in n else 1 - 0.01 * r for n in SYNTHSEG_LABEL_NAMES]
        )
        data.append(np.round(base * scale * rng.uniform(0.99, 1.01, size=base.shape), 1))
    df = pd.DataFrame(data, columns=list(SYNTHSEG_LABEL_NAMES))
    df.index = [f"tmp{r:03d}_final" for r in range(n_rows)]
    ids = pd.DataFrame([list(SYNTHSEG_LABEL_IDS)], columns=df.columns, index=[""])
    pd.concat([ids, df]).to_csv(path)
    return path
