"""Tests for image transforms, datasets and the ADNI split builder (synthetic data only)."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from mrextrap.data.datasets import (
    AutoencoderDataset,
    LatentBetaDataset,
    LongitudinalDataset,
    get_autoencoder_dataloaders,
    get_latent_dataloaders,
    get_longitudinal_dataloaders,
    sort_df_by_age_from_baseline,
)
from mrextrap.data.images import build_t1_transform, clip_and_normalize_image
from mrextrap.data.splits import (
    assign_subjects,
    build_split_csvs,
    diagnosis_label,
    parse_subject_session,
    read_subject_list,
)


def _sid(site: int, rid: int, bids: bool = False) -> str:
    """Synthetic ADNI-style id, built at runtime so no id-shaped literal sits in the source."""
    return f"sub-ADNI{site:03d}S{rid:04d}" if bids else f"{site:03d}_S_{rid:04d}"


SHAPE = (32, 32, 32)
CROP = (16, 16, 16)
OFFSET = (4, 4, 4)


def _write_nifti(path, seed):
    rng = np.random.default_rng(seed)
    vol = rng.uniform(0, 1000, size=SHAPE).astype(np.float32)
    vol[:4] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(path))


def _bids_name(root, subject, session):
    sid = subject.replace("_S_", "S")
    return (
        root
        / f"sub-ADNI{sid}"
        / f"ses-{session}"
        / "anat"
        / f"sub-ADNI{sid}_ses-{session}_T1w.nii.gz"
    )


@pytest.fixture()
def toy_cohort(tmp_path):
    """Synthetic ADNIMERGE-like table + image index (fake ids of the right shape)."""
    rng = np.random.default_rng(0)
    rows, images = [], []
    for i in range(40):
        sid = f"{100 + i:03d}_S_{1000 + i:04d}"
        base_age = 60 + i * 0.5
        n_scans = int(rng.integers(1, 5))
        months = sorted(rng.choice([0, 6, 12, 24, 36], size=n_scans, replace=False))
        dx_seq = ["CN"] * n_scans if i % 3 == 0 else ["MCI"] * n_scans
        if i % 5 == 0:
            dx_seq[-1] = "Dementia"
        for m, dx in zip(months, dx_seq):
            vis = "bl" if m == 0 else f"m{m:02d}"
            rows.append({"PTID": sid, "VISCODE": vis, "AGE": base_age, "DX": dx})
            images.append({"filename": str(_bids_name(tmp_path / "img", sid, f"M{m:02d}"))})
    adnimerge = pd.DataFrame(rows)
    # an image whose visit has no metadata row, and a metadata row without image
    adnimerge = adnimerge.iloc[1:].reset_index(drop=True)
    adnimerge_csv = tmp_path / "adnimerge.csv"
    index_csv = tmp_path / "index.csv"
    adnimerge.to_csv(adnimerge_csv, index=False)
    pd.DataFrame(images).to_csv(index_csv, index=False)
    return adnimerge_csv, index_csv


# ------------------------------------------------------------------ images
def test_clip_and_normalize_range_and_quirks():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 100, size=(8, 8, 8))
    out = clip_and_normalize_image(img)
    assert out.min() == pytest.approx(-1.0)
    assert -1.0 <= out.max() < 1.0  # NOTE(paper): +0.001 keeps the max just below 1
    out15 = clip_and_normalize_image(img, normalize_to_1=False)
    assert out15.min() == pytest.approx(-1.5)
    mm = clip_and_normalize_image(img, method="minmax")
    assert mm.min() == 0.0 and mm.max() < 1.0
    # 0.997 quantile clip: an extreme outlier does not dominate the range
    img2 = img.copy()
    img2[0, 0, 0] = 1e6
    assert clip_and_normalize_image(img2).max() > 0.9


def test_transform_shape_and_range(tmp_path):
    path = tmp_path / "a.nii.gz"
    _write_nifti(path, 0)
    vol = build_t1_transform(OFFSET, CROP)(str(path))
    assert tuple(vol.shape) == (1, *CROP)
    assert vol.dtype == torch.float32
    assert vol.min() >= -1.0 and vol.max() <= 1.0


def test_transform_default_crop(tmp_path):
    path = tmp_path / "big.nii.gz"
    nib.save(
        nib.Nifti1Image(np.random.rand(176, 208, 160).astype(np.float32), np.eye(4)), str(path)
    )
    assert tuple(build_t1_transform()(str(path)).shape) == (1, 160, 192, 160)


def test_autoencoder_dataset(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"v{i}.nii.gz"
        _write_nifti(p, i)
        paths.append(str(p))
    ds = AutoencoderDataset(
        pd.DataFrame({"filename": paths}), cropping_offset=OFFSET, crop_size=CROP
    )
    assert len(ds) == 3
    assert tuple(ds[1].shape) == (1, *CROP)


# ------------------------------------------------------------------ longitudinal
def test_longitudinal_item_sorted_by_age(tmp_path):
    rows = []
    for k, m in enumerate([24, 0, 12]):  # deliberately unsorted
        p = tmp_path / f"s{k}.nii.gz"
        _write_nifti(p, k)
        rows.append(
            {
                "subject_id": "A",
                "filename": str(p),
                "age_from_baseline": m / 12,
                "AGE_filled": 70 + m / 12,
            }
        )
    df = sort_df_by_age_from_baseline(pd.DataFrame(rows))
    ds = LongitudinalDataset({"A": df}, cropping_offset=OFFSET, crop_size=CROP)
    item = ds[0]
    assert item["subject_id"] == "A"
    assert tuple(item["vols"].shape) == (3, 1, *CROP)
    assert torch.all(item["ages"][1:] > item["ages"][:-1])
    assert item["ages_abs"].tolist() == pytest.approx([70, 71, 72])
    # volume order follows the sorted ages: first vol is the m0 file (k=1)
    ref = build_t1_transform(OFFSET, CROP)(str(tmp_path / "s1.nii.gz"))
    assert torch.equal(item["vols"][0], ref)


def test_dataloaders_from_split_csvs(tmp_path, toy_cohort):
    adnimerge_csv, index_csv = toy_cohort
    idx = pd.read_csv(index_csv)
    for k, fn in enumerate(idx["filename"]):
        _write_nifti(type(tmp_path)(fn), k)
    paths = build_split_csvs(adnimerge_csv, index_csv, out_dir=tmp_path / "splits")
    cfg = OmegaConf.create(
        {
            "dataset": {
                "train_csv": str(paths["train"]),
                "val_csv": str(paths["val"]),
                "test_csv": str(paths["test"]),
                "cropping_offset": list(OFFSET),
                "crop_size": list(CROP),
            },
            "training": {"batch_size": 2},
        }
    )
    train, _, _ = get_autoencoder_dataloaders(cfg, num_workers=0)
    assert next(iter(train)).shape[1:] == (1, *CROP)
    cfg.training.batch_size = 1
    tr, _, _ = get_longitudinal_dataloaders(cfg, num_workers=0)
    batch = next(iter(tr))
    assert batch["vols"].shape[0] == 1 and batch["vols"].shape[2:] == (1, *CROP)


# ------------------------------------------------------------------ latents
def _latent_file(path, n_subjects, seed, with_nan=False, per_scan_ids=False):
    g = torch.Generator().manual_seed(seed)
    zs, ages, ages_abs, sids, betas = [], [], [], [], []
    for s in range(n_subjects):
        n = 2 + s % 3
        zs.append(torch.randn(n, 3, 4, 4, 4, generator=g) * (s + 1) + 5)
        betas.append(torch.randn(n, 3, 4, 4, 4, generator=g) * 0.1 - 1)
        ages.append(torch.arange(n, dtype=torch.float32).unsqueeze(0))
        ages_abs.append(65 + s + torch.arange(n, dtype=torch.float32).unsqueeze(0))
        sids.append([f"S{seed}_{s}"] * (n if per_scan_ids else 1))
    if with_nan:
        zs[0][0, 0, 0, 0, 0] = float("nan")
    d = {
        "zs": zs,
        "ages": ages,
        "ages_abs": ages_abs,
        "subject_ids": sids,
        "betas": betas,
        "dataset_type": "train",
    }
    torch.save(d, path)
    return d


def test_latent_dataset_standardisation_matches_manual(tmp_path):
    p = tmp_path / "train_latent_betas_l1.pt"
    d = _latent_file(p, 6, 0)
    ds = LatentBetaDataset(p)
    z_all = torch.cat(d["zs"])
    b_all = torch.cat(d["betas"])
    mean = z_all.mean(dim=(0, 2, 3, 4)).reshape(-1, 1, 1, 1)
    std = z_all.std(dim=(0, 2, 3, 4)).reshape(-1, 1, 1, 1)  # unbiased
    assert torch.allclose(ds.latent_means, mean)
    assert torch.allclose(ds.latent_stds, std)
    assert torch.allclose(ds.zs, (z_all - mean) / std)
    assert torch.allclose(
        ds.betas,
        (b_all - b_all.mean(dim=(0, 2, 3, 4)).reshape(-1, 1, 1, 1))
        / b_all.std(dim=(0, 2, 3, 4)).reshape(-1, 1, 1, 1),
    )
    assert len(ds) == len(ds.subject_ids) == z_all.shape[0]
    item = ds[0]
    assert set(item) == {"subject_id", "zs", "ages", "ages_abs", "betas"}
    assert torch.allclose(ds.destandardize_latents(ds.zs), z_all, atol=1e-4)
    stats = ds.normalization_stats()
    assert set(stats) == {
        "latent_means",
        "latent_stds",
        "beta_means",
        "beta_stds",
        "age_mu",
        "age_sigma",
    }
    ages_abs = torch.cat([a[0] for a in d["ages_abs"]])
    assert stats["age_mu"] == pytest.approx(ages_abs.mean().item())
    assert stats["age_sigma"] == pytest.approx(ages_abs.std().item())


def test_latent_val_test_reuse_train_stats(tmp_path):
    for name, seed in (("train", 0), ("val", 1), ("test", 2)):
        _latent_file(tmp_path / f"{name}_latent_betas_l1.pt", 5, seed)
    cfg = OmegaConf.create(
        {
            "dataset": {
                "latents_path_train": str(tmp_path / "train_latent_betas_l1.pt"),
                "latents_path_val": str(tmp_path / "val_latent_betas_l1.pt"),
                "latents_path_test": str(tmp_path / "test_latent_betas_l1.pt"),
            },
            "training": {"batch_size": 4},
        }
    )
    train, val, test = get_latent_dataloaders(cfg, num_workers=0)
    tr, va = train.dataset, val.dataset
    assert torch.equal(va.latent_means, tr.latent_means)
    assert torch.equal(test.dataset.beta_stds, tr.beta_stds)
    raw = torch.load(tmp_path / "val_latent_betas_l1.pt")
    manual = (torch.cat(raw["zs"]) - tr.latent_means) / tr.latent_stds
    assert torch.allclose(va.zs, manual)
    assert next(iter(train))["zs"].shape[1:] == (3, 4, 4, 4)


def test_latent_per_scan_subject_ids_accepted(tmp_path):
    p = tmp_path / "x_latent_betas_l1.pt"
    _latent_file(p, 4, 0, per_scan_ids=True)
    ds = LatentBetaDataset(p)
    assert len(ds.subject_ids) == len(ds)


def test_latent_nan_sequences_removed_but_subject_ids_assert(tmp_path):
    # NOTE(paper) Q13: subject_ids are not filtered in the non-Bayesian path -> assertion fires.
    p = tmp_path / "x_latent_betas_l1.pt"
    _latent_file(p, 4, 0, with_nan=True)
    with pytest.raises(AssertionError):
        LatentBetaDataset(p)


def test_bayesian_path_rejected(tmp_path):
    with pytest.raises(NotImplementedError):
        LatentBetaDataset(tmp_path / "bayesian_latents.pt")


# ------------------------------------------------------------------ splits
def test_diagnosis_label():
    assert diagnosis_label(pd.Series(["CN", "MCI", "Dementia"])) == "Dementia"
    assert diagnosis_label(pd.Series(["CN", np.nan, "MCI"])) == "MCI"
    assert diagnosis_label(["CN", "CN"]) == "CN"
    assert diagnosis_label(pd.Series([np.nan])) == "CN"


def test_parse_subject_session():
    assert parse_subject_session(
        f"/x/{_sid(2, 9002, bids=True)}/ses-M48/anat/{_sid(2, 9002, bids=True)}_ses-M48_T1w.nii.gz"
    ) == (
        _sid(2, 9002),
        "M48",
    )
    with pytest.raises(ValueError):
        parse_subject_session("/x/nothing.nii.gz")


def test_assign_subjects_matches_original_recipe():
    subjects = [f"s{i}" for i in range(200)]
    got = assign_subjects(subjects, seed=42)
    arr = np.array(subjects, dtype=object)
    np.random.seed(42)
    np.random.shuffle(arr)
    n = len(arr)
    assert got["train"] == list(arr[: int(0.90 * n)])
    assert got["val"] == list(arr[int(0.90 * n) : int(0.93 * n)])
    assert got["test"] == list(arr[int(0.93 * n) :])
    assert (len(got["train"]), len(got["val"]), len(got["test"])) == (180, 6, 14)


def test_build_split_csvs_synthetic(tmp_path, toy_cohort):
    adnimerge_csv, index_csv = toy_cohort
    out = tmp_path / "out"
    paths = build_split_csvs(adnimerge_csv, index_csv, out_dir=out, seed=42)
    dfs = {k: pd.read_csv(v) for k, v in paths.items()}
    assert set(dfs) == {"train", "val", "test"}
    sets = {k: set(v.subject_id) for k, v in dfs.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"])
    assert not (sets["val"] & sets["test"])
    n_total = len(sets["train"] | sets["val"] | sets["test"])
    assert n_total >= 38  # rows without metadata are dropped ("nona"), rest kept
    for df in dfs.values():
        assert list(df.columns) == [
            "subject_id",
            "session_id",
            "filename",
            "AGE",
            "DX",
            "DX_filled",
            "age_from_baseline",
            "baseline_age",
            "AGE_filled",
        ]
        assert not df[["DX_filled", "AGE_filled", "age_from_baseline"]].isna().any().any()
        assert np.allclose(df.AGE_filled, df.baseline_age + df.age_from_baseline)
        assert np.allclose(df.age_from_baseline, df.session_id.str[1:].astype(int) / 12)
    # deterministic in the seed
    paths2 = build_split_csvs(adnimerge_csv, index_csv, out_dir=tmp_path / "out2", seed=42)
    for k in paths:
        pd.testing.assert_frame_equal(dfs[k], pd.read_csv(paths2[k]))
    paths3 = build_split_csvs(adnimerge_csv, index_csv, out_dir=tmp_path / "out3", seed=7)
    assert set(pd.read_csv(paths3["test"]).subject_id) != sets["test"]


def test_build_split_csvs_explicit_subjects(tmp_path, toy_cohort):
    adnimerge_csv, index_csv = toy_cohort
    lst = tmp_path / "test_subjects.txt"
    lst.write_text(f"# comment\n{_sid(101, 1001)}\n{_sid(102, 1002)}\n")
    test = read_subject_list(lst)
    paths = build_split_csvs(
        adnimerge_csv,
        index_csv,
        test_subjects=test,
        val_subjects=[_sid(103, 1003)],
        out_dir=tmp_path / "o",
    )
    assert set(pd.read_csv(paths["test"]).subject_id) == set(test)
    assert set(pd.read_csv(paths["val"]).subject_id) == {_sid(103, 1003)}
    tr = set(pd.read_csv(paths["train"]).subject_id)
    assert not (tr & set(test)) and _sid(103, 1003) not in tr
    with pytest.raises(ValueError):
        build_split_csvs(
            adnimerge_csv, index_csv, test_subjects=[_sid(999, 9999)], out_dir=tmp_path / "p"
        )
    with pytest.raises(ValueError):
        build_split_csvs(
            adnimerge_csv,
            index_csv,
            test_subjects=test,
            val_subjects=test[:1],
            out_dir=tmp_path / "q",
        )


def test_dx_fill_and_age_rules(tmp_path):
    adnimerge = pd.DataFrame(
        {
            "PTID": [_sid(1, 1), _sid(1, 1), _sid(2, 2), _sid(2, 2)],
            "VISCODE": ["bl", "m12", "bl", "m12"],
            "AGE": [70.0, 70.0, 80.0, 80.0],
            "DX": ["MCI", np.nan, np.nan, "CN"],
        }
    )
    files = [
        _bids_name(tmp_path, s, v)
        for s, v in [
            (_sid(1, 1), "M00"),
            (_sid(1, 1), "M12"),
            (_sid(2, 2), "M00"),
            (_sid(2, 2), "M12"),
        ]
    ]
    adnimerge.to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame({"filename": [str(f) for f in files]}).to_csv(tmp_path / "i.csv", index=False)
    kw = {"test_subjects": [], "val_subjects": []}
    g = pd.read_csv(
        build_split_csvs(tmp_path / "a.csv", tmp_path / "i.csv", out_dir=tmp_path / "g", **kw)[
            "train"
        ]
    )
    # NOTE(paper): global ffill leaks the previous subject's DX into the next subject's first scan
    assert g.DX_filled.tolist() == ["MCI", "MCI", "MCI", "CN"]
    p = pd.read_csv(
        build_split_csvs(
            tmp_path / "a.csv",
            tmp_path / "i.csv",
            out_dir=tmp_path / "p",
            dx_fill="per_subject_ffill",
            **kw,
        )["train"]
    )
    assert p.subject_id.tolist() == [
        _sid(1, 1),
        _sid(1, 1),
        _sid(2, 2),
    ]  # leading NaN dropped
    assert g.AGE_filled.tolist() == pytest.approx([70, 71, 80, 81])


def test_missing_columns_raise(tmp_path):
    pd.DataFrame({"PTID": ["a"]}).to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame({"filename": ["x"]}).to_csv(tmp_path / "i.csv", index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        build_split_csvs(tmp_path / "a.csv", tmp_path / "i.csv", out_dir=tmp_path / "o")
