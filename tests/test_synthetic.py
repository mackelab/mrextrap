from itertools import pairwise

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

from mrextrap.data.synthetic import (
    SYNTHSEG_LABEL_NAMES,
    make_fake_vols_csv,
    make_phantom_volume,
    make_synthetic_cohort,
    make_synthetic_latent_dict,
    ventricle_volume,
)
from mrextrap.io import validate_latent_dict


def test_phantom_shape_dtype_range():
    v = make_phantom_volume((32, 32, 32), age=70, subject_seed=1)
    assert v.shape == (32, 32, 32) and v.dtype == np.float32
    assert v.min() >= 0 and 500 < v.max() < 1100 and (v == 0).any()


def test_phantom_deterministic_and_seed_dependent():
    a = make_phantom_volume((24,) * 3, 70, 3)
    assert np.array_equal(a, make_phantom_volume((24,) * 3, 70, 3))
    assert not np.array_equal(a, make_phantom_volume((24,) * 3, 70, 4))


def test_ventricle_volume_monotone_in_age():
    vols = [ventricle_volume(make_phantom_volume((32,) * 3, a, 0, 0.003)) for a in range(60, 90, 3)]
    assert all(b > a for a, b in pairwise(vols))


def test_phantom_survives_clip_and_normalize():
    v = make_phantom_volume((32,) * 3, 70, 0)
    q = np.clip(v, 0, np.quantile(v.reshape(-1), 0.997))
    n = 2 * ((q - q.min()) / (q.max() - q.min() + 0.001) - 0.5)
    assert -1 <= n.min() and n.max() <= 1


def test_cohort_files_and_schema(tmp_path):
    csvs = make_synthetic_cohort(tmp_path, n_subjects=6, scans_per_subject=(2, 5), seed=0)
    assert set(csvs) == {"train", "val", "test"}
    seen = set()
    for p in csvs.values():
        df = pd.read_csv(p)
        assert list(df.columns) == [
            "subject_id",
            "session_id",
            "filename",
            "AGE_filled",
            "age_from_baseline",
            "DX_filled",
        ]
        assert len(df) > 0 and df.DX_filled.isin(["CN", "MCI", "Dementia"]).all()
        sids = set(df.subject_id)
        assert not sids & seen  # split by subject
        seen |= sids
        for sid, g in df.groupby("subject_id"):
            assert 2 <= len(g) <= 5
            assert g.age_from_baseline.is_monotonic_increasing and g.age_from_baseline.iloc[0] == 0
            assert np.allclose(g.AGE_filled - g.AGE_filled.iloc[0], g.age_from_baseline)
        img = nib.load(df.filename.iloc[0])
        assert img.shape == (32, 32, 32)
    assert len(seen) == 6


def test_cohort_deterministic(tmp_path):
    a = make_synthetic_cohort(tmp_path / "a", n_subjects=4, seed=2)
    b = make_synthetic_cohort(tmp_path / "b", n_subjects=4, seed=2)
    for s in a:
        da, db = pd.read_csv(a[s]), pd.read_csv(b[s])
        assert np.allclose(da.AGE_filled, db.AGE_filled)
    x = nib.load(pd.read_csv(a["train"]).filename[0]).get_fdata()
    y = nib.load(pd.read_csv(b["train"]).filename[0]).get_fdata()
    assert np.array_equal(x, y)


def test_cohort_fixture(synthetic_cohort):
    assert all(p.exists() for p in synthetic_cohort.csvs.values())


def test_latent_dict_schema_and_determinism():
    d = make_synthetic_latent_dict(5, (3, 5), (4, 4, 4, 4), seed=1, include_betas=True)
    validate_latent_dict(d, require_betas=True)
    assert d["zs"][0].shape[1:] == (4, 4, 4, 4) and d["zs"][0].dtype == torch.float32
    assert d["ages"][0].shape == (1, d["zs"][0].shape[0])
    d2 = make_synthetic_latent_dict(5, (3, 5), (4, 4, 4, 4), seed=1, include_betas=True)
    assert all(torch.equal(a, b) for a, b in zip(d["zs"], d2["zs"]))


def test_latent_dict_is_linear_in_age():
    d = make_synthetic_latent_dict(2, (6, 6), seed=0, noise=0.0)
    z, age, beta = d["zs"][0], d["ages"][0][0], d["true_betas"][0]
    assert torch.allclose(z[3] - z[0], beta * (age[3] - age[0]), atol=1e-5)


def test_latent_fixture_supports_multivolume(synthetic_latent_dict):
    for s, d in synthetic_latent_dict.items():
        validate_latent_dict(d)
        assert d["dataset_type"] == s
        ok = [a[0, -1] - a[0, 0] >= 5 and a.shape[1] >= 5 for a in d["ages"]]
        assert sum(ok) >= 2


def test_fake_vols_csv_layout(fake_vols_csv):
    df = pd.read_csv(fake_vols_csv)
    assert df.columns[0] == "Unnamed: 0" and list(df.columns[1:]) == list(SYNTHSEG_LABEL_NAMES)
    assert len(df) == 5  # id row + 4 scans
    segs = df[1:].drop(columns=["Unnamed: 0"])
    assert (segs.dtypes == float).all()
    assert df.iloc[0, 1] == 2  # label-id row
    vent = [c for c in segs.columns if "ventricle" in c]
    assert segs[vent].sum(axis=1).is_monotonic_increasing


def test_fake_vols_csv_deterministic(tmp_path):
    a = make_fake_vols_csv(tmp_path / "a.csv", 3, seed=5).read_text()
    assert a == make_fake_vols_csv(tmp_path / "b.csv", 3, seed=5).read_text()


def test_tiny_configs_and_device(tiny_ae_cfg, tiny_unet_cfg, device):
    assert device == "cpu" and tiny_ae_cfg.model.z_channels == 4
    assert list(tiny_unet_cfg.model.image_size) == [4, 4, 4, 4]


@pytest.mark.parametrize("n", [3])
def test_small_cohort_all_splits_nonempty(tmp_path, n):
    for p in make_synthetic_cohort(tmp_path, n_subjects=n).values():
        assert len(pd.read_csv(p)) > 0
