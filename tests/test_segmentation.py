"""Segmentation tests: label tables, region volumes (Q9), backends and SynthSeg architecture.

All CPU, offline, no weights (random-initialised network, fake segmenter).
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

from mrextrap.data.synthetic import SYNTHSEG_LABEL_IDS, SYNTHSEG_LABEL_NAMES
from mrextrap.segmentation import (
    REGIONS,
    FakeSegmenter,
    SegmentationBackend,
    predicted_region_volumes,
    read_vols_csv,
    real_region_volumes,
    region_volumes,
    run_synthseg,
)
from mrextrap.segmentation.synthseg_net import (
    WEIGHTS_ENV_VAR,
    SynthSeg,
    load_synthseg,
    resolve_weights_path,
)
from mrextrap.segmentation.synthseg_postprocess import postprocess
from mrextrap.segmentation.synthseg_predict import (
    load_label_tables,
    preprocess,
    seg_output_path,
)

GROUPS = {
    "hippocampus": [17, 53],
    "ventricle": [4, 5, 14, 15, 43, 44],
    "cortex": [3, 8, 42, 47],
    "white matter": [2, 7, 41, 46],
}


def test_label_tables_match_real_synthseg():
    t = load_label_tables()
    assert len(t["unique_labels"]) == 32 and t["unique_labels"][0] == 0
    names = t["names"][t["unique_indices"]]
    assert names[0] == "background"
    assert tuple(t["unique_labels"][1:]) == SYNTHSEG_LABEL_IDS
    assert tuple(names[1:]) == SYNTHSEG_LABEL_NAMES


def _ids_for(region):
    return [i for i, n in zip(SYNTHSEG_LABEL_IDS, SYNTHSEG_LABEL_NAMES) if region in n]


@pytest.mark.parametrize("region", REGIONS)
def test_substring_groups_are_the_paper_label_groups(region):
    assert sorted(_ids_for(region)) == GROUPS[region]


def test_region_volumes_hand_computed(fake_vols_csv):
    raw = read_vols_csv(fake_vols_csv)
    assert list(raw.index) == [1, 2, 3, 4]  # label-id row dropped, index starts at 1
    assert "Unnamed: 0" not in raw.columns
    out = region_volumes(raw)
    by_id = dict(zip(SYNTHSEG_LABEL_IDS, SYNTHSEG_LABEL_NAMES))
    for region, ids in GROUPS.items():
        cols = [by_id[i] for i in ids]
        np.testing.assert_allclose(out[region], raw[cols].sum(axis=1))
        np.testing.assert_allclose(
            out[f"{region}_pct_tbv"], raw[cols].sum(axis=1) / raw.sum(axis=1).iloc[0] * 100
        )
    np.testing.assert_allclose(out["TBV"], raw.sum(axis=1))


def test_pct_tbv_first_row_vs_passed_tbv(fake_vols_csv):
    real = real_region_volumes(fake_vols_csv)
    tbv0 = real["TBV"].iloc[0]
    np.testing.assert_allclose(real["hippocampus_pct_tbv"], real["hippocampus"] / tbv0 * 100)
    pred = predicted_region_volumes(fake_vols_csv, tbv=1000.0)
    np.testing.assert_allclose(pred["ventricle_pct_tbv"], pred["ventricle"] / 1000.0 * 100)
    assert pred["TBV"].iloc[0] == tbv0  # TBV column itself is per-row


def test_region_volumes_duplicate_named_columns():
    # pandas mangles duplicate header names to "x.1"; substring matching still catches them
    df = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0]],
        columns=["left hippocampus", "left hippocampus.1", "x", "left cerebral cortex"],
    )
    out = region_volumes(df, regions=("hippocampus", "cortex"))
    assert out["hippocampus"].iloc[0] == 3.0 and out["cortex"].iloc[0] == 4.0
    assert out["TBV"].iloc[0] == 10.0


def test_fake_segmenter_layout_and_roundtrip(tmp_path):
    rng = np.random.RandomState(0)
    vols = [rng.rand(12, 12, 12).astype(np.float32) for _ in range(3)]
    seg = FakeSegmenter()
    assert isinstance(seg, SegmentationBackend)
    csv_path = seg.segment(vols, tmp_path)
    assert csv_path == tmp_path / "vols.csv"
    for i in range(3):
        assert (tmp_path / f"tmp{i:03d}_final.nii.gz").exists()
        s = nib.load(str(tmp_path / f"tmp{i:03d}_final_synthseg_robust.nii.gz")).get_fdata()
        assert s.shape == (12, 12, 12)
    raw = pd.read_csv(csv_path, header=None)
    assert raw.shape == (2 + 3, 1 + 31)
    assert raw.iloc[1, 1:].astype(int).tolist() == list(SYNTHSEG_LABEL_IDS)
    assert list(raw.iloc[0, 1:]) == list(SYNTHSEG_LABEL_NAMES)
    assert list(raw.iloc[2:, 0]) == ["tmp000_final", "tmp001_final", "tmp002_final"]
    df = read_vols_csv(csv_path)
    assert len(df) == 3
    total_fg = (
        nib.load(str(tmp_path / "tmp000_final_synthseg_robust.nii.gz")).get_fdata() > 0
    ).sum()
    assert df.iloc[0].sum() == total_fg
    assert region_volumes(df)["TBV"].iloc[0] == total_fg


def test_seg_output_naming(tmp_path):
    p = seg_output_path(tmp_path / "in" / "tmp003_final.nii.gz", tmp_path / "in", tmp_path / "out")
    assert p == str(tmp_path / "out" / "tmp003_final_synthseg_robust.nii.gz")


def test_run_synthseg_uses_predictor_with_legacy_files(tmp_path):
    calls = {}

    class StubPredictor:
        def segment_folder(self, image_dir, out_dir, vol_csv):
            calls["args"] = (str(image_dir), str(out_dir), str(vol_csv))

    x = np.random.rand(8, 8, 8).astype(np.float32)
    out = run_synthseg([x, x], tmp_path, predictor=StubPredictor())
    assert out == tmp_path / "vols.csv"
    assert calls["args"] == (str(tmp_path),) * 2 + (str(tmp_path / "vols.csv"),)
    img = nib.load(str(tmp_path / "tmp001_final.nii.gz"))
    np.testing.assert_array_equal(img.affine, np.eye(4))
    run_synthseg(x, tmp_path / "single", predictor=StubPredictor())
    assert (tmp_path / "single" / "tmp_final.nii.gz").exists()


def test_weights_resolution_errors_and_env(tmp_path, monkeypatch):
    monkeypatch.delenv(WEIGHTS_ENV_VAR, raising=False)
    with pytest.raises(FileNotFoundError, match="mrextrap_weights.zip"):
        resolve_weights_path(run_dir=tmp_path)
    f = tmp_path / "weights" / "synthseg_torch.pt"
    f.parent.mkdir()
    f.write_bytes(b"x")
    assert resolve_weights_path(run_dir=tmp_path) == f  # <run_dir>/weights/synthseg_torch.pt
    monkeypatch.setenv(WEIGHTS_ENV_VAR, str(f))
    assert resolve_weights_path() == f


def test_synthseg_forward_random_weights_and_state_dict_roundtrip(tmp_path):
    torch.manual_seed(0)
    net = SynthSeg().eval()
    with torch.no_grad():
        y = net(torch.rand(1, 1, 32, 32, 32))
    assert y.shape == (1, 32, 32, 32, 32)
    torch.testing.assert_close(y.sum(dim=1), torch.ones(1, 32, 32, 32))
    path = tmp_path / "synthseg_torch.pt"
    torch.save(net.state_dict(), path)
    net2 = load_synthseg(weights=path, device="cpu")
    assert not net2.training
    for k, v in net.state_dict().items():
        torch.testing.assert_close(net2.state_dict()[k], v)


def test_preprocess_pads_to_160_and_postprocess_pastes_back(tmp_path):
    x = np.random.RandomState(0).rand(24, 20, 28).astype(np.float32)
    p = tmp_path / "tmp000_final.nii.gz"
    nib.save(nib.Nifti1Image(x, np.eye(4)), str(p))
    im, aff, _h, _res, shape, pad_idx, crop_idx, _ = preprocess(str(p), crop=None, min_pad=160)
    assert im.shape == (1, 160, 160, 160, 1) and shape == [24, 20, 28] and crop_idx is None
    assert im.min() >= 0 and im.max() <= 1
    # postprocess: one-hot posteriors of a blob of label channel 3, on the padded grid
    labels = load_label_tables()["unique_labels"]
    post = np.zeros((1, 160, 160, 160, 32))
    post[..., 0] = 1.0
    sl = tuple(slice(pad_idx[i] + 4, pad_idx[i] + 10) for i in range(3))
    post[(0, *sl)] = 0.0
    post[(0, *sl, 3)] = 1.0
    seg, _post = postprocess(post, shape, pad_idx, crop_idx, labels, None, aff, None, None)
    assert seg.shape == (24, 20, 28)
    assert set(np.unique(seg)) == {0, int(labels[3])}
    assert (seg == labels[3]).sum() == 6**3
