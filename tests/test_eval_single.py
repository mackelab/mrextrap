"""WP11: single-volume evaluation, AE metrics, interpolation and statistics (CPU, FakeSegmenter)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from scipy import stats as sp_stats

from mrextrap.evaluation import stats
from mrextrap.evaluation.autoencoder import (
    generalized_dice,
    reconstruction_metrics,
)
from mrextrap.evaluation.interpolation import (
    interpolate_volumes,
    interpolation_r2,
    interpolation_volumes,
    linearity_r2,
)
from mrextrap.evaluation.single_volume import (
    count_baseline_not_first,
    evaluate_single_volume,
    load_daninet_pairs,
    parse_daninet_filename,
    predict_trajectory,
    prepare_daninet_eval,
)
from mrextrap.prediction import DiffusionPrior, GlobalPrior, UNetPrior
from mrextrap.segmentation.volumes import REGIONS, FakeSegmenter
from mrextrap.training.autoencoder import build_vae
from mrextrap.training.diffusion_prior import build_denoiser
from mrextrap.training.unet_prior import build_unet_prior


def _sid(site: int, rid: int, bids: bool = False) -> str:
    """Synthetic ADNI-style id, built at runtime so no id-shaped literal sits in the source."""
    return f"sub-ADNI{site:03d}S{rid:04d}" if bids else f"{site:03d}_S_{rid:04d}"


SHAPE = (32, 32, 32)


# ---------------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------------
@pytest.fixture
def test_latents(synthetic_latent_dict):
    return synthetic_latent_dict["test"]


@pytest.fixture
def pairs(test_latents):
    """Baseline = 2nd scan for the first subject (index > 0), first scan for the others."""
    out = []
    for i, sids in enumerate(test_latents["subject_ids"]):
        ages_abs = test_latents["ages_abs"][i][0]
        a = ages_abs[1 if i == 0 else 0]
        b = ages_abs[-1]
        out.append((sids[0], float(a), float(b)))
    return out


@pytest.fixture
def real_vols_dir(tmp_path, test_latents):
    """``<sid>/vols.csv`` with one FakeSegmenter row per scan of the (random) real volumes."""
    root = tmp_path / "test_real"
    rng = np.random.RandomState(0)
    seg = FakeSegmenter()
    for sids, z in zip(test_latents["subject_ids"], test_latents["zs"], strict=True):
        vols = [rng.rand(*SHAPE).astype(np.float32) for _ in range(len(z))]
        seg.segment(vols, root / sids[0])
    return root


@pytest.fixture
def vae(tiny_ae_cfg):
    return build_vae(tiny_ae_cfg.model).eval()


def _norm():
    return {
        "latent_means": torch.zeros(1, 4, 1, 1, 1),
        "latent_stds": torch.ones(1, 4, 1, 1, 1),
        "beta_means": torch.zeros(1, 4, 1, 1, 1),
        "beta_stds": torch.ones(1, 4, 1, 1, 1),
        "age_mu": torch.tensor(75.0),
        "age_sigma": torch.tensor(5.0),
    }


@pytest.fixture
def predictors(tiny_unet_cfg):
    tiny_unet_cfg.model.model_channels = 32  # GroupNorm(32) needs a multiple of 32
    g = torch.Generator().manual_seed(0)
    global_prior = GlobalPrior(0.1 * torch.randn(4, 4, 4, 4, generator=g), torch.ones(4, 4, 4, 4))
    unet = UNetPrior(build_unet_prior(tiny_unet_cfg.model).eval(), _norm())
    denoiser_cfg = tiny_unet_cfg.model.copy()
    diffusion = DiffusionPrior(build_denoiser(denoiser_cfg).eval(), _norm(), NT=3)
    return {"global": global_prior, "unet": unet, "diffusion": diffusion}


# ---------------------------------------------------------------------------------------------
# Pairs
# ---------------------------------------------------------------------------------------------
def test_parse_daninet_filename():
    name = f"79.063_1_2_1_ADNI_{_sid(6, 9003)}_MR_MT1__N3m_Br_2013_S187543_I369218.nii.png"
    assert parse_daninet_filename(name) == (_sid(6, 9003), 79.063)
    with pytest.raises(ValueError):
        parse_daninet_filename("no_subject_here.png")


def test_load_daninet_pairs_both_formats(tmp_path):
    p = tmp_path / "pairs.csv"
    p.write_text(
        "sid,baseline_age,target_age\nB_02,70.5,72.5\nA_01,60,64.25\n"
        f"71.1_1_1_1_ADNI_{_sid(128, 9004)}_MR_x.nii.png,76.8\n"
    )
    pairs = load_daninet_pairs(p)
    assert pairs == [(_sid(128, 9004), 71.1, 76.8), ("A_01", 60.0, 64.25), ("B_02", 70.5, 72.5)]
    p.write_text("a,1,2\na,3,4\n")
    with pytest.raises(ValueError):
        load_daninet_pairs(p)


def test_prepare_selects_closest_ages(test_latents, pairs):
    dx = {p[0]: "MCI" for p in pairs}
    subjects = prepare_daninet_eval(test_latents, pairs, dx)
    assert [s["subject_id"] for s in subjects] == [p[0] for p in pairs]
    assert subjects[0]["baseline"]["index"] == 1
    assert all(s["baseline"]["index"] == 0 for s in subjects[1:])
    n = len(test_latents["zs"][0])
    assert subjects[0]["prediction"]["indices"] == [n - 1]
    assert subjects[0]["disease_status"] == "MCI"
    assert count_baseline_not_first(subjects) == 1
    # subjects that are not listed are dropped
    assert len(prepare_daninet_eval(test_latents, pairs[:2])) == 2


# ---------------------------------------------------------------------------------------------
# Q1: delta t
# ---------------------------------------------------------------------------------------------
def test_delta_t_differs_only_when_baseline_index_positive(test_latents, pairs):
    subjects = prepare_daninet_eval(test_latents, pairs)
    prior = GlobalPrior(torch.ones(4, 4, 4, 4), torch.ones(4, 4, 4, 4))
    for s in subjects:
        z0 = s["baseline"]["z0"]
        dt_paper = predict_trajectory("global", s, prior, True) - z0
        dt_fixed = predict_trajectory("global", s, prior, False) - z0
        assert torch.allclose(dt_paper.flatten(1).mean(1), s["prediction"]["ages"].float())
        if s["baseline"]["index"] > 0:
            assert not torch.allclose(dt_paper, dt_fixed)
            expected = s["prediction"]["ages"] - s["baseline"]["age0"]
            assert torch.allclose(
                dt_fixed.flatten(1).mean(1), expected.float(), atol=1e-3, rtol=1e-3
            )
        else:
            assert torch.allclose(dt_paper, dt_fixed)


def test_diffusion_always_uses_target_minus_baseline(test_latents, pairs, predictors):
    s = prepare_daninet_eval(test_latents, pairs)[0]
    assert s["baseline"]["index"] > 0
    a = predict_trajectory("diffusion", s, predictors["diffusion"], True, num_samples=2)
    b = predict_trajectory("diffusion", s, predictors["diffusion"], False, num_samples=2)
    assert torch.allclose(a, b)


def test_diffusion_deterministic_for_fixed_seed(test_latents, pairs, predictors):
    s = prepare_daninet_eval(test_latents, pairs)[1]
    kw = {"num_samples": 2, "seed": 0}
    a = predict_trajectory("diffusion", s, predictors["diffusion"], **kw)
    b = predict_trajectory("diffusion", s, predictors["diffusion"], **kw)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------------------------
def test_three_methods_write_results_json(
    tmp_path, test_latents, pairs, predictors, vae, real_vols_dir
):
    dx = {p[0]: "CN" for p in pairs}
    subjects = prepare_daninet_eval(test_latents, pairs, dx)
    seg = FakeSegmenter()
    all_results = {}
    for method in ("global", "unet", "diffusion"):
        out_dir = tmp_path / "eval" / method
        res = evaluate_single_volume(
            method, subjects, predictors[method], vae, seg, real_vols_dir, out_dir, num_samples=2
        )
        saved = json.loads((out_dir / "results.json").read_text())
        assert set(saved) == set(REGIONS)
        for region in REGIONS:
            assert len(saved[region]) == len(subjects)
            assert [r[2] for r in saved[region]] == [s["subject_id"] for s in subjects]
            assert all(r[1] == "CN" and np.isfinite(r[0]) and r[0] >= 0 for r in saved[region])
        assert not list(out_dir.glob("temp_seg_*"))  # cleaned up
        all_results[method] = res
    assert len(all_results) == 3
    # results feed straight into Table 1
    df = stats.table1(all_results)
    assert df.shape[0] == len(REGIONS) and "unet_se" in df.columns


def test_mae_matches_manual_computation(
    tmp_path, test_latents, pairs, predictors, vae, real_vols_dir
):
    from mrextrap.segmentation.volumes import predicted_region_volumes, real_region_volumes

    subjects = prepare_daninet_eval(test_latents, pairs)[:1]
    res = evaluate_single_volume(
        "global",
        subjects,
        predictors["global"],
        vae,
        FakeSegmenter(),
        real_vols_dir,
        tmp_path / "o",
        cleanup=False,
    )
    s = subjects[0]
    z = predict_trajectory("global", s, predictors["global"])
    x = vae.decode(z[:1].float()).detach().numpy()
    seg_dir = tmp_path / "manual"
    csv = FakeSegmenter().segment([x[0, 0]], seg_dir)
    real = real_region_volumes(real_vols_dir / s["subject_id"] / "vols.csv")
    pred = predicted_region_volumes(csv, real["TBV"].iloc[0])
    idx = s["prediction"]["indices"][0]
    for region in REGIONS:
        expected = abs(real[f"{region}_pct_tbv"].iloc[idx] - pred[f"{region}_pct_tbv"].iloc[0])
        assert res[region][0][0] == pytest.approx(expected)


def test_failures_raise_unless_skipped(
    tmp_path, test_latents, pairs, predictors, vae, real_vols_dir
):
    subjects = prepare_daninet_eval(test_latents, pairs)
    with pytest.raises(FileNotFoundError):
        evaluate_single_volume(
            "global",
            subjects,
            predictors["global"],
            vae,
            FakeSegmenter(),
            tmp_path / "missing",
            tmp_path / "o",
        )
    res = evaluate_single_volume(
        "global",
        subjects,
        predictors["global"],
        vae,
        FakeSegmenter(),
        tmp_path / "missing",
        tmp_path / "o2",
        skip_failures=True,
    )
    assert all(len(v) == 0 for v in res.values())


# ---------------------------------------------------------------------------------------------
# Stats and Table 1
# ---------------------------------------------------------------------------------------------
def test_paired_tests_reproduce_scipy():
    rng = np.random.RandomState(1)
    a, b = rng.rand(50), rng.rand(50) + 0.05
    out = stats.paired_tests(a, b)
    t, p = sp_stats.ttest_rel(a, b)
    assert out["t"] == pytest.approx(t) and out["p_t"] == pytest.approx(p)
    d = a - b
    z = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    assert out["z"] == pytest.approx(z)
    assert out["p_z"] == pytest.approx(2 * (1 - sp_stats.norm.cdf(abs(z))))


def test_standard_error_and_z_test():
    x = np.array([1.0, 2.0, 4.0, 7.0])
    assert stats.standard_error(x) == pytest.approx(np.std(x, ddof=0) / 2)
    out = stats.z_test_vs_reported(1.0, 0.3, 1.5, 0.4)
    assert out["z"] == pytest.approx(-1.0)
    assert out["p_z"] == pytest.approx(2 * sp_stats.norm.sf(1.0))


def test_align_pair_by_subject_id():
    a = [(1.0, "CN", "s1"), (2.0, "CN", "s2"), (3.0, "MCI", "s3")]
    b = [(30.0, "MCI", "s3"), (10.0, "CN", "s1"), (20.0, "CN", "s2")]
    x, y = stats.align_pair(a, b)
    assert x.tolist() == [1.0, 2.0, 3.0] and y.tolist() == [10.0, 20.0, 30.0]
    with pytest.raises(ValueError):
        stats.align_pair(a, [(1.0, "CN", "other")] * 3)
    with pytest.raises(ValueError):
        stats.align_pair([1.0, 2.0], [1.0])


def _table_inputs():
    base = np.array([0.1, 0.2, 0.3, 0.4])
    return {
        "global": {r: base * (1 + i) for i, r in enumerate(REGIONS)},
        "unet": {r: base * (1 + i) + 0.05 for i, r in enumerate(REGIONS)},
        "diffusion": {r: [(v, "CN", f"s{k}") for k, v in enumerate(base * 2)] for r in REGIONS},
    }


GOLDEN_TABLE1 = """\
Region | Global | UNet | Diffusion | DANINet
Hippocampus | 0.250 ± 0.056 | 0.300 ± 0.056 | 0.500 ± 0.112 | 0.060 ± 0.003
Ventricles | 0.500 ± 0.112 | 0.550 ± 0.112 | 0.500 ± 0.112 | 0.257 ± 0.017
Grey matter | 0.750 ± 0.168 | 0.800 ± 0.168 | 0.500 ± 0.112 | 0.829 ± 0.046
White matter | 1.000 ± 0.224 | 1.050 ± 0.224 | 0.500 ± 0.112 | 0.829 ± 0.046
"""


def test_table1_golden_string():
    df = stats.table1(_table_inputs())
    assert stats.format_table1(df) == GOLDEN_TABLE1


def test_table1_latex_and_daninet_constants():
    df = stats.table1(_table_inputs())
    tex = stats.table1_latex(df)
    assert "\\begin{tabular}{lccccc}" not in tex and "\\begin{tabular}{lcccc}" in tex
    assert "$0.250 \\pm 0.056$" in tex
    assert stats.DANINET_REFERENCE["hippocampus"] == (0.060, 0.042)
    tests = stats.pairwise_tests(_table_inputs(), [("global", "unet")])
    assert list(tests["n"]) == [4] * 4
    dn = stats.daninet_tests(df)
    assert set(dn["method"]) == {"global", "unet", "diffusion"}


def test_run_table1_writes_files(tmp_path):
    results_dir = tmp_path / "run" / "eval" / "single_volume"
    for method, per_region in _table_inputs().items():
        (results_dir / method).mkdir(parents=True)
        rows = {
            r: [
                v if isinstance(v, tuple) else (float(v), "CN", f"s{k}") for k, v in enumerate(vals)
            ]
            for r, vals in per_region.items()
        }
        (results_dir / method / "results.json").write_text(json.dumps(rows))
    cfg = OmegaConf.create(
        {"paths": {"run_dir": str(tmp_path / "run")}, "eval": {"results_dir": str(results_dir)}}
    )
    out = stats.run_table1(cfg)
    assert (out / "table1.txt").read_text() == GOLDEN_TABLE1
    assert (out / "table1_tests.csv").exists() and (out / "table1.tex").exists()


# ---------------------------------------------------------------------------------------------
# Autoencoder metrics
# ---------------------------------------------------------------------------------------------
def _loader(n_subjects=2, n_scans=2, seed=0):
    rng = np.random.RandomState(seed)
    return [
        {
            "vols": torch.tensor(rng.rand(1, n_scans, 1, *SHAPE), dtype=torch.float32),
            "ages": torch.arange(n_scans).float().reshape(1, -1) * 1.5,
            "subject_id": [f"S{i}"],
        }
        for i in range(n_subjects)
    ]


def test_reconstruction_metrics(tmp_path, vae):
    loader = _loader()
    m = reconstruction_metrics(vae, loader, FakeSegmenter(), tmp_path, split="test")
    assert m["n_scans"] == 4 and m["n_subjects"] == 2
    assert -1.0 <= m["ssim"] <= 1.0 and 0.0 <= m["gdice"] <= 1.0
    assert (tmp_path / "test_real" / "S0" / "vols.csv").exists()
    assert (tmp_path / "test_recon" / "S1" / "vols.csv").exists()
    assert json.loads((tmp_path / "test_ae_metrics.json").read_text())["n_scans"] == 4
    # SSIM equals the direct computation
    from mrextrap.losses import SSIM

    ssim = SSIM(data_range=1.0, size_average=False, channel=1, spatial_dims=3)
    x = loader[0]["vols"][0]
    with torch.no_grad():
        xhat = torch.cat([vae(v[None], stochastic_sampling=False)[0] for v in x])
    assert m["ssim_per_scan"][:2] == pytest.approx(ssim(x, xhat).tolist(), abs=1e-6)


def test_gdice_identical_and_notebook_hippocampus_quirk():
    seg = np.zeros((16, 16, 16), dtype=np.float64)
    seg[2:6, 2:6, 2:6] = 17  # hippocampus
    seg[8:12, 8:12, 8:12] = 3  # cortex
    assert generalized_dice(seg, seg) == pytest.approx(1.0)
    assert generalized_dice(seg, seg, mode="notebook") == pytest.approx(1.0)
    other = seg.copy()
    other[8:12, 8:12, 8:12] = 0  # cortex lost, hippocampus intact
    # NOTE(paper): the notebook procedure only sees the hippocampus channel ...
    assert generalized_dice(seg, other, mode="notebook") == pytest.approx(1.0)
    # ... the default (correct) 5-region mode does not
    assert generalized_dice(seg, other, mode="regions") < 1.0
    moved = seg.copy()
    moved[2:6, 2:6, 2:6] = 0
    moved[3:7, 3:7, 3:7] = 17
    assert generalized_dice(seg, moved, mode="notebook") < 1.0


def test_gdice_regions_is_default_and_differs_from_notebook():
    from mrextrap.evaluation.autoencoder import per_region_dice

    seg = np.zeros((16, 16, 16), dtype=np.float64)
    seg[2:6, 2:6, 2:6] = 17  # hippocampus
    seg[8:12, 8:12, 8:12] = 3  # cortex
    seg[2:6, 8:12, 8:12] = 4  # ventricle
    other = seg.copy()
    other[8:12, 8:12, 8:12] = 0  # cortex lost
    other[2:6, 8:12, 8:12] = 5  # ventricle labels differ but still ventricle
    assert generalized_dice(seg, other) == generalized_dice(seg, other, mode="regions")
    assert generalized_dice(seg, other, mode="notebook") == pytest.approx(1.0)
    assert 0.3 < generalized_dice(seg, other, mode="regions") < 0.9
    dice = per_region_dice(seg, other)
    assert dice["hippocampus"] == pytest.approx(1.0) and dice["ventricle"] == pytest.approx(1.0)
    assert dice["cortex"] == 0.0 and np.isnan(dice["amygdala"])
    # a wrong hippocampus is penalised by both modes
    moved = seg.copy()
    moved[2:6, 2:6, 2:6] = 0
    moved[3:7, 3:7, 3:7] = 17
    assert generalized_dice(seg, moved, mode="regions") < 1.0
    with pytest.raises(ValueError):
        generalized_dice(seg, seg, mode="nope")


# ---------------------------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------------------------
def test_interpolate_volumes_modes(vae):
    g = torch.Generator().manual_seed(0)
    xs, xe = torch.rand(1, 1, *SHAPE, generator=g), torch.rand(1, 1, *SHAPE, generator=g)
    alphas = torch.linspace(0, 1, 4)
    v = interpolate_volumes(vae, xs, xe, alphas, "voxel")
    assert v.shape == (4, 1, *SHAPE)
    assert torch.allclose(v[0], xs) and torch.allclose(v[-1], xe)
    assert torch.allclose(v[1], xs + alphas[1] * (xe - xs))
    lat = interpolate_volumes(vae, xs, xe, alphas, "latent")
    with torch.no_grad():
        z0, z1 = vae.encode(xs)[0], vae.encode(xe)[0]
        assert torch.allclose(lat[0:1], vae.decode(z0), atol=1e-3, rtol=1e-3)
        assert torch.allclose(lat[-1:], vae.decode(z1), atol=1e-3, rtol=1e-3)
        assert torch.allclose(
            lat[2:3], vae.decode(z0 + alphas[2] * (z1 - z0)), atol=1e-3, rtol=1e-3
        )
    with pytest.raises(ValueError):
        interpolate_volumes(vae, xs, xe, alphas, "bogus")


@pytest.mark.parametrize("mode", ["voxel", "latent"])
def test_interpolation_volumes_segments_expected_volumes(tmp_path, vae, mode):
    loader = _loader(n_subjects=2, n_scans=3)
    res = interpolation_volumes(vae, loader, FakeSegmenter(), tmp_path, mode=mode, n_alpha=5)
    assert set(res) == {"S0", "S1"}
    assert res["S0"]["age_gap"] == pytest.approx(3.0)
    for region in REGIONS:
        pc = res["S0"]["pct_change"][region]
        assert pc.shape == (5,) and pc[0] == 0.0
    # the segmented volumes are the interpolated ones
    x = loader[0]["vols"][0]
    xi = interpolate_volumes(vae, x[:1], x[-1:], torch.linspace(0, 1, 5), mode)
    csv = FakeSegmenter().segment([v[0].numpy() for v in xi], tmp_path / "ref")
    assert (tmp_path / "S0" / "vols.csv").read_text() == csv.read_text()


def test_interpolation_skips_single_scan_subjects(tmp_path, vae):
    res = interpolation_volumes(vae, _loader(n_scans=1), FakeSegmenter(), tmp_path)
    assert res == {}


def test_linearity_r2_and_filters():
    a = np.linspace(0, 1, 10)
    assert linearity_r2(3 * a + 1) == pytest.approx(1.0)
    noisy = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1.0])
    assert linearity_r2(noisy) < 0.2
    results = {
        "a": {"age_gap": 2.0, "pct_change": {r: 2 * a for r in REGIONS}},
        "b": {"age_gap": 5.0, "pct_change": {r: 2 * a for r in REGIONS}},  # too long a gap
        "c": {"age_gap": 1.0, "pct_change": {r: noisy for r in REGIONS}},
    }
    r2 = interpolation_r2(results, max_age_gap=4.1)
    assert len(r2["hippocampus"]) == 2
    assert len(interpolation_r2(results, max_age_gap=4.1, min_r2=0.7)["hippocampus"]) == 1
    assert len(interpolation_r2(results, max_age_gap=None)["hippocampus"]) == 3
