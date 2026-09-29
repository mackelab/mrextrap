"""Multi-volume evaluation (WP12): synthetic cohort, fake decoder, FakeSegmenter (CPU, offline)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from mrextrap.data.synthetic import make_fake_vols_csv
from mrextrap.evaluation import multivolume as mv
from mrextrap.prediction import GlobalPrior
from mrextrap.segmentation.volumes import (
    REGIONS,
    FakeSegmenter,
    predicted_region_volumes,
    real_region_volumes,
)


class FakeVAE(torch.nn.Module):
    """Decoder stand-in: 4x4x4 latent -> 8^3 volume with intensities depending on the latent."""

    def decode(self, z):
        return F.interpolate(torch.tanh(z[:, :1]), size=(8, 8, 8), mode="trilinear")


def _dx(sid):
    return ("CN", "MCI", "Dementia")[int(sid[-3:]) % 3]


@pytest.fixture
def cohort(synthetic_latent_dict, tmp_path):
    ld = synthetic_latent_dict["test"]
    real_dir = tmp_path / "real"
    for i, sids in enumerate(ld["subject_ids"]):
        make_fake_vols_csv(real_dir / sids[0] / "vols.csv", n_rows=len(sids), seed=i)
    earliest, latest = mv.prepare_data_unified(ld, _dx, 2, (1, 2, 3), 3)
    shape = ld["zs"][0].shape[1:]
    prior = GlobalPrior(
        0.05 * torch.randn(shape), 0.1 * torch.ones(shape), 0.01 * torch.ones(shape)
    )
    return {"ld": ld, "real_dir": real_dir, "earliest": earliest, "latest": latest, "prior": prior}


def _run(cohort, tmp_path, fmt="latest", prior=None, **kw):
    return mv.evaluate_conditioning(
        cohort[fmt], prior or cohort["prior"], FakeVAE(), FakeSegmenter(), cohort["real_dir"],
        tmp_path / f"out_{fmt}", raise_on_error=True, **kw,
    )  # fmt: skip


# ---------------------------------------------------------------------------- data preparation
def test_earliest_latest_invariants(cohort):
    earliest, latest = cohort["earliest"], cohort["latest"]
    assert len(earliest) == len(latest) >= 2
    for e, la in zip(earliest, latest, strict=True):
        assert e["subject_id"] == la["subject_id"]
        for k in ("zs", "ages", "ages_abs"):
            assert (e["prediction"][k] == la["prediction"][k]).all()
        assert e["prediction"]["indices"] == la["prediction"]["indices"]
        for k in ("zs", "ages", "ages_abs"):
            assert (e["conditioning"][k][:-1] == la["conditioning"][k][1:]).all()
        assert e["baseline"]["index"] == 0
        # latest baseline = last earliest conditioning scan; first scan is now a conditioning scan
        assert la["baseline"]["index"] == e["conditioning"]["indices"][-1]
        assert la["conditioning"]["indices"][0] == 0
        assert (la["baseline"]["z0"] == e["conditioning"]["zs"][-1:]).all()
        # predictions come after every conditioning scan
        assert min(e["prediction"]["indices"]) > max(e["conditioning"]["indices"])
        # conditioning within 0.5 y of 1, 2, 3 years after the first scan
        offs = (e["conditioning"]["ages"] - e["baseline"]["age0"]).tolist()
        assert all(abs(o - y) <= 0.5 for o, y in zip(offs, (1, 2, 3), strict=True))
    assert [s["subject_id"] for s in earliest] == sorted(s["subject_id"] for s in earliest)


def test_prepare_filters_subjects(synthetic_latent_dict):
    ld = synthetic_latent_dict["test"]
    # demanding a 100-year future span leaves nobody; a 30-year conditioning offset leaves nobody
    assert mv.prepare_data_unified(ld, _dx, min_future_years=100)[0] == []
    assert mv.prepare_data_unified(ld, _dx, conditioning_years=(30,))[0] == []


def test_dx_lookup_mapping_and_missing(synthetic_latent_dict):
    ld = synthetic_latent_dict["test"]
    lookup = {s[0]: "CN" for s in ld["subject_ids"]}
    e, _ = mv.prepare_data_unified(ld, lookup)
    assert {s["disease_status"] for s in e} == {"CN"}
    with pytest.raises(ValueError, match="not found"):
        mv.prepare_data_unified(ld, {})


# ---------------------------------------------------------------------------- evaluation
@pytest.mark.parametrize("fmt", ["earliest", "latest"])
def test_results_schema_and_layout(cohort, tmp_path, fmt):
    results = _run(cohort, tmp_path, fmt)
    out_dir = tmp_path / f"out_{fmt}"
    assert set(results) == set(REGIONS)
    on_disk = json.loads((out_dir / "results_var1.00.json").read_text())
    assert on_disk == json.loads(json.dumps(results))
    expected = 0
    for sd in cohort[fmt]:
        n_max = len(sd["conditioning"]["zs"])
        expected += (n_max + 1) * len(sd["prediction"]["indices"])
        for n in range(n_max + 1):
            seg_dir = out_dir / f"temp_seg_{sd['subject_id']}" / f"ncond{n}"
            assert (seg_dir / "vols.csv").is_file()
            assert (seg_dir / "tmp000_final.nii.gz").is_file()
    for region in REGIONS:
        rows = on_disk[region]
        assert len(rows) == expected
        n_cond, t_idx, mae, age, dx, sid = rows[0]
        assert isinstance(n_cond, int) and isinstance(t_idx, int)
        assert isinstance(mae, float) and mae >= 0
        assert isinstance(age, float)
        assert dx in ("CN", "MCI", "Dementia") and isinstance(sid, str)
        assert {r[0] for r in rows} == {0, 1, 2, 3}


def test_ncond0_equals_prior_trajectory(cohort, tmp_path):
    """n=0 MAEs equal an independent decode -> segment -> %TBV of ``mean * dt + z0``."""
    results = _run(cohort, tmp_path)
    sd = cohort["latest"][0]
    dt = (sd["prediction"]["ages"] - sd["baseline"]["age0"]).reshape(-1, 1, 1, 1, 1)
    z = cohort["prior"].beta_mean * dt + sd["baseline"]["z0"]
    x = torch.cat([FakeVAE().decode(zi[None]) for zi in z]).numpy()
    ref_dir = tmp_path / "ref"
    vols_csv = FakeSegmenter().segment([xi[0] for xi in x], ref_dir)
    real = real_region_volumes(cohort["real_dir"] / sd["subject_id"] / "vols.csv")
    pred = predicted_region_volumes(vols_csv, real["TBV"].iloc[0])
    for region in REGIONS:
        want = np.abs(
            real[f"{region}_pct_tbv"].iloc[sd["prediction"]["indices"]].values
            - pred[f"{region}_pct_tbv"].values
        )
        got = [r[2] for r in results[region] if r[5] == sd["subject_id"] and r[0] == 0]
        np.testing.assert_allclose(got, want)


def test_uninformative_observations_give_prior(cohort, tmp_path):
    """Posterior with obs_var -> inf equals the prior, so every n_cond has the n=0 MAEs."""
    p = cohort["prior"]
    flat = GlobalPrior(p.beta_mean, p.beta_std, torch.full_like(p.beta_mean, 1e18))
    results = _run(cohort, tmp_path, prior=flat)
    for region in REGIONS:
        by_n = {}
        for n, t, mae, _, _, sid in results[region]:
            by_n.setdefault(n, []).append((sid, t, mae))
        for n in (1, 2, 3):
            np.testing.assert_allclose(
                [m for *_, m in by_n[n]], [m for *_, m in by_n[0]], atol=1e-6
            )


def test_uses_last_n_conditioning_scans(cohort, tmp_path, monkeypatch):
    calls = []
    orig = mv.update_beta_posterior_closed_form

    def spy(mean, var, z0, deltaages, z_obs, obs_var):
        calls.append((z0, deltaages, z_obs))
        return orig(mean, var, z0, deltaages, z_obs, obs_var=obs_var)

    monkeypatch.setattr(mv, "update_beta_posterior_closed_form", spy)
    sd = cohort["latest"][0]
    mv.evaluate_conditioning(
        [sd], cohort["prior"], FakeVAE(), FakeSegmenter(), cohort["real_dir"], tmp_path / "o",
        raise_on_error=True,
    )  # fmt: skip
    assert len(calls) == 3
    for n, (z0, dta, z_obs) in enumerate(calls, start=1):
        assert (z0 == sd["baseline"]["z0"]).all()
        assert (z_obs == sd["conditioning"]["zs"][-n:]).all()
        assert torch.allclose(dta, sd["conditioning"]["ages"][-n:] - sd["baseline"]["age0"])
    # latest format: conditioning scans precede the baseline -> negative time offsets
    assert (calls[-1][1] < 0).all()


def test_uses_argument_not_global(cohort, tmp_path):
    other = mv.prepare_data_unified(cohort["ld"], _dx)[0][:1]
    res = _run({**cohort, "latest": other}, tmp_path)
    assert {r[5] for r in res[REGIONS[0]]} == {other[0]["subject_id"]}


def test_failed_subject_is_skipped_unless_raising(cohort, tmp_path):
    bad = dict(cohort["latest"][0], subject_id="MISSING")
    res = mv.evaluate_conditioning(
        [bad] + cohort["latest"][1:2], cohort["prior"], FakeVAE(), FakeSegmenter(),
        cohort["real_dir"], tmp_path / "o",
    )  # fmt: skip
    assert {r[5] for r in res[REGIONS[0]]} == {cohort["latest"][1]["subject_id"]}
    with pytest.raises(FileNotFoundError):
        mv.evaluate_conditioning(
            [bad], cohort["prior"], FakeVAE(), FakeSegmenter(), cohort["real_dir"],
            tmp_path / "o2", raise_on_error=True,
        )  # fmt: skip


def test_global_params_from_file_and_dict(cohort, tmp_path):
    p = cohort["prior"]
    d = {
        "global_beta_mean": p.beta_mean,
        "global_beta_std": p.beta_std,
        "global_obs_var": p.obs_var,
    }
    torch.save(d, tmp_path / "gp.pt")
    for src in (d, tmp_path / "gp.pt", p):
        q = mv.load_global_params(src)
        assert torch.equal(q.beta_mean, p.beta_mean) and torch.equal(q.obs_var, p.obs_var)


# ---------------------------------------------------------------------------- processing
def _hand_results():
    rows = []
    for n, scale in ((0, 1.0), (1, 0.5)):
        for t, (age, mae) in enumerate(((5.0, 0.1), (6.0, 0.2), (8.0, 0.4))):
            rows.append([n, t, mae * scale, age, "CN", "A"])
        for t, (age, mae) in enumerate(((3.0, 0.3), (5.0, 0.5))):
            rows.append([n, t, mae * scale, age, "MCI", "B"])
    return {"hippocampus": rows}


def test_process_results_unified_golden_latest():
    out = mv.process_results_unified(_hand_results(), "latest")
    a = out["hippocampus"]["A"]
    assert a["baseline_age"] == 5.0 and a["diagnosis"] == "CN"
    # median of ages from the subject's first prediction age = 1.0 -> near <= 1.0
    assert a["by_conditioning"][0] == {
        "near_term": [{"mae": 0.1, "prediction_time": 0.0}, {"mae": 0.2, "prediction_time": 1.0}],
        "far_term": [{"mae": 0.4, "prediction_time": 3.0}],
    }
    assert a["by_conditioning"][1]["far_term"] == [{"mae": 0.2, "prediction_time": 3.0}]
    b = out["hippocampus"]["B"]
    assert b["baseline_age"] == 3.0 and b["diagnosis"] == "MCI"
    assert b["by_conditioning"][0] == {
        "near_term": [{"mae": 0.3, "prediction_time": 0.0}],
        "far_term": [{"mae": 0.5, "prediction_time": 2.0}],
    }


def test_process_results_unified_golden_earliest():
    out = mv.process_results_unified(_hand_results(), "earliest")
    a, b = out["hippocampus"]["A"], out["hippocampus"]["B"]
    # only ages > 4 count; median of [5,6,8]*2 + [5]*2 = 5.5
    assert a["baseline_age"] == 0
    assert [p["prediction_time"] for p in a["by_conditioning"][0]["near_term"]] == [5.0]
    assert [p["prediction_time"] for p in a["by_conditioning"][0]["far_term"]] == [6.0, 8.0]
    assert b["by_conditioning"][1] == {
        "near_term": [{"mae": 0.25, "prediction_time": 5.0}],
        "far_term": [],
    }


def test_process_results_rejects_unknown_format():
    with pytest.raises(ValueError):
        mv.process_results_unified(_hand_results(), "middle")


def test_mae_across_conditioning_toy():
    proc = {
        r: mv.process_results_unified(_hand_results(), "latest")["hippocampus"] for r in REGIONS
    }
    d = mv.mae_across_conditioning(proc, "near_term")["hippocampus"]
    assert d["x"].tolist() == [0, 1]
    np.testing.assert_allclose(d["mean"], [np.mean([0.1, 0.2, 0.3]), np.mean([0.05, 0.1, 0.15])])
    np.testing.assert_allclose(d["sem"][0], np.std([0.1, 0.2, 0.3]) / np.sqrt(3))


# ---------------------------------------------------------------------------- progression shift
def _structured(subjects):
    """subjects: sid -> (dx, prior_mae, post_mae) with all points in the near term."""
    return {
        r: {
            sid: {
                "diagnosis": dx,
                "by_conditioning": {
                    0: {"near_term": [{"mae": pm, "prediction_time": 1.0}], "far_term": []},
                    3: {"near_term": [{"mae": qm, "prediction_time": 1.0}], "far_term": []},
                },
            }
            for sid, (dx, pm, qm) in subjects.items()
        }
        for r in REGIONS
    }


def test_mae_vs_progression_correlation_toy():
    shifts = {"a": 0.1, "b": 0.3, "c": 0.5, "d": 0.9}
    subjects = {s: ("CN", 9.0, 2.0 * x + 1.0) for s, x in shifts.items()}
    prog = {
        s: {r: {"conditioning": 0.2, "prediction": 0.2 - x, "full": 0.0} for r in REGIONS}
        for s, x in shifts.items()
    }
    out = mv.mae_vs_progression(_structured(subjects), prog)
    d = out["ventricle"]
    assert d["num_subjects"] == 4
    assert d["correlation"] == pytest.approx(1.0)
    assert d["slope"] == pytest.approx(2.0)
    np.testing.assert_allclose(d["x"], list(shifts.values()))  # absolute shift
    prior = mv.mae_vs_progression(_structured(subjects), prog, use_prior=True)["ventricle"]
    np.testing.assert_allclose(prior["y"], 9.0)  # prior MAE, independent of the shift
    # anti-correlated data
    subjects = {s: ("CN", 0.0, 5.0 - 2.0 * x) for s, x in shifts.items()}
    anti = mv.mae_vs_progression(_structured(subjects), prog)["cortex"]
    assert anti["correlation"] == pytest.approx(-1.0)
    # filters: diagnosis and missing subjects/terms
    only = mv.mae_vs_progression(_structured(subjects), prog, diagnosis_filter="MCI")
    assert only["cortex"]["num_subjects"] == 0 and only["cortex"]["correlation"] is None
    far = mv.mae_vs_progression(_structured(subjects), prog, term_type="far_term")
    assert far["cortex"]["num_subjects"] == 0


def test_compute_progression_rates_matches_polyfit(cohort):
    sd = cohort["earliest"][0]
    rates = mv.compute_progression_rates([sd], cohort["real_dir"])[sd["subject_id"]]
    real = real_region_volumes(cohort["real_dir"] / sd["subject_id"] / "vols.csv")
    idx = [0, *sd["conditioning"]["indices"]]
    ages = np.concatenate(
        [sd["baseline"]["age_abs0"].numpy(), sd["conditioning"]["ages_abs"].numpy()]
    )
    v = real["ventricle_pct_tbv"].values[idx]
    assert rates["ventricle"]["conditioning"] == pytest.approx(np.polyfit(ages, v, 1)[0])
    assert set(rates["ventricle"]) == {"conditioning", "prediction", "full"}
    # fake vols: ventricles grow with the scan index, the cortex shrinks
    assert rates["ventricle"]["full"] > 0 > rates["cortex"]["full"]


# ---------------------------------------------------------------------------- end to end
def test_subject_trajectories_and_evaluate_multivolume(cohort, tmp_path):
    summaries = mv.evaluate_multivolume(
        cohort["ld"], _dx, cohort["prior"], FakeVAE(), FakeSegmenter(), cohort["real_dir"],
        tmp_path / "mv", raise_on_error=True,
    )  # fmt: skip
    assert set(summaries) == {"earliest", "latest"}
    for fmt, s in summaries.items():
        assert s["n_subjects"] == len(cohort[fmt])
        assert (tmp_path / "mv" / fmt / "summary.json").is_file()
        json.dumps(s)  # serialisable
        assert set(s["mae_across_conditioning"]["near_term"]) == set(REGIONS)

    sd = cohort["latest"][0]
    t = mv.subject_trajectories(sd, cohort["real_dir"], tmp_path / "mv" / "latest", (1, 3))
    n_all = len(sd["conditioning"]["indices"]) + 1 + len(sd["prediction"]["indices"])
    assert len(t["all_ages"]) == len(t["all_idx"]) == n_all
    assert np.all(np.diff(t["all_ages"]) > 0)
    for r in REGIONS:
        assert len(t["real"][r]) == n_all
        assert len(t["prior"][r]) == len(t["pred_ages"]) == len(t["post"][3][r])
    # %TBV of predictions uses the first real scan's TBV
    real = real_region_volumes(cohort["real_dir"] / sd["subject_id"] / "vols.csv")
    seg = pd.read_csv(
        tmp_path / "mv" / "latest" / f"temp_seg_{sd['subject_id']}" / "ncond0" / "vols.csv"
    )
    assert seg.shape[0] - 1 == len(t["pred_ages"])
    assert real["TBV"].iloc[0] > 0


def test_diagnosis_lookup_from_csv(tmp_path):
    p = tmp_path / "split.csv"
    pd.DataFrame(
        {
            "subject_id": ["a", "a", "b", "b", "c"],
            "DX_filled": ["CN", "MCI", "CN", "CN", "Dementia"],
        }
    ).to_csv(p, index=False)
    assert mv.diagnosis_lookup_from_csv(p) == {"a": "MCI", "b": "CN", "c": "Dementia"}
