#!/usr/bin/env python
"""Validate the MRExtrap release code against the cached legacy (paper) artefacts on the cluster.

Reads the run directory produced by ``import_legacy_artifacts.py`` (RELEASE_PLAN 2.1) and writes
``validation_report.{json,md}`` (PASS/FAIL table with tolerances) plus, if any Q2 check ran,
``report_q2_obs_var.{json,md}``. Reports contain counts and aggregate statistics only; the writer
refuses any text that looks like a subject id, and rebuilt split CSVs live in a temp dir that is
deleted afterwards.

Checks (RELEASE_PLAN 6 plus author requests)
  cpu:       V2 global params, V4 Table 1, V5a Q1 baseline count, V7 Fig. 3 data, V9 Fig. 5 stats,
             V12 normalisation, S1 split subject sets, Q2a obs_var (original vs intended formula)
  gpu:       V0 converted checkpoints, V1 re-encoding, V3 beta re-fit, V11 SynthSeg, V5 Global
             single-volume, V8 Global multi-volume (earliest+latest), Q2b multi-volume with both obs_var
  gpu-long:  V6 UNet (and diffusion with --include-diffusion) single-volume, V10 interpolation modes

``--expected FILE`` (JSON, all keys optional) supplies reference numbers that are not in the repo::

    {"table1": {"global": {"hippocampus": [mae, se], ...}, "unet": {...}, "diffusion": {...}},
     "p_values": {"hippocampus": {"unet_vs_global": 0.0429, "unet_vs_diffusion": ..., "diffusion_vs_global": ...}},
     "fig3_r": {"hippocampus": 0.33, ...} | [0.33, 0.73, 0.68, 0.71],
     "fig3_n_subjects": 32,
     "normalization": {"latent_means": [...], "latent_stds": [...], "beta_means": [...], "beta_stds": [...],
                       "age_mu": 0.0, "age_sigma": 0.0}}

Examples (cluster; RUN=run_dir from the import tool)::

    python tools/validate_against_legacy.py --run-dir $RUN --checks cpu --daninet-pairs PAIRS.csv \\
        --expected expected.json --adnimerge-csv ADNIMERGE.csv --image-index-csv INDEX.csv --legacy-root .
    python tools/validate_against_legacy.py --run-dir $RUN --checks gpu --device cuda --ae-config configs/autoencoder.yaml \\
        --config configs/autoencoder.yaml --converted-dir $REL --ae-legacy-dir exp/train_autoencoder_older8/epoch_124/epoch_124 \\
        --daninet-pairs PAIRS.csv --n-eval 32
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SUBJECT_ID_PATTERN, markdown_table, write_report

REGIONS = ("hippocampus", "ventricle", "cortex", "white matter")
CPU_CHECKS = ("V2", "V4", "V5a", "V7", "V9", "V12", "S1", "Q2a")
GPU_CHECKS = ("V0", "V1", "V3", "V11", "V5", "V8", "Q2b")
GPU_LONG_CHECKS = ("V6", "V10")
FIG3_R_PAPER = (0.33, 0.73, 0.68, 0.71)
STATUS_FAILING = ("FAIL", "ERROR")


def scrub(text: str) -> str:
    """Replace anything shaped like a subject id (error messages may contain file paths)."""
    return SUBJECT_ID_PATTERN.sub("<sid>", str(text))


def result(status: str, message: str = "", tolerance: str = "", **data) -> dict:
    return {"status": status, "message": scrub(message), "tolerance": tolerance, "data": data}


# --------------------------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------------------------
class Ctx:
    """Arguments plus lazily loaded, cached shared objects."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run_dir = Path(args.run_dir)
        self.out_dir = Path(args.out_dir) if args.out_dir else self.run_dir / "validation"
        self.device = args.device
        self.cache: dict = {}
        self.expected = json.loads(Path(args.expected).read_text()) if args.expected else {}

    def path(self, *parts) -> Path:
        return self.run_dir.joinpath(*parts)

    def need(self, *parts) -> Path:
        p = self.path(*parts)
        if not p.exists():
            raise FileNotFoundError(f"missing {p.relative_to(self.run_dir)} in run_dir")
        return p

    def latents(self, split: str) -> dict:
        key = f"latents_{split}"
        if key not in self.cache:
            from mrextrap.io import load_latent_dict

            for name in (f"{split}_latent_betas_l1.pt", f"{split}_zs_ages.pt"):
                if self.path("latents", name).is_file():
                    self.cache[key] = load_latent_dict(self.path("latents", name))
                    break
            else:
                raise FileNotFoundError(f"no latent file for split {split} in run_dir/latents")
        return self.cache[key]

    def betas_train(self) -> dict:
        d = self.latents("train")
        if "betas" not in d:
            raise ValueError("train latent file has no betas (need latent_betas_l1.pt)")
        return d

    def vae(self):
        if "vae" not in self.cache:
            from omegaconf import OmegaConf

            from mrextrap.encoding import load_autoencoder

            model_cfg = OmegaConf.load(self.args.ae_config).model if self.args.ae_config else None
            path = Path(self.args.ae_path) if self.args.ae_path else self.path("autoencoder")
            self.cache["vae"] = load_autoencoder(path, device=self.device, model_cfg=model_cfg)
        return self.cache["vae"]

    def segmenter(self):
        if "seg" not in self.cache:
            from mrextrap.segmentation.volumes import SynthSegSegmenter

            w = self.args.synthseg_weights or self.path("weights", "synthseg_torch.pt")
            self.cache["seg"] = SynthSegSegmenter(
                weights=w if Path(w).is_file() else None, run_dir=self.run_dir, device=self.device
            )
        return self.cache["seg"]

    def dx_lookup(self) -> dict:
        from mrextrap.evaluation.multivolume import diagnosis_lookup_from_csv

        return diagnosis_lookup_from_csv(self.need("splits", "test.csv"))

    def real_dir(self, split="test") -> Path:
        return self.need("volumes", f"{split}_real")

    def daninet_subjects(self) -> list[dict]:
        if "dsubs" not in self.cache:
            from mrextrap.evaluation.single_volume import load_daninet_pairs, prepare_daninet_eval

            if not self.args.daninet_pairs:
                raise ValueError("--daninet-pairs is required for this check")
            pairs = load_daninet_pairs(self.args.daninet_pairs)
            self.cache["dsubs"] = prepare_daninet_eval(
                self.latents("test"), pairs, self.dx_lookup()
            )
        return self.cache["dsubs"]


# --------------------------------------------------------------------------------------------
# Numerical helpers
# --------------------------------------------------------------------------------------------
def tensor_diff_summary(a: torch.Tensor, b: torch.Tensor) -> dict:
    d = (a.double() - b.double()).abs()
    return {
        "equal": bool(torch.equal(a, b)),
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "max_rel_diff": float((d / b.double().abs().clamp_min(1e-12)).max()),
    }


def intended_residuals(betas, zs, ages_abs) -> torch.Tensor:
    """Per-subject residual of the linear model with the *intended* per-scan alignment.

    Same subject filter (>= 4 scans) and N/(N-1) scaling as ``progression.compute_residuals``, but the
    age term is aligned with the scan it belongs to: for reference ``r`` and scan ``j``
    ``res[r, j] = (z_j - z_r) - (age_j - age_r) * beta_r`` (shape (N, ...) instead of the original
    unintended (N, N, ...) broadcast), averaged over the references ``r``.
    """
    out = []
    for si in range(len(betas)):
        n = len(betas[si])
        if n < 4:
            continue
        z, b = zs[si], betas[si]
        age = ages_abs[si][0].reshape(-1, 1, 1, 1, 1)
        per_ref = []
        for r in range(n):
            res = (z - z[r : r + 1]) - (age - age[r : r + 1]) * b[r : r + 1]
            per_ref.append(res * n / (n - 1))
        out.append(torch.stack(per_ref).mean(dim=0))
    return torch.cat(out, dim=0)


def intended_obs_var(betas, zs, ages_abs) -> torch.Tensor:
    """``var`` over subject-averaged residuals, computed from ``intended_residuals``."""
    return intended_residuals(betas, zs, ages_abs).var(dim=0)


def summarize_var(v: torch.Tensor) -> dict:
    v = v.double()
    return {
        "shape": list(v.shape),
        "mean": float(v.mean()),
        "median": float(v.median()),
        "min": float(v.min()),
        "max": float(v.max()),
        "per_channel_mean": [float(x) for x in v.mean(dim=tuple(range(1, v.ndim)))],
    }


def mae_table(results: dict) -> dict:
    """``{region: {n_cond: mean MAE}}`` from ``{region: [[n_cond, t, mae, age, dx, sid], ...]}``."""
    out = {}
    for region, rows in results.items():
        by: dict[int, list[float]] = {}
        for row in rows:
            by.setdefault(int(row[0]), []).append(float(row[2]))
        out[region] = {n: float(np.mean(v)) for n, v in sorted(by.items())}
    return out


def compare_multivolume(new: dict, ref: dict) -> dict:
    """Compare two multi-volume result dicts on shared (region, sid, n_cond, t_idx) keys."""
    diffs, n_new, n_ref = [], 0, 0
    for region, new_rows in new.items():
        ref_map = {(r[5], int(r[0]), int(r[1])): float(r[2]) for r in ref.get(region, [])}
        n_ref += len(ref_map)
        for r in new_rows:
            n_new += 1
            key = (r[5], int(r[0]), int(r[1]))
            if key in ref_map:
                diffs.append(abs(float(r[2]) - ref_map[key]))
    d = np.array(diffs) if diffs else np.array([np.nan])
    return {"n_new": n_new, "n_reference": n_ref, "n_matched": len(diffs),
            "max_abs_diff": float(np.nanmax(d)), "mean_abs_diff": float(np.nanmean(d))}  # fmt: skip


# --------------------------------------------------------------------------------------------
# CPU checks
# --------------------------------------------------------------------------------------------
def check_v2(ctx: Ctx) -> dict:
    from mrextrap.progression import compute_global_prior

    ref = torch.load(ctx.need("priors", "global_params.pt"), map_location="cpu")
    new = compute_global_prior(ctx.betas_train())
    per_key = {
        k: tensor_diff_summary(new[k], ref[k])
        for k in ("global_beta_mean", "global_beta_std", "global_obs_var")
    }
    if all(v["equal"] for v in per_key.values()):
        return result("PASS", "torch.equal for mean/std/obs_var", "exact equality", keys=per_key)
    near = all(
        torch.allclose(new[k].double(), ref[k].double(), rtol=ctx.args.rtol_params, atol=0)
        for k in per_key
    )
    return result(
        "NEAR" if near else "FAIL",
        "not bit-identical",
        f"rtol {ctx.args.rtol_params}",
        keys=per_key,
    )


def _table1_from_legacy(ctx: Ctx) -> dict:
    from mrextrap.evaluation.stats import load_results

    out = {}
    for method in ("global", "unet"):
        p = ctx.path("eval", "single_volume", method, "results.json")
        if p.is_file():
            out[method] = load_results(p)
    if "global" in out and not any(len(v) for v in out["global"].values()):
        # The legacy Global results.json is empty; its per-subject segmentations are intact.
        out["global"] = global_rows_from_segs(ctx)
    return out


def global_rows_from_segs(ctx: Ctx) -> dict[str, list]:
    """Per-subject Global rows ``[mae, dx, subject_id]`` from cached ``temp_seg_<sid>/vols.csv``.

    Used when the legacy ``exp_eval/global_prior/single_volume/results.json`` has empty region lists;
    mirrors :func:`diffusion_maes` (target = first prediction index, %TBV of the first real scan).
    """
    from mrextrap.segmentation.volumes import read_vols_csv, real_region_volumes, region_volumes

    dx = ctx.dx_lookup()
    out = {r: [] for r in REGIONS}
    for s in ctx.daninet_subjects():
        sid = s["subject_id"]
        real = real_region_volumes(ctx.real_dir() / sid / "vols.csv")
        df = read_vols_csv(
            ctx.need("eval", "single_volume", "global", f"temp_seg_{sid}", "vols.csv")
        )
        pred = region_volumes(df, tbv=real["TBV"].iloc[0])
        for r in REGIONS:
            real_v = real[f"{r}_pct_tbv"].iloc[s["prediction"]["indices"][0]]
            out[r].append(
                [
                    float(abs(real_v - pred[f"{r}_pct_tbv"].iloc[0])),
                    dx.get(sid) if isinstance(dx, dict) else None,
                    sid,
                ]
            )
    return out


def diffusion_maes(ctx: Ctx) -> dict[str, list[float]]:
    """Per-subject Diffusion MAE per region from the cached ``vols.csv`` (legacy eval L1640-1700).

    NOTE(paper): row ``i`` of the diffusion ``vols.csv`` belongs to the ``i``-th subject of the test
    latent dict; %TBV uses the TBV of the subject's first real scan.
    """
    from mrextrap.segmentation.volumes import read_vols_csv, real_region_volumes, region_volumes

    subjects = ctx.daninet_subjects()
    n_all = len(ctx.latents("test")["subject_ids"])
    if len(subjects) != n_all:
        raise ValueError(
            f"DANINet pairs cover {len(subjects)} of {n_all} test subjects; row mapping ambiguous"
        )
    df = read_vols_csv(ctx.need("eval", "single_volume", "diffusion", "legacy_segs", "vols.csv"))
    out = {r: [] for r in REGIONS}
    for i, s in enumerate(subjects):
        real = real_region_volumes(ctx.real_dir() / s["subject_id"] / "vols.csv")
        pred = region_volumes(df, tbv=real["TBV"].iloc[0])
        for r in REGIONS:
            real_v = real[f"{r}_pct_tbv"].iloc[s["prediction"]["indices"][0]]
            out[r].append(float(abs(real_v - pred[f"{r}_pct_tbv"].iloc[i])))
    return out


def check_v4(ctx: Ctx) -> dict:
    from mrextrap.evaluation.stats import align_pair, maes, paired_tests, standard_error

    exp = ctx.expected
    res = _table1_from_legacy(ctx)
    if not res:
        raise FileNotFoundError("no eval/single_volume/{global,unet}/results.json in run_dir")
    per_method = {m: {r: maes(rows) for r, rows in d.items()} for m, d in res.items()}
    notes = []
    try:
        per_method["diffusion"] = {r: np.array(v) for r, v in diffusion_maes(ctx).items()}
    except Exception as e:  # noqa: BLE001 diffusion row needs pairs + real vols; report and continue
        notes.append(f"diffusion row skipped: {scrub(e)}")
    table = {
        m: {r: [float(np.mean(x)), standard_error(x), len(x)] for r, x in d.items()}
        for m, d in per_method.items()
    }
    tol = ctx.args.tol_table
    mismatches, compared = [], 0
    for m, regs in exp.get("table1", {}).items():
        for r, (mae, se) in regs.items():
            if m in table and r in table[m]:
                compared += 1
                if abs(table[m][r][0] - mae) > tol or abs(table[m][r][1] - se) > tol:
                    mismatches.append(f"{m}/{r}")
    tests = {}
    pairs = (("unet", "diffusion"), ("global", "unet"), ("global", "diffusion"))
    p_mismatch, p_compared = [], 0
    for a, b in pairs:
        if a in per_method and b in per_method:
            for r in REGIONS:
                if a == "diffusion" or b == "diffusion":
                    x, y = per_method[a][r], per_method[b][r]  # latent order on both sides
                else:
                    x, y = align_pair(res[a][r], res[b][r])
                t = paired_tests(x, y)
                tests[f"{a}_vs_{b}/{r}"] = t
                ref_p = exp.get("p_values", {}).get(r, {}).get(f"{a}_vs_{b}")
                if ref_p is not None:
                    p_compared += 1
                    if abs(t["p_z"] - ref_p) > ctx.args.tol_p:
                        p_mismatch.append(f"{a}_vs_{b}/{r}")
    if not exp:
        status, msg = "INFO", "no --expected given: numbers reported only"
    elif not compared and not p_compared:
        status, msg = "INFO", "--expected has no table1/p_values entries matching the results"
    else:
        status = "PASS" if not mismatches and not p_mismatch else "FAIL"
        msg = f"{compared} table cells and {p_compared} p-values compared; mismatches: {mismatches + p_mismatch}"
    return result(status, "; ".join([msg] + notes), f"|delta| <= {tol} (mean, SE); p_z <= {ctx.args.tol_p}",
                  table1=table, paired_tests=tests)  # fmt: skip


def check_v5a(ctx: Ctx) -> dict:
    from mrextrap.evaluation.single_volume import count_baseline_not_first

    subjects = ctx.daninet_subjects()
    n_aff = count_baseline_not_first(subjects)
    return result("INFO", f"{n_aff} of {len(subjects)} DANINet subjects have baseline index > 0 (Q1)",
                  "", n_subjects=len(subjects), n_baseline_not_first=n_aff)  # fmt: skip


def check_v7(ctx: Ctx) -> dict:
    from mrextrap.evaluation import multivolume as mv

    latent, dx = ctx.latents("test"), ctx.dx_lookup()
    earliest, latest = mv.prepare_data_unified(latent, dx, 2, (1, 2, 3), 3)
    prepared = {"earliest": earliest, "latest": latest}
    data, processed = {}, {}
    for fmt in ("earliest", "latest"):
        p = ctx.path("eval", "multivolume", fmt, "results_var1.00.json")
        res = json.loads(p.read_text())
        processed[fmt] = mv.process_results_unified(res, fmt)
        n_subj = {r: len({row[5] for row in rows}) for r, rows in res.items()}
        data[fmt] = {"n_subjects_in_results": n_subj, "n_subjects_prepared": len(prepared[fmt]),
                     "mean_mae_by_ncond": mae_table(res)}  # fmt: skip
    progression = mv.compute_progression_rates(earliest, ctx.real_dir())
    shift = mv.mae_vs_progression(
        processed["earliest"], progression
    )  # NOTE(paper) Q8: panel b = earliest
    r_vals = {r: shift[r]["correlation"] for r in REGIONS}
    data["fig3b_r_earliest"] = r_vals
    data["fig3b_n_subjects"] = {r: shift[r]["num_subjects"] for r in REGIONS}
    exp_r = ctx.expected.get("fig3_r", list(FIG3_R_PAPER))
    exp_vals = sorted(exp_r.values() if isinstance(exp_r, dict) else exp_r)
    got = sorted(round(v, 2) for v in r_vals.values() if v is not None)
    ok_r = len(got) == len(exp_vals) and all(
        abs(a - b) <= ctx.args.tol_r for a, b in zip(got, exp_vals)
    )
    exp_n = ctx.expected.get("fig3_n_subjects")
    n_latest = max(data["latest"]["n_subjects_in_results"].values())
    ok_n = exp_n is None or n_latest == exp_n
    data["n_subjects_latest"] = n_latest
    return result("PASS" if ok_r and ok_n else "FAIL",
                  f"{n_latest} subjects in latest results (Q8: caption 32 / text 24); r sorted {got} vs {exp_vals}",
                  f"|r - r_paper| <= {ctx.args.tol_r}; figure diff vs thesis render: compare manually", **data)  # fmt: skip


def check_v9(ctx: Ctx) -> dict:
    import pandas as pd

    from mrextrap.data.splits import diagnosis_label
    from mrextrap.figures.fig5_beta import age_bin_table, beta_l1_norms, beta_pca

    d = ctx.betas_train()
    emb, plotted = beta_pca(d["betas"])
    df = pd.read_csv(ctx.need("splits", "train.csv"))
    dx_by = {sid: diagnosis_label(g["DX_filled"]) for sid, g in df.groupby("subject_id")}
    dx = [dx_by.get(s[0], "NA") for s in d["subject_ids"]]
    norms = beta_l1_norms(d["betas"])
    ages = np.array([float(a[0].mean()) for a in d["ages_abs"]])  # NOTE(paper) Q15: mean age
    log_norm = np.log10(norms)
    box = {}
    for k in sorted(set(dx)):
        v = log_norm[[x == k for x in dx]]
        box[k] = {
            "n": len(v),
            **{q: float(np.percentile(v, p)) for q, p in (("q1", 25), ("median", 50), ("q3", 75))},
        }
    mean_t, sem_t = age_bin_table(ages, norms, dx)
    return result("INFO", "Fig. 5 statistics from train latents (compare with the rendered figure)", "visual/numeric check",
                  pca_std=[float(x) for x in emb.std(axis=0)], n_plotted_a=len(plotted),
                  boxplot_log10_l1=box, age_bins_mean=json.loads(mean_t.round(4).to_json(orient="split")),
                  age_bins_sem=json.loads(sem_t.round(4).to_json(orient="split")))  # fmt: skip


def check_v12(ctx: Ctx) -> dict:
    from mrextrap.progression import compute_normalization

    norm = compute_normalization(ctx.betas_train())
    vals = {k: v.flatten().tolist() for k, v in norm.items()}
    data = {"computed": vals}
    exp = ctx.expected.get("normalization")
    status, msg = "INFO", "printed for comparison with the original training logs"
    if exp:
        bad = [
            k
            for k, v in exp.items()
            if k in vals and not np.allclose(vals[k], np.ravel(v), rtol=1e-4, atol=1e-6)
        ]
        status, msg = (
            ("PASS" if not bad else "FAIL"),
            f"compared with --expected; mismatching keys: {bad}",
        )
    p = ctx.path("priors", "normalization.pt")
    if p.is_file():
        ref = torch.load(p, map_location="cpu")
        eq = {
            k: bool(torch.allclose(norm[k].double(), torch.as_tensor(ref[k]).double(), rtol=1e-6))
            for k in norm
        }
        data["matches_priors_normalization_pt"] = eq
        if not all(eq.values()):
            status, msg = "FAIL", msg + "; differs from priors/normalization.pt"
    return result(status, msg, "rtol 1e-4 vs logs, 1e-6 vs normalization.pt", **data)


def check_s1(ctx: Ctx) -> dict:
    """Rebuild the splits with ``mrextrap.data.splits`` and compare subject sets (counts only)."""
    import pandas as pd

    from mrextrap.data.splits import build_split_csvs, read_subject_list

    a = ctx.args
    if not (a.adnimerge_csv and a.image_index_csv):
        raise ValueError("--adnimerge-csv and --image-index-csv are required")
    if a.legacy_splits_dir:
        legacy_dir, pat = Path(a.legacy_splits_dir), "{}_df_nona.csv"
    elif a.legacy_root:
        legacy_dir, pat = Path(a.legacy_root) / "data", "{}_df_nona.csv"
    else:
        legacy_dir, pat = ctx.path("splits"), "{}.csv"
    if not (legacy_dir / pat.format("train")).is_file() and (legacy_dir / "train.csv").is_file():
        pat = "{}.csv"
    test = read_subject_list(a.test_subjects_file) if a.test_subjects_file else None
    val = read_subject_list(a.val_subjects_file) if a.val_subjects_file else None
    with tempfile.TemporaryDirectory() as tmp:
        paths = build_split_csvs(a.adnimerge_csv, a.image_index_csv, test, val, tmp, a.split_seed)
        new = {s: pd.read_csv(p) for s, p in paths.items()}
    old = {s: pd.read_csv(legacy_dir / pat.format(s)) for s in ("train", "val", "test")}
    sets_new = {s: set(df["subject_id"]) for s, df in new.items()}
    sets_old = {s: set(df["subject_id"]) for s, df in old.items()}
    per_split, all_equal = {}, True
    for s in ("train", "val", "test"):
        only_old, only_new = sets_old[s] - sets_new[s], sets_new[s] - sets_old[s]
        all_equal &= not only_old and not only_new
        per_split[s] = {"legacy_subjects": len(sets_old[s]), "rebuilt_subjects": len(sets_new[s]),
                        "legacy_scans": len(old[s]), "rebuilt_scans": len(new[s]),
                        "legacy_only": len(only_old), "rebuilt_only": len(only_new)}  # fmt: skip
    moved = {f"legacy_{a_}->rebuilt_{b_}": len(sets_old[a_] & sets_new[b_])
             for a_ in sets_old for b_ in sets_new}  # fmt: skip
    overlap = {
        f"{a_}&{b_}": len(sets_new[a_] & sets_new[b_])
        for a_, b_ in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    return result("PASS" if all_equal else "FAIL", "subject sets " + ("identical" if all_equal else "differ (counts in data)"),
                  "identical subject sets per split", per_split=per_split, confusion=moved, rebuilt_overlap=overlap,
                  explicit_lists_used=bool(test or val), seed=a.split_seed)  # fmt: skip


def check_q2a(ctx: Ctx) -> dict:
    from mrextrap.progression import compute_obs_var

    d = ctx.betas_train()
    orig = compute_obs_var(d["betas"], d["zs"], d["ages_abs"])
    intended = intended_obs_var(d["betas"], d["zs"], d["ages_abs"])
    rel = {
        "relative_difference_of_means": float((intended.mean() - orig.mean()) / orig.mean()),
        "mean_ratio_intended_over_original": float(intended.mean() / orig.mean()),
        "frobenius_relative_difference": float((intended - orig).norm() / orig.norm()),
        "median_elementwise_ratio": float((intended / orig).median()),
    }
    data = {"original": summarize_var(orig), "intended": summarize_var(intended), "relative": rel}
    p = ctx.path("priors", "global_params.pt")
    if p.is_file():
        data["original_vs_legacy_global_params"] = tensor_diff_summary(
            orig, torch.load(p, map_location="cpu")["global_obs_var"]
        )
    ctx.cache["q2_obs_var"] = {"original": orig, "intended": intended}
    return result("INFO", f"obs_var mean original {orig.mean():.4g} vs intended {intended.mean():.4g} "
                  f"(relative difference {rel['relative_difference_of_means']:+.2%}); exploratory, release default unchanged",
                  "", **data)  # fmt: skip


# --------------------------------------------------------------------------------------------
# GPU checks
# --------------------------------------------------------------------------------------------
def _decode_check(vae_a, vae_b, zs, device) -> float:
    with torch.no_grad():
        return max(
            float((vae_a.decode(z.to(device)).cpu() - vae_b.decode(z.to(device)).cpu()).abs().max())
            for z in zs
        )


def check_v0(ctx: Ctx) -> dict:
    from convert_checkpoints import (
        AE_CANDIDATES,
        DIFFUSION_CANDIDATES,
        UNET_CANDIDATES,
        find_weights,
    )

    from mrextrap.io import load_state_dict_any

    a = ctx.args
    if not a.converted_dir:
        raise ValueError("--converted-dir (output of convert_checkpoints.py) is required")
    conv = Path(a.converted_dir)
    pairs = [
        ("autoencoder", a.ae_legacy_dir, AE_CANDIDATES, conv / "autoencoder" / "model.safetensors"),
        ("unet_prior", a.unet_legacy_dir, UNET_CANDIDATES, conv / "unet_prior" / "checkpoints" / f"epoch_{a.prior_epoch}" / "model.safetensors"),
        ("diffusion_prior", a.diffusion_legacy_dir, DIFFUSION_CANDIDATES, conv / "diffusion_prior" / "checkpoints" / f"epoch_{a.prior_epoch}" / "ema_denoiser.safetensors"),
    ]  # fmt: skip
    data, ok = {}, True
    for name, legacy_dir, cands, converted in pairs:
        if not legacy_dir:
            continue
        old, new = (
            load_state_dict_any(find_weights(legacy_dir, cands)),
            load_state_dict_any(converted),
        )
        same = set(old) == set(new) and all(torch.equal(old[k], new[k]) for k in old if k in new)
        data[name] = {"identical_state_dict": same, "n_tensors": len(old)}
        ok &= same
    if a.ae_legacy_dir and a.ae_config:
        from omegaconf import OmegaConf

        from mrextrap.training.autoencoder import build_vae

        cfg = OmegaConf.load(a.ae_config)
        legacy = build_vae(cfg.model)
        legacy.load_state_dict(load_state_dict_any(find_weights(a.ae_legacy_dir, AE_CANDIDATES)))
        legacy = legacy.to(ctx.device).eval()
        zs = [z[:1] for z in ctx.latents("test")["zs"][: a.n_encode]]
        diff = _decode_check(ctx.vae(), legacy, zs, ctx.device)
        data["decoder_max_abs_diff_state_dict_path"] = diff
        ok &= diff == 0.0
        try:  # legacy loading path: accelerate load_state into a prepared model (as in the eval scripts)
            from accelerate import Accelerator

            acc = Accelerator(mixed_precision="bf16", cpu=(ctx.device == "cpu"))
            model = acc.prepare(build_vae(cfg.model))
            acc.load_state(str(a.ae_legacy_dir))
            diff_acc = _decode_check(ctx.vae(), acc.unwrap_model(model).eval(), zs, ctx.device)
            data["decoder_max_abs_diff_accelerate_path"] = diff_acc
            ok &= diff_acc == 0.0
        except Exception as e:  # noqa: BLE001
            data["accelerate_path_error"] = scrub(e)
    return result(
        "PASS" if ok else "FAIL",
        "converted vs legacy weights and decoder outputs",
        "bit-identical",
        **data,
    )


def check_v1(ctx: Ctx) -> dict:
    from mrextrap.config import load_config
    from mrextrap.data.datasets import get_longitudinal_dataloaders
    from mrextrap.encoding import encode_split

    a = ctx.args
    if not a.config:
        raise ValueError("--config (AE config with dataset settings) is required")
    cfg = load_config(a.config, [f"paths.run_dir={ctx.run_dir}"])
    cfg.training.batch_size = 1
    test_loader = get_longitudinal_dataloaders(cfg, num_workers=0)[2]

    class Take:  # first n multi-scan subjects of the test loader
        def __init__(self):
            self.items = []
            for batch in test_loader:
                if batch["vols"].shape[1] > 1:
                    self.items.append(batch)
                if len(self.items) == a.n_encode:
                    break

        def __iter__(self):
            return iter(self.items)

        def __len__(self):
            return len(self.items)

    new = encode_split(ctx.vae(), Take(), ctx.device, "test")
    cached = ctx.latents("test")
    pos = {s[0]: i for i, s in enumerate(cached["subject_ids"])}
    diffs = [
        float((z.cpu() - cached["zs"][pos[sid[0]]]).abs().max())
        for z, sid in zip(new["zs"], new["subject_ids"])
    ]
    ok = bool(diffs) and max(diffs) < a.tol_encode
    return result("PASS" if ok else "FAIL", f"re-encoded {len(diffs)} test subjects", f"max|dz| < {a.tol_encode}",
                  max_abs_dz_per_subject=diffs)  # fmt: skip


def check_v3(ctx: Ctx) -> dict:
    from mrextrap.progression import fit_betas_per_reference

    d = ctx.betas_train()
    idx = [i for i, z in enumerate(d["zs"]) if z.shape[0] >= 2][: ctx.args.n_refit]
    torch.manual_seed(42)  # NOTE(paper) Q10: seeded once, RNG stream consumed in subject order
    stats = []
    for i in idx:
        b = fit_betas_per_reference(
            d["zs"][i], d["ages"][i].permute(1, 0), use_l1=True, device=ctx.device
        )
        ref = d["betas"][i]
        stats.append({"rel_l2_error": float((b - ref).norm() / ref.norm()),
                      "max_abs_diff": float((b - ref).abs().max()),
                      "correlation": float(np.corrcoef(b.flatten().numpy(), ref.flatten().numpy())[0, 1])})  # fmt: skip
    med = float(np.median([s["rel_l2_error"] for s in stats]))
    return result("PASS" if med <= ctx.args.tol_beta_rel else "FAIL", f"median relative L2 error {med:.4f} over {len(stats)} subjects",
                  f"median rel L2 <= {ctx.args.tol_beta_rel} (Adam fit, nondeterministic)", per_subject=stats)  # fmt: skip


def check_v11(ctx: Ctx) -> dict:
    import nibabel as nib

    from mrextrap.segmentation.volumes import read_vols_csv

    real = ctx.real_dir()
    sids = sorted(p.name for p in real.iterdir() if p.is_dir())[: ctx.args.n_encode]
    worst, n_rows = [], 0
    with tempfile.TemporaryDirectory() as tmp:
        for k, sid in enumerate(sids):
            files = sorted((real / sid).glob("tmp*_final.nii.gz"))
            arrays = [nib.load(str(f)).get_fdata() for f in files]
            csv_new = ctx.segmenter().segment(arrays, Path(tmp) / f"s{k}")
            new, old = read_vols_csv(csv_new), read_vols_csv(real / sid / "vols.csv")
            n = min(len(new), len(old))
            n_rows += n
            cols = [c for c in new.columns if c in old.columns]
            worst.append(
                float(
                    np.abs(
                        new[cols].iloc[:n].astype(float).values
                        - old[cols].iloc[:n].astype(float).values
                    ).max()
                )
            )
    ok = bool(worst) and max(worst) <= ctx.args.tol_vols
    return result("PASS" if ok else "FAIL", f"{len(worst)} subjects, {n_rows} scans segmented", f"max|dvol| <= {ctx.args.tol_vols}",
                  max_abs_vol_diff_per_subject=worst)  # fmt: skip


def _load_ref_results(ctx: Ctx, method: str) -> dict:
    from mrextrap.evaluation.stats import load_results

    return load_results(ctx.need("eval", "single_volume", method, "results.json"))


def _compare_single(new: dict, ref: dict, n: int | None) -> dict:
    from mrextrap.evaluation.stats import align_pair, maes

    per_region = {}
    for r in REGIONS:
        rows_ref = ref[r][:n] if n else ref[r]
        a, b = align_pair(new[r], rows_ref)
        per_region[r] = {"n": len(a), "max_abs_diff": float(np.abs(a - b).max()), "mean_abs_diff": float(np.abs(a - b).mean()),
                         "mean_new": float(a.mean()), "mean_ref": float(maes(rows_ref).mean())}  # fmt: skip
    return per_region


def _single_volume(ctx: Ctx, method: str) -> dict:
    from mrextrap.evaluation.single_volume import evaluate_single_volume
    from mrextrap.prediction import DiffusionPrior, GlobalPrior, UNetPrior

    a = ctx.args
    subjects = ctx.daninet_subjects()[: a.n_eval]
    if method == "global":
        predictor = GlobalPrior.from_file(ctx.need("priors", "global_params.pt"))
    else:
        from omegaconf import OmegaConf

        norm = torch.load(ctx.need("priors", "normalization.pt"), map_location="cpu")
        if method == "unet":
            from mrextrap.training.unet_prior import load_unet_prior

            cfg = OmegaConf.load(a.unet_config or ctx.need("unet_prior", "config.yaml"))
            ckpt = (
                Path(a.unet_checkpoint)
                if a.unet_checkpoint
                else ctx.need("unet_prior", "checkpoints", f"epoch_{a.prior_epoch}")
            )
            predictor = UNetPrior(load_unet_prior(cfg, ckpt).to(ctx.device), norm)
        else:
            from mrextrap.training.diffusion_prior import load_diffusion_prior

            cfg = OmegaConf.load(a.diffusion_config or ctx.need("diffusion_prior", "config.yaml"))
            ckpt = (
                Path(a.diffusion_checkpoint)
                if a.diffusion_checkpoint
                else ctx.need("diffusion_prior", "checkpoints", f"epoch_{a.prior_epoch}")
            )
            predictor = DiffusionPrior(
                load_diffusion_prior(cfg, ckpt).to(ctx.device), norm, NT=int(cfg.training.NT)
            )
    return evaluate_single_volume(method, subjects, predictor, ctx.vae(), ctx.segmenter(), ctx.real_dir(),
                                  ctx.out_dir / "rerun" / f"single_{method}", device=ctx.device)  # fmt: skip


def check_v5(ctx: Ctx) -> dict:
    new = _single_volume(ctx, "global")
    cmp = _compare_single(new, _load_ref_results(ctx, "global"), ctx.args.n_eval)
    worst = max(v["max_abs_diff"] for v in cmp.values())
    return result("PASS" if worst <= ctx.args.tol_mae else "FAIL", f"Global single-volume, worst |dMAE| {worst:.4g} % TBV",
                  f"max|dMAE| <= {ctx.args.tol_mae}", per_region=cmp)  # fmt: skip


def check_v6(ctx: Ctx) -> dict:
    data, worst = {}, 0.0
    new = _single_volume(ctx, "unet")
    data["unet"] = _compare_single(new, _load_ref_results(ctx, "unet"), ctx.args.n_eval)
    worst = max(v["max_abs_diff"] for v in data["unet"].values())
    ok = worst <= ctx.args.tol_mae
    if ctx.args.include_diffusion:
        new_d = _single_volume(ctx, "diffusion")
        ref_d = {r: [(m,) for m in v] for r, v in diffusion_maes(ctx).items()}
        data["diffusion"] = _compare_single(new_d, ref_d, ctx.args.n_eval)
        worst_d = max(v["max_abs_diff"] for v in data["diffusion"].values())
        ok &= worst_d <= ctx.args.tol_mae
        worst = max(worst, worst_d)
    return result(
        "PASS" if ok else "FAIL",
        f"worst |dMAE| {worst:.4g} % TBV",
        f"max|dMAE| <= {ctx.args.tol_mae}",
        **data,
    )


def _multivolume(ctx: Ctx, tag: str, global_params) -> dict:
    """Run earliest+latest multi-volume evaluation with the given global params; cached per tag."""
    key = f"mv_{tag}"
    if key in ctx.cache:
        return ctx.cache[key]
    from mrextrap.evaluation import multivolume as mv

    earliest, latest = mv.prepare_data_unified(
        ctx.latents("test"), ctx.dx_lookup(), 2, (1, 2, 3), 3
    )
    n = ctx.args.n_eval
    out = {}
    for fmt, filtered in (("earliest", earliest), ("latest", latest)):
        out[fmt] = mv.evaluate_conditioning(filtered[:n], global_params, ctx.vae(), ctx.segmenter(), ctx.real_dir(),
                                            ctx.out_dir / "rerun" / f"multivol_{tag}" / fmt, device=ctx.device)  # fmt: skip
    ctx.cache[key] = out
    return out


def check_v8(ctx: Ctx) -> dict:
    params = ctx.need("priors", "global_params.pt")
    new = _multivolume(ctx, "original", params)
    data = {}
    for fmt in ("earliest", "latest"):
        ref = json.loads(ctx.need("eval", "multivolume", fmt, "results_var1.00.json").read_text())
        data[fmt] = compare_multivolume(new[fmt], ref)
    worst = max(v["max_abs_diff"] for v in data.values())
    matched = all(v["n_matched"] > 0 for v in data.values())
    return result("PASS" if matched and worst <= ctx.args.tol_mae else "FAIL", f"worst |dMAE| {worst:.4g} % TBV",
                  f"max|dMAE| <= {ctx.args.tol_mae}", **data)  # fmt: skip


def check_q2b(ctx: Ctx) -> dict:
    from mrextrap.prediction import GlobalPrior

    base = GlobalPrior.from_file(ctx.need("priors", "global_params.pt"))
    if "q2_obs_var" not in ctx.cache:
        check_q2a(ctx)
    ov = ctx.cache["q2_obs_var"]
    orig = _multivolume(ctx, "original", GlobalPrior(base.beta_mean, base.beta_std, ov["original"]))
    intended = _multivolume(
        ctx, "intended", GlobalPrior(base.beta_mean, base.beta_std, ov["intended"])
    )
    data = {}
    for fmt in ("earliest", "latest"):
        t_o, t_i = mae_table(orig[fmt]), mae_table(intended[fmt])
        data[fmt] = {r: {str(n): {"original": t_o[r][n], "intended": t_i[r][n], "diff_intended_minus_original": t_i[r][n] - t_o[r][n]}
                         for n in t_o[r] if n in t_i[r]} for r in t_o}  # fmt: skip
    return result(
        "INFO",
        "multi-volume MAE with original vs intended obs_var (exploratory)",
        "",
        mae_by_region_ncond=data,
    )


def check_v10(ctx: Ctx) -> dict:
    from mrextrap.config import load_config
    from mrextrap.data.datasets import get_longitudinal_dataloaders
    from mrextrap.evaluation.interpolation import interpolation_volumes

    if not ctx.args.config:
        raise ValueError("--config is required")
    cfg = load_config(ctx.args.config, [f"paths.run_dir={ctx.run_dir}"])
    cfg.training.batch_size = 1
    loader = get_longitudinal_dataloaders(cfg, num_workers=0)[2]
    outs = {}
    for mode in ("voxel", "latent"):
        out = ctx.out_dir / "rerun" / f"interp_{mode}"
        interpolation_volumes(ctx.vae(), loader, ctx.segmenter(), out, mode=mode, n_alpha=10)
        outs[mode] = str(out)
    return result(
        "INFO",
        "interpolation volumes written for both modes (Q6); compare linearity manually",
        "",
        output_dirs=outs,
    )


CHECKS = {
    "V0": ("converted checkpoints identical", check_v0),
    "V1": ("re-encode vs cached latents", check_v1),
    "V2": ("global params from train betas", check_v2),
    "V3": ("beta re-fit (5 train subjects)", check_v3),
    "V4": ("Table 1 from cached results", check_v4),
    "V5": ("Global single-volume end-to-end", check_v5),
    "V5a": ("DANINet baseline index > 0 (Q1)", check_v5a),
    "V6": ("UNet/diffusion single-volume", check_v6),
    "V7": ("Fig. 3 data", check_v7),
    "V8": ("Global multi-volume vs cached", check_v8),
    "V9": ("Fig. 5 statistics", check_v9),
    "V10": ("Fig. 2 interpolation, both modes", check_v10),
    "V11": ("SynthSeg vs cached vols.csv", check_v11),
    "V12": ("normalisation statistics", check_v12),
    "S1": ("split subject sets vs legacy", check_s1),
    "Q2a": ("obs_var original vs intended", check_q2a),
    "Q2b": ("multi-volume with both obs_var", check_q2b),
}


def select_checks(spec: str) -> list[str]:
    groups = {"cpu": CPU_CHECKS, "gpu": GPU_CHECKS, "gpu-long": GPU_LONG_CHECKS,
              "all": CPU_CHECKS + GPU_CHECKS + GPU_LONG_CHECKS}  # fmt: skip
    out: list[str] = []
    for token in (t.strip() for t in spec.split(",") if t.strip()):
        for c in groups.get(token, (token,)):
            if c not in CHECKS:
                raise SystemExit(
                    f"unknown check {c!r}; known: {sorted(CHECKS)} and groups {sorted(groups)}"
                )
            if c not in out:
                out.append(c)
    return out


# --------------------------------------------------------------------------------------------
# Runner and reports
# --------------------------------------------------------------------------------------------
def run_checks(ctx: Ctx, ids: list[str]) -> dict:
    results = {}
    for cid in ids:
        name, fn = CHECKS[cid]
        try:
            r = fn(ctx)
        except (ImportError, FileNotFoundError, ValueError) as e:
            r = result("SKIP", f"{type(e).__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            r = result(
                "ERROR", f"{type(e).__name__}: {e}", trace=scrub(traceback.format_exc(limit=3))
            )
        r["name"] = name
        results[cid] = r
        print(f"[{r['status']:5s}] {cid:4s} {name}: {r['message']}", flush=True)
    return results


def write_reports(ctx: Ctx, results: dict) -> None:
    counts = {
        s: sum(1 for r in results.values() if r["status"] == s)
        for s in ("PASS", "NEAR", "INFO", "SKIP", "FAIL", "ERROR")
    }
    report = {
        "run_dir": str(ctx.run_dir),
        "device": ctx.device,
        "summary": counts,
        "checks": results,
    }
    rows = [
        (cid, r["name"], r["status"], r["tolerance"], r["message"]) for cid, r in results.items()
    ]
    md = "# Validation against legacy artefacts\n\n" + markdown_table(
        ["check", "what", "status", "tolerance", "message"], rows
    )
    md += "\n\n" + ", ".join(f"{k}: {v}" for k, v in counts.items()) + "\n"
    write_report(ctx.out_dir, "validation_report", report, md)
    if "Q2a" in results or "Q2b" in results:
        q = {
            "label": "EXPLORATORY: original vs intended observation-noise variance. NOT the release default "
            "(the release keeps the original N x N broadcast behaviour, Q2).",
            "Q2a_obs_var": results.get("Q2a"),
            "Q2b_multivolume_mae": results.get("Q2b"),
        }
        md_q = (
            "# Q2 obs_var: original vs intended formula (exploratory)\n\n"
            "**Not the release default.** The release keeps the original behaviour (unintended N x N broadcast).\n\n"
        )
        a = (results.get("Q2a") or {}).get("data", {})
        if a:
            md_q += markdown_table(
                ["variant", "mean", "median", "min", "max"],
                [
                    (k, *(f"{a[k][m]:.4g}" for m in ("mean", "median", "min", "max")))
                    for k in ("original", "intended")
                ],
            )
            md_q += (
                "\n\nRelative difference: "
                + ", ".join(f"{k} = {v:.4g}" for k, v in a["relative"].items())
                + "\n"
            )
        b = (results.get("Q2b") or {}).get("data", {}).get("mae_by_region_ncond", {})
        for fmt, regs in b.items():
            md_q += f"\n## Multi-volume MAE (% TBV), {fmt} baseline\n\n"
            md_q += (
                markdown_table(
                    ["region", "n_cond", "original", "intended", "intended - original"],
                    [
                        (
                            r,
                            n,
                            f"{v['original']:.4f}",
                            f"{v['intended']:.4f}",
                            f"{v['diff_intended_minus_original']:+.4f}",
                        )
                        for r, ns in regs.items()
                        for n, v in ns.items()
                    ],
                )
                + "\n"
            )
        write_report(ctx.out_dir, "report_q2_obs_var", q, md_q)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run-dir", default=".", help="run directory (release layout)")
    p.add_argument("--out-dir", help="report directory (default run_dir/validation)")
    p.add_argument(
        "--checks", default="cpu", help="comma list of ids or groups cpu|gpu|gpu-long|all"
    )
    p.add_argument("--list", action="store_true", help="list checks and exit")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--expected", help="JSON with reference numbers (see module docstring)")
    p.add_argument("--daninet-pairs", help="csv: sid, baseline_age, target_age (user supplied)")
    p.add_argument("--legacy-root", help="legacy repo root (for data/{split}_df_nona.csv)")
    p.add_argument("--legacy-splits-dir")
    p.add_argument("--adnimerge-csv")
    p.add_argument("--image-index-csv")
    p.add_argument("--test-subjects-file")
    p.add_argument("--val-subjects-file")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--config", help="AE config yaml with dataset settings (V1, V10)")
    p.add_argument("--ae-config", help="AE config yaml (model block)")
    p.add_argument("--ae-path", help="AE weights dir/file (default run_dir/autoencoder)")
    p.add_argument("--unet-config")
    p.add_argument("--unet-checkpoint")
    p.add_argument("--diffusion-config")
    p.add_argument("--diffusion-checkpoint")
    p.add_argument("--include-diffusion", action="store_true")
    p.add_argument("--prior-epoch", type=int, default=2000)
    p.add_argument("--converted-dir", help="output of convert_checkpoints.py")
    p.add_argument("--ae-legacy-dir")
    p.add_argument("--unet-legacy-dir")
    p.add_argument("--diffusion-legacy-dir")
    p.add_argument("--synthseg-weights")
    p.add_argument("--n-encode", type=int, default=3, help="subjects for V0/V1/V11")
    p.add_argument("--n-refit", type=int, default=5, help="subjects for V3")
    p.add_argument("--n-eval", type=int, default=None, help="limit subjects in V5/V6/V8/Q2b")
    p.add_argument("--tol-encode", type=float, default=1e-4)
    p.add_argument("--tol-vols", type=float, default=1e-3)
    p.add_argument("--tol-mae", type=float, default=1e-2)
    p.add_argument("--tol-beta-rel", type=float, default=0.1)
    p.add_argument("--tol-table", type=float, default=5e-4)
    p.add_argument("--tol-p", type=float, default=5e-4)
    p.add_argument("--tol-r", type=float, default=5e-3)
    p.add_argument("--rtol-params", type=float, default=1e-6)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        for cid, (name, _) in CHECKS.items():
            print(f"{cid:4s} {name}")
        return 0
    ctx = Ctx(args)
    results = run_checks(ctx, select_checks(args.checks))
    write_reports(ctx, results)
    return 1 if any(r["status"] in STATUS_FAILING for r in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
