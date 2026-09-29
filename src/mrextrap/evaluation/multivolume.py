"""Multi-volume evaluation with the Global prior (paper Fig. 3, qualitative Fig. 4 data).

The Global prior ``beta ~ N(mean, std^2)`` is updated in closed form with 0-3 additional scans
(``update_beta_posterior_closed_form``) and the resulting linear latent trajectory is decoded,
segmented and compared with the real regional volumes (% of the first real scan's TBV).

Ported from ``eval_global_prior_varcond.py`` (``prepare_data_for_evaluation_unified``,
``evaluate_predictions_per_future_timepoint``,
``main_evaluation_conditioning_on_volumes_global_prior``, ``process_results_unified``,
``compute_and_plot_progression_rates``, ``analyze_mae_vs_progression_shift_all_regions``) and from
the thesis re-render ``prep_data.py`` (de-plotted panel data). Numerics, orderings and the results
format are kept; the only change to the legacy evaluation loop is that the latent dictionary is
passed as an argument instead of being read from a module-level global.

Layout under ``out_dir`` (one directory per baseline format, e.g. ``.../multivolume/latest``)::

    results_var1.00.json                 {region: [[n_cond, t_idx, mae, age, dx, sid], ...]}
    temp_seg_<sid>/ncond<n>/             decoded predictions + segmentations + vols.csv

Real-scan volumes are read from ``real_vols_dir/<sid>/vols.csv``.
"""

from __future__ import annotations

import json
import logging
import traceback
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from mrextrap.prediction import GlobalPrior, update_beta_posterior_closed_form
from mrextrap.segmentation.volumes import (
    REGIONS,
    predicted_region_volumes,
    real_region_volumes,
)

log = logging.getLogger(__name__)

BASELINE_FORMATS = ("earliest", "latest")
DEFAULT_CONDITIONING_YEARS = (1, 2, 3)


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------
def _diagnosis(dx_lookup: Mapping[str, str] | Callable[[str], str], sid: str) -> str:
    if callable(dx_lookup):
        return dx_lookup(sid)
    if sid not in dx_lookup:
        raise ValueError(f"Subject {sid} not found in the diagnosis lookup")
    return dx_lookup[sid]


def _floats(x) -> np.ndarray:
    return np.asarray(torch.as_tensor(x).detach().cpu().numpy(), dtype=float).reshape(-1)


def _slope(x, y) -> float:
    return float(np.polyfit(x, y, 1)[0])


def _real_volumes(real_vols_dir: str | Path, sid: str):
    return real_region_volumes(Path(real_vols_dir) / sid / "vols.csv")


def load_global_params(global_params) -> GlobalPrior:
    """Accept a ``GlobalPrior``, a ``global_params.pt`` path or the legacy dict."""
    if isinstance(global_params, GlobalPrior):
        return global_params
    if isinstance(global_params, str | Path):
        return GlobalPrior.from_file(global_params)
    return GlobalPrior(
        global_params["global_beta_mean"],
        global_params["global_beta_std"],
        global_params.get("global_obs_var"),
    )


# --------------------------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------------------------
def prepare_data_unified(
    latent_dict: dict,
    dx_lookup: Mapping[str, str] | Callable[[str], str],
    min_future_years: float = 2,
    conditioning_years: Sequence[float] = DEFAULT_CONDITIONING_YEARS,
    min_baseline_years: float | None = None,
) -> tuple[list[dict], list[dict]]:
    """Subjects with all conditioning scans and >= 1 later scan, in earliest and latest format.

    ``dx_lookup`` maps subject id -> "CN"/"MCI"/"Dementia" (mapping or callable).

    Earliest format: baseline = first scan; conditioning = the scans closest (within 0.5 y) to
    ``conditioning_years`` after it; predictions = all scans after the last conditioning scan.
    Latest format (same subjects, same predictions): baseline = the last conditioning scan, the
    first scan plus the earlier conditioning scans become the conditioning set (negative time
    offsets to the baseline). Both lists are sorted by subject id.

    NOTE(paper): ``min_baseline_years`` is accepted for compatibility but has no effect (the
    legacy function only computes its default and never uses it).
    """
    if min_baseline_years is None:
        min_baseline_years = max(conditioning_years)
    earliest_out: list[dict] = []
    latest_out: list[dict] = []
    stats = {
        "total_subjects": len(latent_dict["subject_ids"]),
        "too_few_timepoints": 0,
        "insufficient_span": 0,
        "missing_conditioning": 0,
        "no_prediction_points": 0,
    }

    for sub_idx in range(stats["total_subjects"]):
        ages = latent_dict["ages"][sub_idx][0]
        zs = latent_dict["zs"][sub_idx]
        ages_abs = latent_dict["ages_abs"][sub_idx][0]
        subject_id = latent_dict["subject_ids"][sub_idx][0]
        betas = latent_dict.get("betas")
        true_betas = betas[sub_idx] if betas is not None else None
        disease_status = _diagnosis(dx_lookup, subject_id)

        if len(ages) < 2:
            stats["too_few_timepoints"] += 1
            continue

        baseline_idx = 0
        baseline_age = ages[baseline_idx]
        if (ages[-1] - baseline_age) < min_future_years:
            stats["insufficient_span"] += 1
            continue

        baseline_data = {
            "z0": zs[baseline_idx : baseline_idx + 1],
            "age0": ages[baseline_idx : baseline_idx + 1],
            "age_abs0": ages_abs[baseline_idx : baseline_idx + 1],
            "index": baseline_idx,
        }

        cond = {"zs": [], "ages": [], "ages_abs": [], "indices": []}
        last_conditioning_idx = baseline_idx
        for year in conditioning_years:
            target_age = baseline_age + year
            if baseline_idx + 1 >= len(ages):
                break
            subset = ages[baseline_idx + 1 :]
            idx = baseline_idx + 1 + int(np.argmin(np.abs(subset - target_age)))
            if np.abs(ages[idx] - target_age) <= 0.5:
                cond["zs"].append(zs[idx : idx + 1])
                cond["ages"].append(ages[idx : idx + 1])
                cond["ages_abs"].append(ages_abs[idx : idx + 1])
                cond["indices"].append(idx)
                last_conditioning_idx = max(last_conditioning_idx, idx)

        if len(cond["zs"]) != len(conditioning_years):
            stats["missing_conditioning"] += 1
            continue

        pred_start = last_conditioning_idx + 1
        if pred_start >= len(ages):
            stats["no_prediction_points"] += 1
            continue

        prediction = {
            "zs": zs[pred_start:],
            "ages": ages[pred_start:],
            "ages_abs": ages_abs[pred_start:],
            "indices": list(range(pred_start, len(ages))),
        }
        earliest_out.append(
            {
                "subject_id": subject_id,
                "baseline": baseline_data,
                "prediction": prediction,
                "conditioning": {
                    "zs": torch.cat(cond["zs"], dim=0),
                    "ages": torch.cat(cond["ages"], dim=0),
                    "ages_abs": torch.cat(cond["ages_abs"], dim=0),
                    "indices": cond["indices"],
                },
                "true_betas": true_betas,
                "disease_status": disease_status,
            }
        )
        latest_out.append(
            {
                "subject_id": subject_id,
                "baseline": {
                    "z0": cond["zs"][-1],
                    "age0": cond["ages"][-1],
                    "age_abs0": cond["ages_abs"][-1],
                    "index": cond["indices"][-1],
                },
                "prediction": prediction,
                "conditioning": {
                    "zs": torch.cat([baseline_data["z0"]] + cond["zs"][:-1], dim=0),
                    "ages": torch.cat([baseline_data["age0"]] + cond["ages"][:-1], dim=0),
                    "ages_abs": torch.cat(
                        [baseline_data["age_abs0"]] + cond["ages_abs"][:-1], dim=0
                    ),
                    "indices": [baseline_data["index"]] + cond["indices"][:-1],
                },
                "true_betas": true_betas,
                "disease_status": disease_status,
            }
        )

    earliest_out.sort(key=lambda x: x["subject_id"])
    latest_out.sort(key=lambda x: x["subject_id"])
    log.info("Multi-volume filtering: %s -> %d subjects", stats, len(earliest_out))
    return earliest_out, latest_out


# --------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------
def evaluate_predictions_per_future_timepoint(
    z_pred: torch.Tensor,
    subject_data: dict,
    vae,
    segmenter,
    real_vols_dir: str | Path,
    seg_root: str | Path,
    n_conditioning: int = 0,
    device: str = "cpu",
    regions: Sequence[str] = REGIONS,
) -> dict[str, list[tuple[int, float, float]]]:
    """Decode ``z_pred`` [T,C,D,H,W], segment, and return ``{region: [(t, mae, age), ...]}``.

    ``mae`` is the absolute error of the region volume (% of the first real scan's TBV) at each
    future timepoint ``t``; ``age`` is the target age in years since the subject's first scan.
    Segmentations are stored in ``seg_root/temp_seg_<sid>/ncond<n>``.
    """
    prediction = subject_data["prediction"]
    subject_id = subject_data["subject_id"]

    real_pred_volumes = _real_volumes(real_vols_dir, subject_id)
    tbv = real_pred_volumes["TBV"].iloc[0]

    # NOTE(paper) (Q11): decoding runs in fp32 (no autocast around ``vae.decode``).
    x_preds = []
    with torch.no_grad():
        for z_pred_ in z_pred:
            x_pred = vae.decode(z_pred_.unsqueeze(0).float().to(device)).cpu()
            x_preds.append(x_pred)
    x_pred = torch.cat(x_preds).cpu().detach().numpy()

    seg_path = Path(seg_root) / f"temp_seg_{subject_id}" / f"ncond{n_conditioning}"
    vols_csv = segmenter.segment([x_[0] for x_ in x_pred], seg_path)
    pred_volumes = predicted_region_volumes(vols_csv, tbv)

    maes_per_timepoint: dict[str, list[tuple[int, float, float]]] = {}
    prediction_ages = prediction["ages"].numpy()
    for region in regions:
        real_vols = real_pred_volumes[f"{region}_pct_tbv"].iloc[prediction["indices"]].values
        pred_vols = pred_volumes[f"{region}_pct_tbv"].values
        maes_per_timepoint[region] = [
            (t, float(np.abs(real_vols[t] - pred_vols[t])), float(prediction_ages[t]))
            for t in range(len(prediction_ages))
        ]
    return maes_per_timepoint


def evaluate_conditioning(
    filtered: list[dict],
    global_params,
    vae,
    segmenter,
    real_vols_dir: str | Path,
    out_dir: str | Path,
    device: str = "cpu",
    relative_obs_var: float = 1.0,
    regions: Sequence[str] = REGIONS,
    raise_on_error: bool = False,
) -> dict[str, list[list]]:
    """Global prior + closed-form posterior with 0..n additional scans (``main_evaluation_...``).

    ``filtered`` is one list of ``prepare_data_unified``. For each subject the prior trajectory
    (n=0) and, for n = 1..max, the posterior given the **last n** conditioning scans are decoded
    and scored. Returns ``{region: [(n_cond, t_idx, mae, age, dx, sid), ...]}`` and writes it to
    ``out_dir/results_var{relative_obs_var:.2f}.json`` (tuples become lists in JSON).

    NOTE(paper): ``relative_obs_var`` only names the file; the observation variance is always the
    global ``global_obs_var`` (Q2). Subjects that raise are skipped and logged, as in the legacy loop
    (``raise_on_error=True`` re-raises).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prior = load_global_params(global_params)
    beta_mean, beta_var = prior.mean_var("cpu")
    obs_var = prior.obs_var

    results: dict[str, list] = {region: [] for region in regions}
    successful = failed = 0

    def store(n_cond: int, maes: dict, subject_data: dict) -> None:
        for region in regions:
            for timepoint_idx, mae, age in maes[region]:
                results[region].append(
                    [n_cond, timepoint_idx, mae, age, subject_data["disease_status"],
                     subject_data["subject_id"]]
                )  # fmt: skip

    for subject_data in filtered:
        try:
            max_conditioning = len(subject_data["conditioning"]["zs"])
            baseline = subject_data["baseline"]
            prediction_ages = subject_data["prediction"]["ages"].float()
            deltaages_pred = prediction_ages - baseline["age0"].float()

            def trajectory(beta, baseline=baseline, deltaages_pred=deltaages_pred):
                return beta * deltaages_pred.reshape(-1, 1, 1, 1, 1) + baseline["z0"]

            def evaluate(z_pred, n, subject_data=subject_data):
                return evaluate_predictions_per_future_timepoint(
                    z_pred, subject_data, vae, segmenter, real_vols_dir, out_dir,
                    n_conditioning=n, device=device, regions=regions,
                )  # fmt: skip

            store(0, evaluate(trajectory(beta_mean), 0), subject_data)

            for n_conditioning in range(1, max_conditioning + 1):
                cond = subject_data["conditioning"]
                # NOTE(paper): the *last* n conditioning scans are used.
                deltaages = cond["ages"][-n_conditioning:] - baseline["age0"]
                beta_post_mean, _ = update_beta_posterior_closed_form(
                    beta_mean, beta_var, baseline["z0"], deltaages,
                    cond["zs"][-n_conditioning:], obs_var=obs_var,
                )  # fmt: skip
                store(n_conditioning, evaluate(trajectory(beta_post_mean), n_conditioning),
                      subject_data)  # fmt: skip
            successful += 1
        except Exception as e:  # legacy behaviour: skip and report
            if raise_on_error:
                raise
            log.warning("Failed for subject %s: %s\n%s", subject_data["subject_id"], e,
                        traceback.format_exc())  # fmt: skip
            failed += 1

    log.info(
        "Multi-volume evaluation: %d/%d subjects ok, %d failed", successful, len(filtered), failed
    )
    with open(out_dir / f"results_var{relative_obs_var:.2f}.json", "w") as f:
        json.dump(results, f)
    return results


# --------------------------------------------------------------------------------------------
# Results processing
# --------------------------------------------------------------------------------------------
def process_results_unified(
    results: dict, baseline_format: str = "latest", earliest_min_age: float = 4
) -> dict:
    """Group results by region/subject/n_cond and split predictions into near and far term.

    Returns ``{region: {sid: {"baseline_age", "diagnosis", "by_conditioning":
    {n: {"near_term": [{"mae", "prediction_time"}], "far_term": [...]}}}}}``. The near/far split is
    at the **global median** age from baseline over all regions and subjects (near <= median).

    NOTE(paper): (Q8-related) for ``"latest"`` the "age from baseline" is the age minus the
    subject's *smallest prediction age* (not the baseline scan); for ``"earliest"`` it is the age
    since the first scan and only points with age > ``earliest_min_age`` (4 y, hard-coded in the
    legacy code) are analysed. Entries are ``(n_cond, idx, mae, age, dx, sid)``.
    """
    if baseline_format not in BASELINE_FORMATS:
        raise ValueError(f"baseline_format must be one of {BASELINE_FORMATS}")
    latest = baseline_format == "latest"

    all_ages_from_baseline: list[float] = []
    for entries in results.values():
        ages_by_subject = defaultdict(list)
        for _, _, _, age, _, sid in entries:
            ages_by_subject[sid].append(age)
        for ages in ages_by_subject.values():
            if latest:
                b = min(ages)
                all_ages_from_baseline.extend([a - b for a in ages])
            else:
                all_ages_from_baseline.extend([a for a in ages if a > earliest_min_age])
    median = np.median(all_ages_from_baseline)
    log.info("Global median age from baseline: %.2f (n=%d)", median, len(all_ages_from_baseline))

    processed = {}
    for region, entries in results.items():
        region_data = defaultdict(list)
        for num_cond, idx, mae, age, dx, sid in entries:
            region_data[sid].append(
                {"num_conditioning_vols": num_cond, "index": idx, "mae": mae, "age": age,
                 "diagnosis": dx, "subject_id": sid}
            )  # fmt: skip
        structured = {}
        for sid, subj_entries in region_data.items():
            baseline = min(e["age"] for e in subj_entries) if latest else 0
            by_conditioning = defaultdict(lambda: {"near_term": [], "far_term": []})
            for e in subj_entries:
                afb = e["age"] - baseline
                if latest or afb > earliest_min_age:
                    term = "near_term" if afb <= median else "far_term"
                    by_conditioning[e["num_conditioning_vols"]][term].append(
                        {"mae": e["mae"], "prediction_time": afb}
                    )
            structured[sid] = {
                "baseline_age": baseline,
                "diagnosis": subj_entries[0]["diagnosis"],
                "by_conditioning": dict(by_conditioning),
            }
        processed[region] = structured
    return processed


def mae_across_conditioning(
    results_structured: dict, term_type: str, regions: Sequence[str] = REGIONS
) -> dict[str, dict[str, np.ndarray]]:
    """Fig. 3a data: per region, mean +- SEM (``std/sqrt(n)``, ddof=0) of the MAE vs. n_cond."""
    out = {}
    for region in regions:
        by = defaultdict(list)
        for info in results_structured[region].values():
            for k, pdict in info["by_conditioning"].items():
                by[k].extend(p["mae"] for p in pdict[term_type])
        ks = sorted(k for k in by if by[k])
        out[region] = {
            "x": np.array(ks),
            "mean": np.array([np.mean(by[k]) for k in ks]),
            "sem": np.array([np.std(by[k]) / np.sqrt(len(by[k])) for k in ks]),
        }
    return out


# --------------------------------------------------------------------------------------------
# Progression-rate shift analysis
# --------------------------------------------------------------------------------------------
def compute_progression_rates(
    filtered: list[dict], real_vols_dir: str | Path, regions: Sequence[str] = REGIONS
) -> dict[str, dict[str, dict[str, float]]]:
    """Per subject and region: linear-fit slope (% TBV per year) of the real volumes.

    ``conditioning``: baseline and conditioning scans; ``prediction``: baseline and prediction
    scans; ``full``: all scans. Ages are absolute ages, points ordered conditioning, baseline,
    prediction (as in the legacy code; the slope does not depend on the order).
    """
    progression: dict = {}
    for sd in filtered:
        sid = sd["subject_id"]
        real = _real_volumes(real_vols_dir, sid)
        c_ages = _floats(sd["conditioning"]["ages_abs"])
        ages = np.concatenate(
            [c_ages, _floats(sd["baseline"]["age_abs0"]), _floats(sd["prediction"]["ages_abs"])]
        )
        idxs = np.array(
            list(sd["conditioning"]["indices"])
            + [sd["baseline"]["index"]]
            + list(sd["prediction"]["indices"])
        )
        n = len(sd["conditioning"]["indices"])
        progression[sid] = {}
        for region in regions:
            v = real[f"{region}_pct_tbv"].values[idxs]
            progression[sid][region] = {
                "conditioning": _slope(ages[: n + 1], v[: n + 1]),
                "prediction": _slope(ages[n:], v[n:]),
                "full": _slope(ages, v),
            }
    return progression


def mae_vs_progression(
    results_structured: dict,
    progression: dict,
    term_type: str | None = None,
    diagnosis_filter: str | None = None,
    use_prior: bool = False,
    regions: Sequence[str] = REGIONS,
) -> dict[str, dict]:
    """MAE vs |change in progression rate| (Fig. 3b; ``analyze_mae_vs_progression_...``).

    x = |prediction rate - conditioning rate| of the real volumes; y = mean MAE of the posterior
    with the largest number of conditioning scans (or the prior, ``use_prior``), over all
    prediction points or one ``term_type`` ("near_term"/"far_term"). Subjects need both prior and
    posterior points. Returns per region ``x, y, dx, subject_ids, num_subjects, correlation
    (Pearson r), slope`` (None if fewer than 2 subjects).
    """
    out = {}
    for region in regions:
        xs, ys, dxs, sids = [], [], [], []
        for sid, info in results_structured[region].items():
            if diagnosis_filter and info["diagnosis"] != diagnosis_filter:
                continue
            if sid not in progression:
                continue
            p = progression[sid][region]
            bc = info["by_conditioning"]
            if not bc or not bc.get(0):
                continue
            cond_keys = [k for k in bc if k > 0]
            if not cond_keys or not bc.get(max(cond_keys)):
                continue
            prior_data, post_data = bc[0], bc[max(cond_keys)]
            if term_type is None:
                prior_maes = prior_data["near_term"] + prior_data["far_term"]
                post_maes = post_data["near_term"] + post_data["far_term"]
            else:
                prior_maes, post_maes = prior_data[term_type], post_data[term_type]
            if prior_maes and post_maes:
                xs.append(abs(p["prediction"] - p["conditioning"]))
                ys.append(np.mean([q["mae"] for q in (prior_maes if use_prior else post_maes)]))
                dxs.append(info["diagnosis"])
                sids.append(sid)
        x, y = np.array(xs), np.array(ys)
        entry = {"x": x, "y": y, "dx": dxs, "subject_ids": sids, "num_subjects": len(xs),
                 "correlation": None, "slope": None}  # fmt: skip
        if len(xs) >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0:  # constant input has no fit
            entry["correlation"] = float(np.corrcoef(x, y)[0, 1])
            entry["slope"] = float(np.polyfit(x, y, 1)[0])
        out[region] = entry
    return out


# --------------------------------------------------------------------------------------------
# Per-subject trajectories (Fig. 3c, Fig. 4 data)
# --------------------------------------------------------------------------------------------
def subject_trajectories(
    sd: dict,
    real_vols_dir: str | Path,
    seg_root: str | Path,
    conditioning_volumes: Sequence[int] = (3,),
    regions: Sequence[str] = REGIONS,
) -> dict:
    """Real, prior (ncond0) and posterior (ncond<n>) %TBV trajectories of one subject."""
    sid = sd["subject_id"]
    c_ages = _floats(sd["conditioning"]["ages_abs"])
    c_idx = np.array(sd["conditioning"]["indices"])
    order = np.argsort(c_ages)
    pred_ages = _floats(sd["prediction"]["ages_abs"])
    baseline_age = float(_floats(sd["baseline"]["age_abs0"])[0])
    all_ages = np.concatenate([c_ages[order], [baseline_age], pred_ages])
    all_idx = np.concatenate(
        [c_idx[order], [sd["baseline"]["index"]], sd["prediction"]["indices"]]
    ).astype(int)
    s = np.argsort(all_ages)
    all_ages, all_idx = all_ages[s], all_idx[s]

    real = _real_volumes(real_vols_dir, sid).reset_index(drop=True)
    tbv = real["TBV"].iloc[0]  # NOTE(paper): %TBV relative to the first real scan
    real = real.iloc[all_idx]
    seg_dir = Path(seg_root) / f"temp_seg_{sid}"
    prior = predicted_region_volumes(seg_dir / "ncond0" / "vols.csv", tbv)
    post = {
        n: predicted_region_volumes(seg_dir / f"ncond{n}" / "vols.csv", tbv)
        for n in conditioning_volumes
    }
    return {
        "all_ages": all_ages,
        "all_idx": all_idx,
        "baseline_age": baseline_age,
        "baseline_index": sd["baseline"]["index"],
        "pred_ages": pred_ages,
        "pred_indices": list(sd["prediction"]["indices"]),
        "real": {r: real[f"{r}_pct_tbv"].values for r in regions},
        "prior": {r: prior[f"{r}_pct_tbv"].values for r in regions},
        "post": {n: {r: v[f"{r}_pct_tbv"].values for r in regions} for n, v in post.items()},
    }


# --------------------------------------------------------------------------------------------
# Config-driven entry point (``mrextrap eval-multi``)
# --------------------------------------------------------------------------------------------
def diagnosis_lookup_from_csv(csv_path: str | Path) -> dict[str, str]:
    """Subject-level diagnosis (Dementia > MCI > CN) from a split CSV (``subject_id``, ``DX_filled``)."""
    import pandas as pd

    from mrextrap.data.splits import diagnosis_label

    df = pd.read_csv(csv_path)
    return {sid: diagnosis_label(g["DX_filled"]) for sid, g in df.groupby("subject_id")}


def summarize(processed: dict, progression: dict, filtered: list[dict]) -> dict:
    """JSON-serialisable headline numbers: subject count, mean MAE per n_cond, Fig. 3b r values."""
    across = {
        term: {
            r: {"n_cond": d["x"].tolist(), "mean": d["mean"].tolist(), "sem": d["sem"].tolist()}
            for r, d in mae_across_conditioning(processed, term).items()
        }
        for term in ("near_term", "far_term")
    }
    shift = mae_vs_progression(processed, progression)
    return {
        "n_subjects": len(filtered),
        "mae_across_conditioning": across,
        "mae_vs_progression": {
            r: {k: d[k] for k in ("num_subjects", "correlation", "slope")} for r, d in shift.items()
        },
    }


def _save_figure_data(
    fmt_dir: Path,
    processed: dict,
    progression: dict,
    filtered: list[dict],
    real_vols_dir: str | Path,
    conditioning_years: Sequence[float],
    n_trajectories: int = 2,
) -> None:
    """Write ``fmt_dir/figure_data.pkl`` (Fig. 3 panel data, read by ``figures.caches``).

    Contains panel a (``across``: MAE vs number of conditioning volumes per term), panel b
    (``shift``: MAE vs progression-rate shift) and, when three volumes were conditioned on, the
    trajectories of the first ``n_trajectories`` subjects (panel c).
    """
    import pickle

    data = {
        "across": {t: mae_across_conditioning(processed, t) for t in ("near_term", "far_term")},
        "shift": mae_vs_progression(processed, progression),
        "trajectories": {},
    }
    if len(conditioning_years) == 3:  # Fig. 3c shows the posterior given 3 volumes
        for sd in filtered[:n_trajectories]:
            data["trajectories"][sd["subject_id"]] = subject_trajectories(
                sd, real_vols_dir, fmt_dir, conditioning_volumes=(3,)
            )
    with open(fmt_dir / "figure_data.pkl", "wb") as f:
        pickle.dump(data, f)


def evaluate_multivolume(
    latent_dict: dict,
    dx_lookup,
    global_params,
    vae,
    segmenter,
    real_vols_dir: str | Path,
    results_dir: str | Path,
    conditioning_years: Sequence[float] = DEFAULT_CONDITIONING_YEARS,
    min_future_years: float = 2,
    min_baseline_years: float | None = 3,
    baseline_formats: Sequence[str] = BASELINE_FORMATS,
    device: str = "cpu",
    relative_obs_var: float = 1.0,
    raise_on_error: bool = False,
) -> dict[str, dict]:
    """Run ``evaluate_conditioning`` for each baseline format, then process and summarise.

    Writes ``results_dir/<format>/results_var1.00.json`` and ``results_dir/<format>/summary.json``.
    """
    earliest, latest = prepare_data_unified(
        latent_dict, dx_lookup, min_future_years, conditioning_years, min_baseline_years
    )
    filtered_by_format = {"earliest": earliest, "latest": latest}
    out = {}
    for fmt in baseline_formats:
        fmt_dir = Path(results_dir) / fmt
        filtered = filtered_by_format[fmt]
        results = evaluate_conditioning(
            filtered, global_params, vae, segmenter, real_vols_dir, fmt_dir, device=device,
            relative_obs_var=relative_obs_var, raise_on_error=raise_on_error,
        )  # fmt: skip
        processed = process_results_unified(results, fmt)
        # NOTE(paper) (Q8): Fig. 3b uses the earliest-baseline results, 3a/3c the latest ones.
        progression = compute_progression_rates(earliest, real_vols_dir)
        summary = summarize(processed, progression, filtered)
        with open(fmt_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        _save_figure_data(
            fmt_dir, processed, progression, filtered, real_vols_dir, conditioning_years
        )
        out[fmt] = summary
    return out


def run_multivolume_eval(cfg) -> dict[str, dict]:
    """Config entry point: reads paths from ``cfg.paths`` and ``cfg.eval`` (configs/eval_multivolume.yaml)."""
    from mrextrap.encoding import load_autoencoder
    from mrextrap.io import load_latent_dict
    from mrextrap.segmentation.volumes import make_segmenter

    paths, ev = cfg.paths, cfg.eval
    device = ev.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    latent_dict = load_latent_dict(Path(paths.latents_dir) / "test_latent_betas_l1.pt")
    vae = load_autoencoder(paths.run_dir, device=device)
    segmenter = make_segmenter(cfg, device)
    return evaluate_multivolume(
        latent_dict,
        diagnosis_lookup_from_csv(Path(paths.splits_dir) / "test.csv"),
        Path(paths.priors_dir) / "global_params.pt",
        vae,
        segmenter,
        Path(paths.volumes_dir) / "test_real",
        ev.results_dir,
        conditioning_years=list(ev.conditioning_years),
        min_future_years=ev.min_future_years,
        min_baseline_years=ev.get("min_baseline_years"),
        baseline_formats=list(ev.get("baseline_formats", BASELINE_FORMATS)),
        device=device,
        relative_obs_var=ev.get("obs_var_scale", 1.0),
    )


__all__ = [
    "BASELINE_FORMATS",
    "compute_progression_rates",
    "diagnosis_lookup_from_csv",
    "evaluate_conditioning",
    "evaluate_multivolume",
    "evaluate_predictions_per_future_timepoint",
    "load_global_params",
    "mae_across_conditioning",
    "mae_vs_progression",
    "prepare_data_unified",
    "process_results_unified",
    "run_multivolume_eval",
    "subject_trajectories",
    "summarize",
]
