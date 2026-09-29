"""Single-volume evaluation (Table 1): extrapolate one baseline scan, compare regional volumes.

Ports ``prepare_data_for_daninet_evaluation``, ``evaluate_predictions_daninet``, ``main_evaluation`` and
``main_evaluation_global_prior`` of ``eval_unet_additional.py`` and the diffusion path of
``eval_beta_diffusion.ipynb`` (cells 8, 25-29). The evaluation pairs (subject, baseline age, target
age) are read from a user-supplied file and never hard-coded.
"""

from __future__ import annotations

import csv
import itertools
import json
import logging
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from mrextrap.prediction import (
    DiffusionPrior,
    GlobalPrior,
    UNetPrior,
    delta_t_for_targets,
    extrapolate,
)
from mrextrap.segmentation.volumes import (
    REGIONS,
    predicted_region_volumes,
    real_region_volumes,
)

logger = logging.getLogger(__name__)

METHODS = ("global", "unet", "diffusion")
_FILENAME_SID = re.compile(r"ADNI_(\d+_S_\d+)")


# ---------------------------------------------------------------------------------------------
# Evaluation pairs
# ---------------------------------------------------------------------------------------------
def parse_daninet_filename(name: str) -> tuple[str, float]:
    """``<age>_<..>_ADNI_<sid>_MR_...`` -> ``(sid, baseline_age)`` (DANINet test-pair file names)."""
    match = _FILENAME_SID.search(name)
    if match is None:
        raise ValueError(f"No ADNI subject id found in file name {name!r}")
    return match.group(1), float(name.split("_")[0])


def load_daninet_pairs(path: str | Path) -> list[tuple[str, float, float]]:
    """Read ``(subject_id, baseline_age, target_age)`` pairs from a user file.

    Accepted rows (an optional header row is skipped): ``sid, baseline_age, target_age`` or the
    DANINet format ``<file name>, target_age`` where the file name is
    ``<baseline_age>_..._ADNI_<sid>_...``. Sorted by subject id; a subject may appear only once.
    """
    pairs = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            row = [c.strip() for c in row if c.strip() != ""]
            if not row:
                continue
            if len(row) >= 3:
                try:
                    pairs.append((row[0], float(row[1]), float(row[2])))
                except ValueError:
                    continue  # header row
            elif len(row) == 2:
                try:
                    sid, age_a = parse_daninet_filename(row[0])
                    pairs.append((sid, age_a, float(row[1])))
                except ValueError:
                    continue
    pairs.sort(key=lambda p: p[0])
    for a, b in itertools.pairwise(pairs):
        if a[0] == b[0]:
            raise ValueError(f"Subject {a[0]} appears more than once in {path}")
    return pairs


def prepare_daninet_eval(
    latent_dict: Mapping,
    pairs: Sequence[tuple[str, float, float]],
    dx_lookup: Mapping[str, str] | Callable[[str], str] | None = None,
) -> list[dict]:
    """Pick baseline and target scan of every listed subject (latent-dict order).

    The scans closest in absolute age to ``baseline_age`` / ``target_age`` are used (``argmin``,
    as in the legacy code). ``dx_lookup`` maps subject id -> "CN"/"MCI"/"Dementia" (default "NA").
    """
    by_sid = {p[0]: p for p in pairs}
    subjects = []
    for i, sid_list in enumerate(latent_dict["subject_ids"]):
        sid = sid_list[0]
        if sid not in by_sid:
            continue
        _, age_a, age_b = by_sid[sid]
        ages = latent_dict["ages"][i][0]
        ages_abs = latent_dict["ages_abs"][i][0]
        zs = latent_dict["zs"][i]
        a_idx = int(np.argmin(np.abs(ages_abs.numpy() - age_a)))
        b_idx = int(np.argmin(np.abs(ages_abs.numpy() - age_b)))
        if callable(dx_lookup):
            dx = dx_lookup(sid)
        elif dx_lookup is not None:
            dx = dx_lookup[sid]
        else:
            dx = "NA"
        subjects.append(
            {
                "subject_id": sid,
                "baseline": {
                    "z0": zs[a_idx : a_idx + 1],
                    "age0": ages[a_idx : a_idx + 1],
                    "age_abs0": ages_abs[a_idx : a_idx + 1],
                    "index": a_idx,
                },
                "prediction": {
                    "zs": zs[b_idx : b_idx + 1],
                    "ages": ages[b_idx : b_idx + 1],
                    "ages_abs": ages_abs[b_idx : b_idx + 1],
                    "indices": [b_idx],
                },
                "disease_status": dx,
                "age_errors": (
                    float(abs(ages_abs[a_idx] - age_a)),
                    float(abs(ages_abs[b_idx] - age_b)),
                ),
            }
        )
    missing = set(by_sid) - {s["subject_id"] for s in subjects}
    if missing:
        logger.warning("%d listed subjects are not in the latent dict", len(missing))
    return subjects


def count_baseline_not_first(subjects: Sequence[dict]) -> int:
    """Number of subjects whose baseline scan is not the subject's first scan (affects Q1)."""
    return sum(1 for s in subjects if s["baseline"]["index"] > 0)


# ---------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------
def predict_trajectory(
    method: str,
    subject: dict,
    predictor,
    delta_t_from_first_scan: bool = True,
    device: str = "cpu",
    num_samples: int = 5,
    seed: int = 0,
) -> torch.Tensor:
    """Predicted latents ``[T, C, D, H, W]`` at the target scan(s) of ``subject``."""
    z0 = subject["baseline"]["z0"]
    if method == "global":
        beta, _ = predictor.mean_var()
    elif method == "unet":
        beta, _ = predictor.predict(z0, subject["baseline"]["age_abs0"], device=device)
    elif method == "diffusion":
        betas = predictor.sample(
            z0, subject["baseline"]["age_abs0"], num_samples=num_samples, seed=seed, device=device
        )
        beta = betas.mean(dim=1)
    else:
        raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")
    if method == "diffusion":
        # NOTE(paper): Q1. The diffusion path always uses dt = ages[end] - ages[start] (correct).
        dt = delta_t_for_targets(
            subject["prediction"]["ages"],
            subject["baseline"]["age0"],
            delta_t_from_first_scan=False,
        )
    else:
        # NOTE(paper): Q1. Global/UNet: dt = age since the subject's first scan of the target
        # (default), which equals target - baseline only if the baseline is the first scan.
        dt = delta_t_for_targets(
            subject["prediction"]["ages"], subject["baseline"]["age0"], delta_t_from_first_scan
        )
    return extrapolate(z0, beta, dt)


def evaluate_subject(
    z_pred: torch.Tensor,
    subject: dict,
    vae,
    segmenter,
    real_vols_dir: str | Path,
    out_dir: str | Path,
    device: str = "cpu",
    cleanup: bool = True,
    regions: Sequence[str] = REGIONS,
) -> dict[str, float]:
    """Decode predicted latents, segment them and return the MAE per region (% TBV).

    Port of ``evaluate_predictions_daninet``: the %TBV of the prediction is relative to the TBV of
    the first *real* scan; real volumes are taken at the target scan indices.
    """
    if z_pred.ndim != 5:
        raise ValueError(f"Expected z_pred with 5 dimensions [T,C,D,H,W], got {z_pred.ndim}")
    sid = subject["subject_id"]
    real = real_region_volumes(Path(real_vols_dir) / sid / "vols.csv", regions)
    tbv = real["TBV"].iloc[0]

    x_preds = []
    with torch.no_grad():
        for z in z_pred:
            x_preds.append(vae.decode(z.unsqueeze(0).float().to(device)).cpu())
    x_pred = torch.cat(x_preds).numpy()

    seg_path = Path(out_dir) / f"temp_seg_{sid}"
    if seg_path.exists():  # never reuse stale segmentations
        shutil.rmtree(seg_path)
    vols_csv = segmenter.segment([x[0] for x in x_pred], seg_path)
    pred = predicted_region_volumes(vols_csv, tbv, regions)

    out = {}
    for region in regions:
        real_vals = real[f"{region}_pct_tbv"].iloc[subject["prediction"]["indices"]].values
        out[region] = float(np.mean(np.abs(real_vals - pred[f"{region}_pct_tbv"].values)))
    if cleanup:
        shutil.rmtree(seg_path)
    return out


def evaluate_single_volume(
    method: str,
    subjects: Sequence[dict],
    predictor,
    vae,
    segmenter,
    real_vols_dir: str | Path,
    out_dir: str | Path,
    delta_t_from_first_scan: bool = True,
    device: str = "cpu",
    num_samples: int = 5,
    seed: int = 0,
    cleanup: bool = True,
    skip_failures: bool = False,
    regions: Sequence[str] = REGIONS,
) -> dict[str, list[tuple[float, str, str]]]:
    """Evaluate one method on all prepared subjects; writes ``out_dir/results.json``.

    Args:
        method: "global" (``GlobalPrior``), "unet" (``UNetPrior``) or "diffusion" (``DiffusionPrior``).
        subjects: output of ``prepare_daninet_eval``.
        real_vols_dir: ``volumes/<split>_real`` with ``<sid>/vols.csv`` per subject.
        delta_t_from_first_scan: Q1 switch (default True = paper behaviour), see ``predict_trajectory``.
        skip_failures: legacy behaviour was to log and skip subjects that raised; default re-raises.

    Returns ``{region: [(mae, disease_status, subject_id), ...]}`` in subject order (the same format
    as ``results.json``, with the subject id added so that methods can be paired safely).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_affected = count_baseline_not_first(subjects)
    logger.info(
        "%s: %d subjects, %d with baseline index > 0 (delta_t_from_first_scan=%s)",
        method,
        len(subjects),
        n_affected,
        delta_t_from_first_scan,
    )
    results = {region: [] for region in regions}
    failed = 0
    for subject in subjects:
        try:
            z_pred = predict_trajectory(
                method, subject, predictor, delta_t_from_first_scan, device, num_samples, seed
            )
            maes = evaluate_subject(
                z_pred, subject, vae, segmenter, real_vols_dir, out_dir, device, cleanup, regions
            )
        except Exception:
            if not skip_failures:
                raise
            logger.exception("error processing subject %s", subject["subject_id"])
            failed += 1
            continue
        for region in regions:
            results[region].append((maes[region], subject["disease_status"], subject["subject_id"]))
    with open(out_dir / "results.json", "w") as f:
        json.dump(
            {r: [(float(m), dx, sid) for m, dx, sid in rows] for r, rows in results.items()}, f
        )
    logger.info("%s: %d/%d subjects evaluated", method, len(subjects) - failed, len(subjects))
    return results


# ---------------------------------------------------------------------------------------------
# Config entry point
# ---------------------------------------------------------------------------------------------
def _load_test_latents(latents_dir: Path, split: str) -> dict:
    from mrextrap.io import load_latent_dict

    for name in (f"{split}_zs_ages.pt", f"{split}_latent_betas_l1.pt", f"{split}_latent_betas.pt"):
        if (latents_dir / name).is_file():
            return load_latent_dict(latents_dir / name)
    raise FileNotFoundError(f"No latent file for split {split!r} in {latents_dir}")


def _dx_lookup(split_csv: Path) -> dict[str, str]:
    import pandas as pd

    from mrextrap.data.splits import diagnosis_label

    df = pd.read_csv(split_csv)
    return {sid: diagnosis_label(g.DX_filled) for sid, g in df.groupby("subject_id")}


def run_eval_single(cfg) -> dict[str, Path]:
    """``mrextrap eval-single``: evaluate ``cfg.eval.methods`` on the ``eval.split`` (default test).

    Reads latents, priors, real volumes and the DANINet pairs from the run directory
    (``paths.*``, ``eval.*``) and writes ``<results_dir>/<method>/results.json``. The autoencoder is
    the paper AE from ``<run_dir>/autoencoder`` (or ``eval.autoencoder``).
    """
    from mrextrap.encoding import load_autoencoder
    from mrextrap.segmentation.volumes import make_segmenter

    ev = cfg.eval
    run_dir = Path(str(cfg.paths.run_dir))
    split = ev.get("split", "test")
    device = ev.get("device", "cpu")
    vae = load_autoencoder(ev.get("autoencoder", None) or run_dir / "autoencoder", device=device)
    latent_dict = _load_test_latents(Path(str(cfg.paths.latents_dir)), split)
    pairs = load_daninet_pairs(str(ev.daninet_pairs))
    dx = _dx_lookup(Path(str(cfg.paths.splits_dir)) / f"{split}.csv")
    subjects = prepare_daninet_eval(latent_dict, pairs, dx)
    segmenter = make_segmenter(cfg, device)
    real_vols_dir = Path(str(cfg.paths.volumes_dir)) / f"{split}_real"
    priors_dir = Path(str(cfg.paths.priors_dir))
    results_dir = Path(str(ev.results_dir))

    written = {}
    for method in ev.methods:
        if method == "global":
            predictor = GlobalPrior.from_file(priors_dir / "global_params.pt")
        else:
            from omegaconf import OmegaConf

            norm = torch.load(priors_dir / "normalization.pt", map_location="cpu")
            if method == "unet":
                from mrextrap.training.unet_prior import load_unet_prior

                prior_dir = run_dir / "unet_prior"
                prior_cfg = OmegaConf.load(prior_dir / "config.yaml")
                model = load_unet_prior(
                    prior_cfg, prior_dir / "checkpoints" / str(ev.unet_checkpoint)
                ).to(device)
                predictor = UNetPrior(model, norm)
            else:
                from mrextrap.training.diffusion_prior import load_diffusion_prior

                prior_dir = run_dir / "diffusion_prior"
                prior_cfg = OmegaConf.load(prior_dir / "config.yaml")
                model = load_diffusion_prior(
                    prior_cfg, prior_dir / "checkpoints" / str(ev.diffusion_checkpoint)
                ).to(device)
                predictor = DiffusionPrior(model, norm, NT=int(prior_cfg.training.NT))
        evaluate_single_volume(
            method,
            subjects,
            predictor,
            vae,
            segmenter,
            real_vols_dir,
            results_dir / method,
            delta_t_from_first_scan=bool(ev.get("delta_t_from_first_scan", True)),
            device=device,
            num_samples=int(ev.get("num_samples", 5)),
            seed=int(ev.get("seed", 0)),
        )
        written[method] = results_dir / method / "results.json"
    return written
