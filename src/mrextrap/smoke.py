"""``mrextrap smoke``: the whole pipeline on a tiny synthetic cohort (CPU, no data, no weights).

synthetic cohort -> autoencoder (2 steps) -> encode -> fit betas -> Global prior -> UNet and
diffusion priors (2 epochs on 4x4x4x4 latents) -> autoencoder metrics, real volumes, interpolation
-> single- and multi-volume evaluation -> figures and Table 1. Segmentation uses ``FakeSegmenter``.
The numbers are meaningless; the run checks that every stage reads what the previous one wrote.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from mrextrap.config import load_config

logger = logging.getLogger(__name__)


def _smoke_config_dir() -> Path:
    """``configs/smoke`` of a source checkout, else the copy shipped in the wheel (``mrextrap/_configs``)."""
    here = Path(__file__).resolve()
    for cand in (here.parents[2] / "configs" / "smoke", here.parent / "_configs" / "smoke"):
        if cand.is_dir():
            return cand
    return here.parent / "_configs" / "smoke"


SMOKE_CONFIG_DIR = _smoke_config_dir()
N_SUBJECTS = 24
SCANS_PER_SUBJECT = (6, 7)


def _stage_cfg(name: str, run_dir: Path) -> DictConfig:
    path = SMOKE_CONFIG_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; reinstall the package or run from a source checkout"
        )
    return load_config(path, [f"paths.run_dir={run_dir}"])


def write_pairs(splits_csv: Path, out_csv: Path) -> Path:
    """``sid, baseline_age, target_age`` (first and last scan of every subject) for eval-single."""
    df = pd.read_csv(splits_csv)
    rows = [(sid, g.AGE_filled.min(), g.AGE_filled.max()) for sid, g in df.groupby("subject_id")]
    pd.DataFrame(rows, columns=["subject_id", "baseline_age", "target_age"]).to_csv(
        out_csv, index=False
    )
    return out_csv


def run_smoke(cfg: DictConfig | None = None) -> dict:
    """Run every stage in ``paths.run_dir`` (default: a fresh temporary directory).

    Returns ``{"run_dir", "seconds", "ae_metrics"}``.
    """
    from mrextrap.data.synthetic import make_synthetic_cohort
    from mrextrap.encoding import encode_splits
    from mrextrap.evaluation.autoencoder import run_eval_ae
    from mrextrap.evaluation.interpolation import run_interpolation
    from mrextrap.evaluation.multivolume import run_multivolume_eval
    from mrextrap.evaluation.single_volume import run_eval_single
    from mrextrap.evaluation.stats import run_table1
    from mrextrap.figures import make_all_figures
    from mrextrap.progression import run_fit_betas, run_global_prior
    from mrextrap.stages import run_real_volumes
    from mrextrap.training.autoencoder import train_autoencoder
    from mrextrap.training.diffusion_prior import train_diffusion_prior
    from mrextrap.training.unet_prior import train_unet_prior

    start = time.time()
    user = cfg.get("paths", {}).get("run_dir") if cfg is not None else None
    run_dir = Path(str(user)) if user else Path(tempfile.mkdtemp(prefix="mrextrap_smoke_"))
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("smoke run directory: %s", run_dir)

    make_synthetic_cohort(
        run_dir, n_subjects=N_SUBJECTS, scans_per_subject=SCANS_PER_SUBJECT, shape=(32, 32, 32)
    )
    ae = _stage_cfg("autoencoder", run_dir)
    train_autoencoder(ae)
    encode_splits(ae)
    run_fit_betas(ae)
    run_global_prior(ae)
    train_unet_prior(_stage_cfg("unet_prior", run_dir))
    train_diffusion_prior(_stage_cfg("diffusion_prior", run_dir))

    ev = _stage_cfg("eval", run_dir)
    ae_metrics = run_eval_ae(ae)  # also segments the real scans of the test split
    run_real_volumes(OmegaConf.merge(ev, {"real_volumes": {"splits": ["test"]}}))
    run_interpolation(ae)
    write_pairs(Path(str(ev.paths.splits_dir)) / "test.csv", Path(str(ev.eval.daninet_pairs)))
    run_eval_single(ev)
    run_multivolume_eval(
        OmegaConf.merge(ev, {"eval": {"results_dir": str(run_dir / "eval" / "multivolume")}})
    )
    make_all_figures(ev)
    run_table1(ev)
    expected = [
        run_dir / "autoencoder" / "model.safetensors",
        run_dir / "latents" / "test_latent_betas_l1.pt",
        run_dir / "priors" / "global_params.pt",
        run_dir / "eval" / "single_volume" / "diffusion" / "results.json",
        run_dir / "eval" / "multivolume" / "latest" / "summary.json",
        run_dir / "figures" / "fig3.pdf",
        run_dir / "tables" / "table1.csv",
    ]
    missing = [str(p) for p in expected if not p.is_file()]
    if missing:
        raise RuntimeError(f"smoke run finished without these outputs: {missing}")
    seconds = time.time() - start
    print(f"[smoke] all stages ran in {seconds:.0f} s; outputs in {run_dir}")
    return {"run_dir": run_dir, "seconds": seconds, "ae_metrics": ae_metrics}
