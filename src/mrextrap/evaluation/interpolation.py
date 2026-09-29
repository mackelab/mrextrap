"""Longitudinal interpolation between the first and last scan of a subject (paper Fig. 2b/c).

NOTE(paper): Q6. The paper describes interpolation in *latent* space; the research code
(``eval_autoencoder_interp.py``, interpolation notebook cells 6/13) interpolates in *voxel* space and
segments the interpolated inputs (the autoencoder reconstruction is computed but never used). Both
are implemented: ``mode="latent"`` (paper description, default) and ``mode="voxel"`` (old code).
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from scipy.stats import linregress
from tqdm.auto import tqdm

from mrextrap.segmentation.volumes import REGIONS, real_region_volumes

logger = logging.getLogger(__name__)


def interpolate_volumes(
    vae, x_start: torch.Tensor, x_end: torch.Tensor, alphas: torch.Tensor, mode: str = "latent"
) -> torch.Tensor:
    """Interpolated volumes ``[len(alphas), 1, D, H, W]`` between ``x_start`` and ``x_end`` (``[1,1,D,H,W]``).

    ``"voxel"``: ``x_start + a * (x_end - x_start)`` (the volumes that were segmented in the old code).
    ``"latent"``: encode both (posterior mean), interpolate ``z``, decode.
    """
    if mode == "voxel":
        return torch.cat([x_start + a * (x_end - x_start) for a in alphas], dim=0)
    if mode == "latent":
        with torch.no_grad():
            z_start, z_end = vae.encode(x_start)[0], vae.encode(x_end)[0]
            return torch.cat(
                [vae.decode((z_start + a * (z_end - z_start)).float()).cpu() for a in alphas], dim=0
            )
    raise ValueError(f"mode must be 'latent' or 'voxel', got {mode!r}")


def interpolation_volumes(
    vae,
    loader,
    segmenter,
    out_dir: str | Path,
    mode: str = "latent",
    n_alpha: int = 10,
    device: str = "cpu",
    regions: Sequence[str] = REGIONS,
) -> dict[str, dict]:
    """Interpolate between first and last scan of every subject and segment the ``n_alpha`` volumes.

    Writes ``out_dir/<sid>/vols.csv`` (plus NIfTIs) per subject with at least two scans. Returns
    ``{sid: {"age_gap": years between first and last scan or None, "pct_change": {region: array
    [n_alpha]}}}`` where ``pct_change`` is the %TBV change from the first interpolated volume.
    """
    out_dir = Path(out_dir)
    vae = vae.to(device).eval()
    alphas = torch.linspace(0, 1, n_alpha)
    out = {}
    for batch in tqdm(loader, total=len(loader)):
        x = batch["vols"]
        assert x.shape[0] == 1, "batch size must be 1 (one subject series per batch)"
        sid = batch["subject_id"]
        sid = str(sid[0] if isinstance(sid, (list, tuple)) else sid)
        x = rearrange(x, "b s c d h w -> (b s) c d h w")
        if len(x) < 2:
            continue
        xi = interpolate_volumes(
            vae, x[:1].to(device), x[-1:].to(device), alphas.to(device), mode
        ).cpu()
        seg_dir = out_dir / sid
        if seg_dir.exists():
            shutil.rmtree(seg_dir)
        vols_csv = segmenter.segment([v[0].numpy() for v in xi], seg_dir)
        vols = real_region_volumes(vols_csv, regions)
        ages = batch.get("ages", None)
        age_gap = float(ages.reshape(-1)[-1] - ages.reshape(-1)[0]) if ages is not None else None
        out[sid] = {
            "age_gap": age_gap,
            "pct_change": {
                r: vols[f"{r}_pct_tbv"].values - vols[f"{r}_pct_tbv"].iloc[0] for r in regions
            },
        }
    return out


def linearity_r2(pct_change: np.ndarray, alphas: np.ndarray | None = None) -> float:
    """R^2 of a linear fit of the volume change against ``alpha`` (``linregress`` r squared)."""
    if alphas is None:
        alphas = np.linspace(0, 1, len(pct_change))
    return float(linregress(alphas, pct_change).rvalue ** 2)


def interpolation_r2(
    results: dict[str, dict],
    max_age_gap: float | None = 4.1,
    min_r2: float | None = None,
    regions: Sequence[str] = REGIONS,
) -> dict[str, np.ndarray]:
    """Per-region linearity R^2 over subjects (Fig. 2c).

    ``max_age_gap`` drops subjects whose first-to-last scan gap exceeds it (paper: 4.1 y);
    ``min_r2`` optionally drops poorly linear subjects (the notebook plot used 0.7).
    """
    r2s = {r: [] for r in regions}
    for res in results.values():
        if max_age_gap is not None and res["age_gap"] is not None and res["age_gap"] > max_age_gap:
            continue
        for r in regions:
            v = linearity_r2(res["pct_change"][r])
            if min_r2 is None or v >= min_r2:
                r2s[r].append(v)
    return {r: np.array(v) for r, v in r2s.items()}


def run_interpolation(cfg) -> dict:
    """Config entry point: interpolate with the paper AE in ``<run_dir>/autoencoder``.

    Uses ``cfg.interp`` (``mode`` default "latent", ``split`` default "test", ``n_alpha``, ``device``).
    """
    from mrextrap.data.datasets import get_longitudinal_dataloaders
    from mrextrap.encoding import load_autoencoder
    from mrextrap.segmentation.volumes import make_segmenter

    ev = cfg.get("interp", None) or {}
    run_dir = Path(str(cfg.paths.run_dir))
    device = ev.get("device", "cpu")
    split = ev.get("split", "test")
    mode = ev.get("mode", "latent")
    vae = load_autoencoder(
        ev.get("autoencoder", None) or run_dir / "autoencoder",
        device=device,
        model_cfg=cfg.get("model", None),
    )
    cfg = cfg.copy()
    cfg.training.batch_size = 1
    loaders = dict(
        zip(
            ("train", "val", "test"),
            get_longitudinal_dataloaders(cfg, num_workers=cfg.training.get("num_workers", 0)),
            strict=True,
        )
    )
    segmenter = make_segmenter(cfg, device)
    out_dir = Path(str(cfg.paths.volumes_dir)) / f"{split}_interp_{mode}"
    results = interpolation_volumes(
        vae,
        loaders[split],
        segmenter,
        out_dir,
        mode=mode,
        n_alpha=int(ev.get("n_alpha", 10)),
        device=device,
    )
    import pickle

    with open(out_dir / "interpolation.pkl", "wb") as f:  # read by ``figures.caches`` (Fig. 2)
        pickle.dump({"mode": mode, "results": results}, f)
    return results
