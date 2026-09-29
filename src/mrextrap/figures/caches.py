"""Build the figure cache files (``<run_dir>/figures/cache/figN.pkl``) from the stage outputs.

Each builder returns the data dict documented in the corresponding ``make_figN`` or ``None`` when
the stage outputs it needs are missing. Fig. 4 needs decoded volumes of the chosen subject and is
not built here (see ``make_fig4``).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from mrextrap.io import load_latent_dict


def _path(cfg, key: str) -> Path | None:
    value = cfg.paths.get(key, None)
    return Path(str(value)) if value else None


def _load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_fig5(cfg) -> dict | None:
    """Betas, absolute ages and diagnoses of the train split (``fit-betas`` output)."""
    latents, splits = _path(cfg, "latents_dir"), _path(cfg, "splits_dir")
    if latents is None or splits is None:
        return None
    path, csv = latents / "train_latent_betas_l1.pt", splits / "train.csv"
    if not (path.is_file() and csv.is_file()):
        return None
    from mrextrap.evaluation.multivolume import diagnosis_lookup_from_csv

    d = load_latent_dict(path)
    dx = diagnosis_lookup_from_csv(csv)
    return {
        "betas": list(d["betas"]),
        "ages_abs": list(d["ages_abs"]),
        "diseases": [dx[ids if isinstance(ids, str) else ids[0]] for ids in d["subject_ids"]],
    }


def build_fig3(cfg) -> dict | None:
    """Panels a-c from ``eval/multivolume/{earliest,latest}/figure_data.pkl`` (``eval-multi``)."""
    default = Path(str(cfg.paths.run_dir)) / "eval" / "multivolume"
    root = Path(str(cfg.get("figures", {}).get("multivolume_dir", default)))
    files = {fmt: root / fmt / "figure_data.pkl" for fmt in ("earliest", "latest")}
    if not all(f.is_file() for f in files.values()):
        return None
    data = {fmt: _load(f) for fmt, f in files.items()}
    panel_a = {
        (fmt, term): data[fmt]["across"][term]
        for fmt in ("latest", "earliest")
        for term in ("near_term", "far_term")
    }
    trajectories = data["latest"]["trajectories"]
    if not trajectories:
        return None
    return {
        "panel_a": panel_a,
        "panel_b": data["earliest"]["shift"],  # NOTE(paper): Q8, panel b uses the earliest baseline
        "panel_c": {sid: traj for sid, traj in list(trajectories.items())[:2]},
    }


def build_fig2(cfg) -> dict | None:
    """Latent channels, interpolation curves and latent sequences (``interpolation`` + ``encode``)."""
    ip = cfg.get("interp", None) or {}
    split, mode = ip.get("split", "test"), ip.get("mode", "latent")
    volumes, latents = _path(cfg, "volumes_dir"), _path(cfg, "latents_dir")
    if volumes is None or latents is None:
        return None
    src = volumes / f"{split}_interp_{mode}" / "interpolation.pkl"
    zs_path = latents / f"{split}_zs_ages.pt"
    if not (src.is_file() and zs_path.is_file()):
        return None
    saved = _load(src)
    results = saved["results"]
    regions = list(next(iter(results.values()))["pct_change"])
    zs = list(load_latent_dict(zs_path)["zs"])
    return {
        "latent": np.asarray(zs[0][0]),
        "interp": {
            "mode": saved["mode"],
            "curves": {
                r: [np.asarray(v["pct_change"][r]) for v in results.values()] for r in regions
            },
            "age_gaps": [v["age_gap"] for v in results.values()],
        },
        "zs": [np.asarray(z) for z in zs],
    }


BUILDERS = {"fig2": build_fig2, "fig3": build_fig3, "fig5": build_fig5}


def build_figure_caches(cfg) -> list[Path]:
    """Write every cache whose inputs exist (existing caches are kept); returns the new files."""
    cache_dir = Path(str(cfg.paths.run_dir)) / "figures" / "cache"
    written = []
    for name, builder in BUILDERS.items():
        target = cache_dir / f"{name}.pkl"
        if target.exists():
            continue
        data = builder(cfg)
        if data is None:
            continue
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            pickle.dump(data, f)
        written.append(target)
    return written
