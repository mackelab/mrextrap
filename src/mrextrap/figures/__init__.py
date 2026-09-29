"""Paper figures (Fig. 2-5) and Table 1 rendering.

Each `make_figN(data, out_path)` is a pure plotting function: it reads only the cached
intermediate dict it is given (no GPU, no model code) and writes a PDF.
"""

from __future__ import annotations

from pathlib import Path

from .common import load_cache
from .fig2 import make_fig2
from .fig3_multivol import make_fig3
from .fig4_qualitative import make_fig4
from .fig5_beta import make_fig5
from .table1 import render_table1, write_table1

__all__ = [
    "make_all_figures",
    "make_fig2",
    "make_fig3",
    "make_fig4",
    "make_fig5",
    "render_table1",
    "write_table1",
]

_FIGURES = {"fig2": make_fig2, "fig3": make_fig3, "fig4": make_fig4, "fig5": make_fig5}


def make_all_figures(cfg) -> list[Path]:
    """CLI entry (`mrextrap figures`): render every figure whose cache file exists.

    Missing caches for Fig. 2, 3 and 5 are first built from the stage outputs (`caches.py`).
    Reads `<run_dir>/figures/cache/figN.pkl` (dicts as documented on `make_figN`; for Fig. 4 the
    `vols` entry may be a path to an .npz) and writes `<run_dir>/figures/figN.pdf`.
    """
    import numpy as np

    from .caches import build_figure_caches

    build_figure_caches(cfg)
    run_dir = Path(cfg.paths.run_dir)
    cache_dir, out_dir = run_dir / "figures" / "cache", run_dir / "figures"
    written = []
    for name, fn in _FIGURES.items():
        src = cache_dir / f"{name}.pkl"
        if not src.exists():
            print(f"[figures] skip {name}: {src} not found")
            continue
        data = load_cache(src)
        if name == "fig4" and isinstance(data.get("vols"), (str, Path)):
            data["vols"] = np.load(data["vols"])
        fn(data, out_dir / f"{name}.pdf")
        written.append(out_dir / f"{name}.pdf")
    return written
