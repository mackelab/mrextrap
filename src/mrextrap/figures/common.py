"""Shared plotting helpers: style context, panel letters, saving, brain-slice utilities.

Everything here is CPU-only and headless.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np

STYLE_PATH = Path(__file__).with_name("style.mplstyle")
WIDTH = 5.08  # thesis \textwidth in inches (367 pt)

REGIONS = ("hippocampus", "ventricle", "cortex", "white matter")
REGION_LABEL = {
    "hippocampus": "hippocampus",
    "ventricle": "ventricle",
    "cortex": "grey matter",
    "white matter": "white matter",
}
DX_COLORS = {"CN": "green", "MCI": "darkorange", "Dementia": "darkred"}
C_REAL, C_PRIOR, C_POST = "black", "teal", "#005555"


@contextmanager
def figure_style():
    """Apply the bundled lab style (Arial 7 pt) for the duration of a `with` block."""
    from matplotlib import pyplot as plt

    with plt.style.context(str(STYLE_PATH)):
        yield


def panel_letter(target, letter: str, x: float = 0.0, y: float = 1.0) -> None:
    """Bold panel letter at the top-left of a Figure or SubFigure."""
    target.text(
        x,
        y,
        letter,
        transform=target.transSubfigure
        if hasattr(target, "transSubfigure")
        else target.transFigure,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="top",
    )


def save_figure(fig, out_path: str | Path, metadata: dict | None = None) -> Path:
    """Save `fig` to `out_path` (format from suffix; default PDF) and return the path."""
    out_path = Path(out_path)
    if out_path.suffix == "":
        out_path = out_path.with_suffix(".pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, metadata=metadata)
    return out_path


def load_cache(path: str | Path) -> dict:
    """Load a cached figure-data dict (pickle, written by the evaluation stage)."""
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------ brain slices
def reshape_to_cube(data: np.ndarray, target_size: int, plot_diff: bool = False) -> np.ndarray:
    """Centre-crop / zero-pad a 3D volume to a `target_size`^3 cube (port of plotting.py).

    Intensity volumes are padded with their minimum and min-max normalised; difference
    volumes (`plot_diff=True`) are padded with zeros and left unnormalised.
    """
    source_shape, target_shape = data.shape, [target_size] * 3
    out_img = np.zeros(target_shape, dtype=data.dtype)
    if not plot_diff:
        out_img = out_img + np.min(data)
    pad_width = [
        (max(0, (t - s) // 2), max(0, (t - s + 1) // 2)) for t, s in zip(target_shape, source_shape)
    ]
    crop = tuple(
        slice(max(0, (s - t) // 2), s - max(0, (s - t + 1) // 2))
        for s, t in zip(source_shape, target_shape)
    )
    out_slices = tuple(slice(p[0], -p[1] if p[1] else None) for p in pad_width)
    out_img[out_slices] = data[crop]
    if not plot_diff:
        out_img = (out_img - np.min(out_img)) / (np.max(out_img) - np.min(out_img))
    return out_img.astype(data.dtype)


def get_slices(
    volume: np.ndarray, sx: int | None = None, sy: int | None = None, sz: int | None = None
) -> dict[str, np.ndarray]:
    """Axial / sagittal / coronal slices of a 3D volume (port of plotting.py::get_slices)."""
    sx = sx or volume.shape[0] // 2
    sy = sy or volume.shape[1] // 2
    sz = sz or volume.shape[2] // 2
    return {
        "axial": volume[:, :, sz].T,
        "sagittal": volume[sx, :, :].T,
        "coronal": volume[:, sy, :].T,
    }


def process_mri(
    vol: np.ndarray, target_size: int, sx: int, sy: int, sz: int, plot_diff: bool = False
) -> dict[str, np.ndarray]:
    """Cube a volume and extract three orthogonal slices (indices in un-cubed space)."""
    cube = reshape_to_cube(vol, target_size, plot_diff=plot_diff)
    scale = [target_size / s for s in vol.shape[-3:]]
    return get_slices(cube, int(sx * scale[0]), int(sy * scale[1]), int(sz * scale[2]))
