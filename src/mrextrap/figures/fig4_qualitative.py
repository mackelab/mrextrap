"""Fig. 4: qualitative extrapolation for one subject (volume trajectories + MRI slices).

Ported from the thesis re-render (`plot_figures.py::fig_qual`). The subject is chosen when the
cached data is prepared (config value); nothing subject-specific lives here.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from .common import (
    C_POST,
    C_REAL,
    REGION_LABEL,
    REGIONS,
    WIDTH,
    figure_style,
    process_mri,
    save_figure,
)

# Paper layout (volumes of the preprocessed 1 mm grid, ~ 160 x 192 x 160 voxels).
PAPER_LAYOUT = {
    "slice_idx": (50, 85, 90),
    "precrop": ((0, 1), (5, 10), (10, 10)),
    # inset crops (row slice, col slice) in the cubed image space (177 px)
    "ax_crop": ((45, 140), (55, 122)),
    "sag_crop": ((75, 120), (40, 110)),
    "diff_max": 2.0,
    "xticks": (74, 77, 80, 83),
}
RED = "#c00000"


def _slices(vol, layout, diff=False):
    pc = layout["precrop"]
    v = vol[
        pc[0][0] : -pc[0][1] or None, pc[1][0] : -pc[1][1] or None, pc[2][0] : -pc[2][1] or None
    ]
    si = [s - p[0] for s, p in zip(layout["slice_idx"], pc)]
    return process_mri(v, max(v.shape), *si, plot_diff=diff)


def _draw_mosaic(fig, x0, y0, u, sl, layout, boxes=False):
    """axial (2u x 2u) + sagittal and coronal (u x u) stacked to the right; bottom-left in inches."""
    W, H = fig.get_size_inches()
    ax_a = fig.add_axes([x0 / W, y0 / H, 2 * u / W, 2 * u / H])
    ax_s = fig.add_axes([(x0 + 2 * u) / W, (y0 + u) / H, u / W, u / H])
    ax_c = fig.add_axes([(x0 + 2 * u) / W, y0 / H, u / W, u / H])
    for ax, view in ((ax_a, "axial"), (ax_s, "sagittal"), (ax_c, "coronal")):
        ax.imshow(sl[view], cmap="gray", origin="lower", vmin=0, vmax=1)
        ax.axis("off")
    if boxes:
        for ax, (rs, cs) in ((ax_a, layout["ax_crop"]), (ax_s, layout["sag_crop"])):
            ax.add_patch(
                Rectangle((cs[0], rs[0]), cs[1] - cs[0], rs[1] - rs[0], fill=False, ec=RED, lw=0.7)
            )


def _draw_insets(fig, x0, y0, u, dsl, layout):
    W, H = fig.get_size_inches()
    ims = []
    for k, (view, (rs, cs)) in enumerate(
        (("axial", layout["ax_crop"]), ("sagittal", layout["sag_crop"]))
    ):
        ax = fig.add_axes(
            [(x0 + 0.06) / W, (y0 + (1 - k) * u + 0.04) / H, (u - 0.08) / W, (u - 0.08) / H]
        )
        d = layout["diff_max"]
        ims.append(
            ax.imshow(
                dsl[view][slice(*rs), slice(*cs)],
                cmap="bwr",
                origin="lower",
                vmin=-d,
                vmax=d,
                aspect="equal",
            )
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(RED)
            s.set_linewidth(0.7)
    return ims[0]


def make_fig4(data: dict, out_path, layout: dict | None = None):
    """Render Fig. 4 to `out_path` (PDF) and return the Figure.

    `data` keys:
      traj: subject trajectory dict (as in Fig. 3 panel c) with `post[3]`.
      ages: {"baseline": age, "<idx>": target age, ...} (one entry per shown target index).
      vols: {"baseline": 3D array, "real_<idx>": 3D array, "pred_<idx>": 3D array}
            (dict or the object returned by np.load of an .npz).
    `layout` overrides entries of PAPER_LAYOUT (slice indices/crops depend on volume size).
    """
    layout = {**PAPER_LAYOUT, **(layout or {})}
    t, ages, vols = data["traj"], data["ages"], data["vols"]
    files = list(vols.files) if hasattr(vols, "files") else list(vols)
    targets = sorted((k.split("_")[1] for k in files if k.startswith("real_")), key=int)

    with figure_style():
        H = 4.3
        fig = plt.figure(figsize=(WIDTH, H))

        # (a) volume trajectories, 1 x 4, top band
        gs = fig.add_gridspec(1, 4, left=0.085, right=0.99, top=0.955, bottom=3.0 / H, wspace=0.42)
        tgt_ages = [ages[k] for k in targets]
        b = np.where(np.asarray(t["all_idx"]) == t["baseline_index"])[0][0]
        for j, region in enumerate(REGIONS):
            ax = fig.add_subplot(gs[0, j])
            ax.plot(t["all_ages"], t["real"][region], "o-", color=C_REAL, ms=2.2)
            ax.scatter(
                t["baseline_age"],
                t["real"][region][b],
                marker="+",
                s=45,
                color=C_REAL,
                zorder=5,
                lw=1.0,
            )
            ax.plot(t["pred_ages"], t["post"][3][region], "o-", color=C_POST, ms=2.2)
            for ta in tgt_ages:
                for xs, ys in (
                    (t["all_ages"], t["real"][region]),
                    (t["pred_ages"], t["post"][3][region]),
                ):
                    i = np.argmin(np.abs(np.asarray(xs) - ta))
                    ax.plot(xs[i], ys[i], "o", mfc="none", mec=RED, ms=4.5, mew=0.9, zorder=6)
            ax.set_title(REGION_LABEL[region])
            if layout["xticks"] is not None:
                ax.set_xticks(list(layout["xticks"]))
            else:
                ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
            ax.set_xlabel("age (years)")
            ax.yaxis.set_major_locator(MaxNLocator(3))
            if j == 0:
                ax.set_ylabel("volume (% TBV)")
        fig.legend(
            [
                Line2D([], [], color=C_REAL, marker="o", ms=2.2),
                Line2D([], [], color=C_REAL, marker="+", ls="", ms=6),
                Line2D([], [], color=C_POST, marker="o", ms=2.2),
                Line2D([], [], ls="", marker="o", mfc="none", mec=RED, ms=4.5, mew=0.9),
            ],
            ["real", "latest observed scan", "posterior (n=3)", "shown in b"],
            loc="center",
            bbox_to_anchor=(0.5, 2.44 / H),
            ncol=4,
            frameon=False,
        )
        fig.text(0.0, 1.0, "a", fontsize=9, fontweight="bold", ha="left", va="top")

        # (b) MRI grid
        u, gap, x_base = 0.42, 0.07, 0.17  # inches per mosaic unit, gap, room for row labels
        x_t = [x_base + 3 * u + 2 * gap + i * (4 * u + gap) for i in range(len(targets))]
        y_pred = 0.08
        y_real = y_pred + 2 * u + 0.2
        fig.text(0.0, 2.3 / H, "b", fontsize=9, fontweight="bold", ha="left", va="top")

        base = np.asarray(vols["baseline"])
        _draw_mosaic(fig, x_base, y_real, u, _slices(base, layout), layout)
        fig.text(
            (x_base + 1.5 * u) / WIDTH,
            (y_real + 2 * u + 0.04) / H,
            f"latest observed scan ({int(ages['baseline'])} y)",
            ha="center",
            va="bottom",
        )
        im = None
        for k, xt in zip(targets, x_t):
            real, pred = np.asarray(vols[f"real_{k}"]), np.asarray(vols[f"pred_{k}"])
            _draw_mosaic(fig, xt, y_real, u, _slices(real, layout), layout, boxes=True)
            _draw_mosaic(fig, xt, y_pred, u, _slices(pred, layout), layout, boxes=True)
            im = _draw_insets(
                fig, xt + 3 * u, y_real, u, _slices(real - base, layout, True), layout
            )
            _draw_insets(fig, xt + 3 * u, y_pred, u, _slices(pred - base, layout, True), layout)
            fig.text(
                (xt + 2 * u) / WIDTH,
                (y_real + 2 * u + 0.04) / H,
                f"target age {int(ages[k])} y",
                ha="center",
                va="bottom",
            )
        fig.text(0.02 / WIDTH, (y_real + u) / H, "real", rotation=90, ha="left", va="center")
        fig.text(0.02 / WIDTH, (y_pred + u) / H, "predicted", rotation=90, ha="left", va="center")

        # colour bar in the empty slot below the baseline scan
        cax = fig.add_axes(
            [(x_base + 0.25 * u) / WIDTH, (y_pred + 0.9 * u) / H, 2.5 * u / WIDTH, 0.07 / H]
        )
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        d = layout["diff_max"]
        cb.set_ticks([-d, 0, d], labels=["−", "0", "+"])
        cb.outline.set_linewidth(0.5)
        cax.set_title("difference from\nlatest observed scan", pad=3)
        save_figure(fig, out_path)
    return fig
