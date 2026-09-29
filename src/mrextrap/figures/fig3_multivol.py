"""Fig. 3: multi-volume conditioning with the Global prior.

Ported from the thesis re-render (`plot_figures.py::fig_multivol`). Plot-only: consumes the
cached dict produced by the multi-volume evaluation (see `make_fig3` for the schema).
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from .common import (
    C_POST,
    C_PRIOR,
    C_REAL,
    DX_COLORS,
    REGION_LABEL,
    REGIONS,
    WIDTH,
    figure_style,
    panel_letter,
    save_figure,
)

# (baseline format, term) -> colour of panel a
PANEL_A_COLORS = {
    ("latest", "near_term"): "#4287f5",
    ("latest", "far_term"): "#1a3a75",
    ("earliest", "near_term"): "#42c46f",
    ("earliest", "far_term"): "#1a6334",
}
DEFAULT_YLIM = {"hippocampus": 0.05, "ventricle": 1.0, "cortex": 3.0, "white matter": 1.4}


def make_fig3(data: dict, out_path):
    """Render Fig. 3 to `out_path` (PDF) and return the Figure.

    `data` keys (all plain numpy / python, no torch):
      panel_a: {(fmt, term): {region: {"x", "mean", "sem"}}} with fmt in {latest, earliest},
               term in {near_term, far_term}; x = number of conditioning volumes (0 = prior).
      panel_b: {region: {"x": |delta progression rate|, "y": posterior MAE, "dx": [str]}}.
      panel_c: {label: trajectory} for two example subjects; trajectory as returned by
               `subject_trajectories` (all_ages, all_idx, baseline_age, baseline_index,
               pred_ages, real/prior: {region: array}, post: {3: {region: array}}).

    NOTE(paper) Q8: panel b comes from the *earliest*-baseline results and reproduces the
    paper's r values; panels a and c use the *latest* baseline.
    """
    with figure_style():
        fig = plt.figure(figsize=(WIDTH, 5.75), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
        top, bot_all = fig.subfigures(2, 1, height_ratios=[3.35, 2.4])
        top_plots, top_leg = top.subfigures(2, 1, height_ratios=[2.85, 0.5])
        sfa, sfb = top_plots.subfigures(1, 2, wspace=0.04)
        lega, legb = top_leg.subfigures(1, 2, wspace=0.04)
        bot, legc = bot_all.subfigures(2, 1, height_ratios=[2.25, 0.15])

        _panel_a(sfa, lega, data["panel_a"])
        _panel_b(sfb, legb, data["panel_b"])
        _panel_c(bot, legc, data["panel_c"])
        save_figure(fig, out_path)
    return fig


def _panel_a(sf, leg_sf, panel_a):
    axs = sf.subplots(2, 2, sharex=True).flatten()
    handles, labels = [], []
    for (fmt, term), col in PANEL_A_COLORS.items():
        if (fmt, term) not in panel_a:
            continue
        d = panel_a[(fmt, term)]
        h = None
        for i, region in enumerate(REGIONS):
            h = axs[i].errorbar(
                d[region]["x"],
                d[region]["mean"],
                yerr=d[region]["sem"],
                fmt="o-",
                color=col,
                capsize=1.5,
                elinewidth=0.8,
                ms=2.5,
            )
        handles.append(h)
        labels.append(f"{term.replace('_', ' ')}, {fmt} as baseline")
    for i, region in enumerate(REGIONS):
        ax = axs[i]
        ax.set_title(REGION_LABEL[region])
        ax.set_ylim(0, max(ax.get_ylim()[1], DEFAULT_YLIM[region]))
        ax.set_xticks([0, 1, 2, 3], ["0\n(prior)", "1", "2", "3"])
        ax.yaxis.set_major_locator(MaxNLocator(3))
    sf.supylabel("MAE (% TBV)")
    sf.supxlabel("additional conditioning volumes")
    leg_sf.legend(
        handles,
        labels,
        loc="upper center",
        ncol=1,
        frameon=False,
        handletextpad=0.5,
        labelspacing=0.2,
        borderaxespad=0,
    )
    panel_letter(sf, "a")


def _panel_b(sf, leg_sf, panel_b):
    axs = sf.subplots(2, 2).flatten()
    for i, region in enumerate(REGIONS):
        ax = axs[i]
        d = panel_b[region]
        x, y = np.asarray(d["x"]), np.asarray(d["y"])
        ax.scatter(x, y, c=[DX_COLORS[dx] for dx in d["dx"]], s=7, alpha=0.7, lw=0)
        if len(x) >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0:  # degenerate data (tiny synthetic runs)
            slope, icpt = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, slope * xs + icpt, "--", color="black", alpha=0.7, lw=0.8)
            ax.set_title(f"{REGION_LABEL[region]}, r = {np.corrcoef(x, y)[0, 1]:.2f}")
        else:
            ax.set_title(f"{REGION_LABEL[region]}, r = n/a")
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f" if y.max() < 0.5 else "%.1f"))
        if x.max() < 0.05:
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    sf.supylabel("MAE (posterior, n=3)")
    sf.supxlabel("|Δ regional progression rate|\nbetween prediction and conditioning")
    leg_sf.legend(
        [Line2D([], [], ls="", marker="o", color=c, ms=3.5) for c in DX_COLORS.values()],
        list(DX_COLORS),
        loc="upper center",
        ncol=3,
        frameon=False,
        handletextpad=0.2,
        borderaxespad=0,
    )
    panel_letter(sf, "b")


def _panel_c(sf, leg_sf, panel_c):
    axs = sf.subplots(len(panel_c), 4, squeeze=False)
    for row, (label, t) in enumerate(panel_c.items()):
        a0 = t["all_ages"][0]
        b = np.where(np.asarray(t["all_idx"]) == t["baseline_index"])[0][0]
        for j, region in enumerate(REGIONS):
            ax = axs[row, j]
            ax.plot(t["all_ages"] - a0, t["real"][region], "o-", color=C_REAL, ms=2.2)
            ax.scatter(
                t["baseline_age"] - a0,
                t["real"][region][b],
                marker="+",
                s=45,
                color=C_REAL,
                zorder=5,
                lw=1.0,
            )
            ax.plot(t["pred_ages"] - a0, t["prior"][region], "o-", color=C_PRIOR, ms=2.2, alpha=0.9)
            ax.plot(t["pred_ages"] - a0, t["post"][3][region], "o-", color=C_POST, ms=2.2)
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
            if row == 0:
                ax.set_title(REGION_LABEL[region])
            if j == 0:
                ax.set_ylabel(f"Sub. {label}\nvolume (% TBV)", linespacing=1.5)
    sf.supxlabel("age from baseline (years)")
    leg_sf.legend(
        [
            Line2D([], [], color=C_REAL, marker="o", ms=2.2),
            Line2D([], [], color=C_REAL, marker="+", ls="", ms=6),
            Line2D([], [], color=C_PRIOR, marker="o", ms=2.2),
            Line2D([], [], color=C_POST, marker="o", ms=2.2),
        ],
        ["real", "baseline", "prior", "posterior (n=3)"],
        loc="center",
        ncol=4,
        frameon=False,
        borderaxespad=0,
    )
    panel_letter(sf, "c")
