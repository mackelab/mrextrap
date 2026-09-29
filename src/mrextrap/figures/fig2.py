"""Fig. 2: autoencoder latent space and longitudinal interpolation.

Panels: (a) latent channels of one scan, (b) decoded interpolation example (optional),
(c) regional volume change along the interpolation (linearity, R^2), (d) PCA of latents.
Panel provenance is weak in the legacy notebooks, so these are re-implementations of the
notebook cells (train_latent_regression cell 7; interpolation cells 13, 19-24).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

from .common import REGION_LABEL, REGIONS, WIDTH, figure_style, panel_letter, save_figure

MAX_AGE_GAP = 4.1  # years; subjects with a longer follow-up are excluded from panel c
MIN_R2 = 0.7  # NOTE(paper): the notebook drops curves with R^2 < 0.7 before plotting


def interpolation_r2(alphas, values) -> float:
    """Squared Pearson correlation of a volume-change curve against alpha."""
    return float(linregress(alphas, values).rvalue ** 2)


def summarize_linearity(
    interp: dict, max_age_gap: float = MAX_AGE_GAP, min_r2: float = MIN_R2
) -> dict:
    """Per-region R^2 statistics over the retained subjects (age gap <= max, R^2 >= min)."""
    out = {}
    for region in REGIONS:
        curves = interp["curves"][region]
        r2s = []
        for curve, gap in zip(curves, interp["age_gaps"]):
            if gap > max_age_gap or len(curve) < 3:
                continue
            r2 = interpolation_r2(np.linspace(0, 1, len(curve)), curve)
            if r2 >= min_r2:
                r2s.append(r2)
        out[region] = {
            "n": len(r2s),
            "mean_r2": float(np.mean(r2s)) if r2s else float("nan"),
            "std_r2": float(np.std(r2s)) if r2s else float("nan"),
        }
    return out


def latent_pca(zs: list, clamp: float = 5.0) -> np.ndarray:
    """2-D PCA of latents relative to each subject's first scan (notebook cells 20-21).

    `zs`: list of arrays (N_i, C, D, H, W); returns (sum(N_i - 1), 2).
    """
    from sklearn.decomposition import PCA

    rel = [np.asarray(z)[1:] - np.asarray(z)[:1] for z in zs]
    flat = np.concatenate(rel, axis=0).reshape(sum(len(r) for r in rel), -1)
    flat = np.clip(flat, -clamp, clamp)
    return PCA(n_components=2).fit_transform(flat)


def make_fig2(data: dict, out_path):
    """Render Fig. 2 to `out_path` (PDF) and return the Figure.

    `data` keys:
      latent: array (C, D, H, W), latent of one scan (panel a).
      interp: {"mode": "latent"|"voxel", "curves": {region: [array(n_alpha)]} (% TBV change from
              the alpha=0 volume, one per subject), "age_gaps": [years per subject]}.
              The paper figure must use mode "latent" with the paper autoencoder (older8);
              the mode is written into panel c and the PDF metadata (Q6).
      interp_slices: optional array (n, H, W): decoded slices along alpha (panel b).
      zs: list of arrays (N_i, C, D, H, W): per-subject latent sequences for panel d.
    A JSON sidecar `<out>.summary.json` records the mode and R^2 statistics.
    """
    interp = data["interp"]
    mode = interp.get("mode", "unknown")
    latent = np.asarray(data["latent"])
    with figure_style():
        fig = plt.figure(figsize=(WIDTH, 4.6), layout="constrained")
        top, bottom = fig.subfigures(2, 1, height_ratios=[1.6, 3.0])
        sfa, sfb = top.subfigures(1, 2, width_ratios=[1.0, 1.4])
        sfc, sfd = bottom.subfigures(1, 2, width_ratios=[1.5, 1.0])

        # (a) latent channels (one axial-like slice per channel)
        axs = sfa.subplots(1, latent.shape[0], squeeze=False)[0]
        lo, hi = float(np.min(latent)), float(np.max(latent))
        for c, ax in enumerate(axs):
            ax.imshow(latent[c, :, :, latent.shape[3] // 2].T, cmap="gray", vmin=lo, vmax=hi)
            ax.axis("off")
            ax.set_title(f"ch. {c + 1}")
        panel_letter(sfa, "a")

        # (b) decoded interpolation
        sl = data.get("interp_slices")
        if sl is not None:
            sl = np.asarray(sl)
            axs = sfb.subplots(1, len(sl), squeeze=False)[0]
            alphas = np.linspace(0, 1, len(sl))
            for ax, s, a in zip(axs, sl, alphas):
                ax.imshow(s, cmap="gray", origin="lower")
                ax.axis("off")
                ax.set_title(f"{a:.2f}")
            sfb.supxlabel(r"interpolation $\alpha$")
        panel_letter(sfb, "b", x=0.0)

        # (c) linearity of regional volume changes along the interpolation
        axs = sfc.subplots(2, 2, sharex=True).flatten()
        summary = summarize_linearity(interp)
        for ax, region in zip(axs, REGIONS):
            for curve, gap in zip(interp["curves"][region], interp["age_gaps"]):
                curve = np.asarray(curve)
                if gap > MAX_AGE_GAP or len(curve) < 3:
                    continue
                al = np.linspace(0, 1, len(curve))
                if interpolation_r2(al, curve) < MIN_R2:
                    continue
                ax.plot(al, curve, color="black", alpha=0.1)
                ax.scatter([0, 1], [curve[0], curve[-1]], color="k", alpha=0.5, s=4, lw=0)
            ax.set_title(f"{REGION_LABEL[region]}, R² = {summary[region]['mean_r2']:.2f}")
            ax.set_xticks([0, 1])
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        sfc.supxlabel(rf"interpolation $\alpha$ ({mode} space)")
        sfc.supylabel("% volume change from baseline")
        panel_letter(sfc, "c")

        # (d) PCA of latent trajectories, colour = subject, opacity = time
        ax = sfd.subplots()
        if data.get("zs"):
            emb = latent_pca(data["zs"])
            lens = [len(z) - 1 for z in data["zs"]]
            cmap = plt.get_cmap("tab10")
            start = 0
            for i, n in enumerate(lens):
                e = emb[start : start + n]
                start += n
                col = cmap(i % 10)
                ax.plot(e[:, 0], e[:, 1], color=col, alpha=0.4, lw=0.8)
                ax.scatter(
                    e[:, 0], e[:, 1], color=col, s=6, lw=0, alpha=np.linspace(0.2, 1, max(n, 1))[:n]
                )
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        panel_letter(sfd, "d")

        out = save_figure(fig, out_path, metadata={"Subject": f"interpolation mode: {mode}"})
    sidecar = Path(out).with_suffix(".summary.json")
    sidecar.write_text(json.dumps({"interpolation_mode": mode, "linearity": summary}, indent=2))
    return fig
