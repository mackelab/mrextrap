"""Fig. 5: interpretation of the latent progression rates beta.

Re-implements notebook `plot_latent_embeddings.ipynb` cells 16-18 (a), 33-40 (b), 41 (c).
Inputs are the *train* latent-beta dict (legacy schema) plus per-subject diagnoses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .common import WIDTH, figure_style, panel_letter, save_figure

DX_ORDER = ("CN", "MCI", "Dementia")
DX_LABEL = {"CN": "Healthy", "MCI": "MCI", "Dementia": "Dementia"}
DX_MARKER = {"CN": "o", "MCI": "s", "Dementia": "+"}
DX_COLOR = {"CN": "darkgreen", "MCI": "darkorange", "Dementia": "maroon"}
AGE_BINS = np.arange(57.5, 94.5, 5)  # NOTE(paper) Q15


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def beta_pca(betas: list) -> tuple[np.ndarray, np.ndarray]:
    """Panel a embedding: PCA of per-channel whitened mid-sequence betas.

    NOTE(paper) Q15: only the first half of the subjects is plotted, the beta at the sequence
    midpoint is used, and each channel is whitened separately.
    Returns (embedding for all subjects, indices of the plotted subjects).
    """
    from sklearn.decomposition import PCA

    mid = np.concatenate([_np(b)[len(b) // 2 : len(b) // 2 + 1] for b in betas]).astype(np.float64)
    for c in range(mid.shape[1]):
        ch = mid[:, c] - mid[:, c].mean()
        mid[:, c] = ch / ch.std()
    emb = PCA(n_components=2).fit(mid.reshape(len(mid), -1)).transform(mid.reshape(len(mid), -1))
    return emb, np.arange(len(betas) // 2)


def beta_l1_norms(betas: list) -> np.ndarray:
    """Panel b/c statistic: |beta| summed over channels and space at the first scan
    (`betas[s][0]`), one value per subject (NOTE(paper) Q15)."""
    return np.array([float(np.abs(_np(b)[0]).sum()) for b in betas])


def age_bin_table(
    ages: np.ndarray, norms: np.ndarray, dx: list
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean and SEM of the beta norm per age bin and diagnosis (panel c)."""
    df = pd.DataFrame({"Age": ages, "Beta": norms, "Dx": dx})
    df["AgeBin"] = pd.cut(df["Age"], bins=AGE_BINS, right=False)
    g = df.groupby(["AgeBin", "Dx"], observed=False)["Beta"]
    return g.mean().unstack(), g.sem().unstack()


def make_fig5(data: dict, out_path):
    """Render Fig. 5 to `out_path` (PDF) and return the Figure.

    `data` keys: betas (list of (N_i, C, D, H, W) arrays/tensors), ages_abs (list of (1, N_i)),
    diseases (list of "CN"|"MCI"|"Dementia", one per subject).

    NOTE(paper) Q15: panel c bins by the *mean* age over the series (the paper text says age at
    the first scan); bins are 5 y starting at 57.5; mean +- SEM.
    """
    betas, dx = data["betas"], list(data["diseases"])
    ages_mid = np.array([_np(a)[0][len(_np(a)[0]) // 2] for a in data["ages_abs"]])
    ages_mean = np.array([_np(a)[0].mean() for a in data["ages_abs"]])
    norms = beta_l1_norms(betas)

    with figure_style():
        fig = plt.figure(figsize=(WIDTH, 2.0), layout="constrained")
        sfa, sfb, sfc = fig.subfigures(1, 3, width_ratios=[1.15, 0.8, 1.15])

        # (a) PCA of betas, colour = age, marker = diagnosis
        ax = sfa.subplots()
        emb, shown = beta_pca(betas)
        lo, hi = ages_mid.min(), ages_mid.max()
        cmap = plt.cm.coolwarm
        for i in shown:
            ax.scatter(
                emb[i, 0],
                emb[i, 1],
                color=cmap((ages_mid[i] - lo) / max(hi - lo, 1e-9)),
                alpha=0.5,
                s=8,
                marker=DX_MARKER[dx[i]],
                **({} if DX_MARKER[dx[i]] == "+" else {"edgecolor": "none"}),
            )
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(lo, hi), cmap=cmap)
        sfa.colorbar(sm, ax=ax, orientation="vertical", label="age (years)", shrink=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        handles = [
            ax.scatter(
                [], [], marker=DX_MARKER[d], label=DX_LABEL[d], facecolor="gray", color="gray", s=6
            )
            for d in DX_ORDER
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=6)
        panel_letter(sfa, "a")

        # (b) log10 of beta norm per diagnosis
        ax = sfb.subplots()
        dxa = np.array(dx)
        groups = [np.log10(norms[dxa == d]) for d in DX_ORDER]
        ax.boxplot(
            groups,
            widths=0.5,
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 3,
            },
            medianprops={"color": "black"},
        )
        ax.set_xticks(range(1, len(DX_ORDER) + 1), [DX_LABEL[d] for d in DX_ORDER])
        ax.set_ylabel(r"$\ell_1$ norm of $\beta$ (log$_{10}$)")
        ax.tick_params(axis="x", labelrotation=45)
        panel_letter(sfb, "b")

        # (c) beta norm across 5-year age bins
        ax = sfc.subplots()
        mean, sem = age_bin_table(ages_mean, norms, dx)
        mid = (AGE_BINS[:-1] + AGE_BINS[1:]) / 2
        for d in DX_ORDER:
            if d in mean.columns:
                ax.errorbar(
                    mid,
                    mean[d].values,
                    yerr=sem[d].values,
                    marker="o",
                    ms=2.5,
                    capsize=2,
                    color=DX_COLOR[d],
                    label=DX_LABEL[d],
                )
        ax.set_xlabel("age (years)")
        ax.set_ylabel(r"$\ell_1$ norm of $\beta$")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.legend()
        panel_letter(sfc, "c")
        save_figure(fig, out_path)
    return fig
