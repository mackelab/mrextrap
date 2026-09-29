# ruff: noqa: C408
"""Figure functions render from synthetic cached inputs; Table 1 renderer golden test."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from mrextrap.figures import (
    make_all_figures,
    make_fig2,
    make_fig3,
    make_fig4,
    make_fig5,
    render_table1,
)
from mrextrap.figures.common import REGIONS, load_cache

RNG = np.random.default_rng(0)


def _traj(n_all=6, base=2, post=True):
    ages = 70 + np.arange(n_all, dtype=float)
    pred = ages[base + 1 :]
    reg = lambda n: {r: np.cumsum(RNG.random(n)) for r in REGIONS}
    return dict(
        all_ages=ages,
        all_idx=np.arange(n_all),
        baseline_age=ages[base],
        baseline_index=base,
        pred_ages=pred,
        pred_indices=list(range(base + 1, n_all)),
        real=reg(n_all),
        prior=reg(len(pred)),
        post={3: reg(len(pred))},
    )


def _fig3_data():
    panel_a = {}
    for fmt in ("latest", "earliest"):
        for term in ("near_term", "far_term"):
            panel_a[(fmt, term)] = {
                r: dict(x=np.arange(4), mean=RNG.random(4), sem=RNG.random(4) * 0.1)
                for r in REGIONS
            }
    dxs = np.array(["CN", "MCI", "Dementia"] * 5)
    panel_b = {r: dict(x=RNG.random(15) * 0.1, y=RNG.random(15), dx=list(dxs)) for r in REGIONS}
    return dict(panel_a=panel_a, panel_b=panel_b, panel_c={"A": _traj(), "B": _traj()})


def test_fig3(tmp_path):
    out = tmp_path / "fig3.pdf"
    fig = make_fig3(_fig3_data(), out)
    assert out.stat().st_size > 1000 and fig is not None


def _fig4_data():
    shape = (40, 44, 40)
    vols = {"baseline": RNG.random(shape).astype(np.float32)}
    for k in ("5", "8"):
        vols[f"real_{k}"] = RNG.random(shape).astype(np.float32)
        vols[f"pred_{k}"] = RNG.random(shape).astype(np.float32)
    layout = dict(
        slice_idx=(20, 22, 20),
        precrop=((0, 1), (2, 2), (2, 2)),
        ax_crop=((5, 30), (5, 30)),
        sag_crop=((5, 30), (5, 30)),
        xticks=None,
    )
    return dict(traj=_traj(), ages={"baseline": 72.0, "5": 75.0, "8": 78.0}, vols=vols), layout


def test_fig4(tmp_path):
    data, layout = _fig4_data()
    out = tmp_path / "fig4.pdf"
    make_fig4(data, out, layout=layout)
    assert out.stat().st_size > 1000


def test_fig2(tmp_path):
    curves = {
        r: [np.linspace(0, s, 10) + RNG.normal(0, 1e-3, 10) for s in (1, 2, 3, -1)] for r in REGIONS
    }
    data = dict(
        latent=RNG.normal(size=(4, 8, 10, 8)),
        interp=dict(mode="latent", curves=curves, age_gaps=[1.0, 2.0, 3.0, 5.0]),
        interp_slices=RNG.random((4, 12, 12)),
        zs=[RNG.normal(size=(n, 4, 3, 3, 3)) for n in (3, 4, 5)],
    )
    out = tmp_path / "fig2.pdf"
    make_fig2(data, out)
    side = json.loads((tmp_path / "fig2.summary.json").read_text())
    assert side["interpolation_mode"] == "latent"
    assert side["linearity"]["ventricle"]["n"] == 3  # age gap 5.0 y excluded
    assert side["linearity"]["ventricle"]["mean_r2"] > 0.99
    assert out.stat().st_size > 1000


def test_fig5(tmp_path):
    n = 12
    lens = [4] * n
    data = dict(
        betas=[RNG.normal(size=(k, 4, 3, 3, 3)) * (i + 1) for i, k in enumerate(lens)],
        ages_abs=[70 + 5 * RNG.random((1, k)) + 10 * (i % 3) for i, k in enumerate(lens)],
        diseases=["CN", "MCI", "Dementia"] * 4,
    )
    out = tmp_path / "fig5.pdf"
    make_fig5(data, out)
    assert out.stat().st_size > 1000


def test_make_all_figures_reads_cache(tmp_path):
    import pickle

    from omegaconf import OmegaConf

    cache = tmp_path / "figures" / "cache"
    cache.mkdir(parents=True)
    with open(cache / "fig3.pkl", "wb") as f:
        pickle.dump(_fig3_data(), f)
    written = make_all_figures(OmegaConf.create({"paths": {"run_dir": str(tmp_path)}}))
    assert [p.name for p in written] == ["fig3.pdf"] and written[0].exists()
    assert set(load_cache(cache / "fig3.pkl")) == {"panel_a", "panel_b", "panel_c"}


@pytest.fixture
def table_df():
    return pd.DataFrame(
        {
            "hippocampus_mean": [0.05, 0.04],
            "hippocampus_se": [0.004, 0.003],
            "ventricle_mean": [1.2, 1.5],
            "ventricle_se": [0.1, 0.12],
        },
        index=["Global prior", "UNet prior"],
    )


def test_table1_golden_latex(table_df):
    expected = (
        "\\begin{tabular}{lcc}\n\\toprule\n"
        "Method & hippocampus & ventricle \\\\\n\\midrule\n"
        "Global prior & 0.050 $\\pm$ 0.004 & \\textbf{1.200 $\\pm$ 0.100} \\\\\n"
        "UNet prior & \\textbf{0.040 $\\pm$ 0.003} & 1.500 $\\pm$ 0.120 \\\\\n"
        "\\bottomrule\n\\end{tabular}\n"
    )
    assert render_table1(table_df, "latex") == expected


def test_table1_csv_markdown_multiindex(table_df):
    assert (
        render_table1(table_df, "csv").splitlines()[1] == "Global prior,0.050 ± 0.004,1.200 ± 0.100"
    )
    mi = table_df.copy()
    mi.columns = pd.MultiIndex.from_tuples([tuple(c.rsplit("_", 1)) for c in mi.columns])
    md = render_table1(mi, "markdown", digits=2)
    assert "| UNet prior | **0.04 ± 0.00** | 1.50 ± 0.12 |" in md
