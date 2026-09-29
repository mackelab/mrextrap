"""Statistics and Table 1 (single-volume MAE per region and method).

Ports the table/test code of ``eval_unet_additional.py`` (L1705-1890, L2035-2125): per-subject
absolute errors, ``SE = std(ddof=0) / sqrt(n)``, paired t-test (``scipy.stats.ttest_rel``) and the
paired z-test (normal approximation of the mean difference).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REGIONS = ("hippocampus", "ventricle", "cortex", "white matter")
REGION_LABELS = {
    "hippocampus": "Hippocampus",
    "ventricle": "Ventricles",
    "cortex": "Grey matter",
    "white matter": "White matter",
}
METHOD_LABELS = {
    "global": "Global",
    "unet": "UNet",
    "diffusion": "Diffusion",
    "daninet": "DANINet",
}

# Values are (mean absolute error, std) in % TBV, transcribed from Table 2 ("Proposed") of Ravi et al. 2022,
# Med. Image Anal. 75:102257. Hippocampus = left + right (0.029 + 0.031; SD sqrt(0.028^2 + 0.031^2)).
# Grey and white matter both read 0.829 (SD 0.612) in that table. The standard error divides the std by
# sqrt(178), the number of test subjects.
DANINET_REFERENCE: dict[str, tuple[float, float]] = {
    "hippocampus": (0.060, 0.042),
    "ventricle": (0.257, 0.222),
    "cortex": (0.829, 0.612),
    "white matter": (0.829, 0.612),
}
DANINET_N = 178


# ---------------------------------------------------------------------------------------------
# Loading and alignment
# ---------------------------------------------------------------------------------------------
def load_results(path: str | Path) -> dict[str, list[tuple]]:
    """Load a ``results.json`` ``{region: [[mae, dx, sid], ...]}`` (legacy files: ``[mae, dx]``)."""
    with open(path) as f:
        raw = json.load(f)
    return {region: [tuple(r) for r in rows] for region, rows in raw.items()}


def maes(rows: Sequence) -> np.ndarray:
    """Per-subject MAEs from a list of ``(mae, dx, sid)`` tuples or plain numbers."""
    return np.array([r[0] if isinstance(r, (list, tuple)) else r for r in rows], dtype=float)


def subject_ids(rows: Sequence) -> list[str] | None:
    """Subject ids of a results list, or ``None`` for legacy rows without an id."""
    if all(isinstance(r, (list, tuple)) and len(r) >= 3 for r in rows):
        return [str(r[2]) for r in rows]
    return None


def align_pair(rows_a: Sequence, rows_b: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """Per-subject MAE arrays of two methods in the same subject order.

    Pairs by subject id (order of ``rows_a``) when both carry ids; otherwise positionally with a
    length check (legacy files without ids).
    """
    ids_a, ids_b = subject_ids(rows_a), subject_ids(rows_b)
    a, b = maes(rows_a), maes(rows_b)
    if ids_a is not None and ids_b is not None:
        if set(ids_a) != set(ids_b):
            raise ValueError("Results of the two methods cover different subjects.")
        pos = {s: i for i, s in enumerate(ids_b)}
        return a, b[[pos[s] for s in ids_a]]
    if len(a) != len(b):
        raise ValueError(f"Cannot pair results of different length ({len(a)} vs {len(b)}).")
    return a, b


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------
def standard_error(x) -> float:
    """``np.std(x, ddof=0) / sqrt(n)`` (paper convention for the Table 1 SE)."""
    x = np.asarray(x, dtype=float)
    return float(np.std(x) / math.sqrt(len(x)))


def paired_tests(a, b) -> dict[str, float]:
    """Paired t-test and paired z-test of ``a - b`` (two-sided).

    ``t, p_t`` = ``scipy.stats.ttest_rel(a, b)``; ``z = mean(d) / (std(d, ddof=1) / sqrt(n))`` and
    ``p_z = 2 * (1 - Phi(|z|))`` with ``d = a - b`` (normal approximation, as in the legacy code).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    t, p_t = stats.ttest_rel(a, b, alternative="two-sided")
    d = a - b
    z = float(np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d))))
    p_z = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {"t": float(t), "p_t": float(p_t), "z": z, "p_z": p_z}


def z_test_vs_reported(mean: float, se: float, ref_mean: float, ref_se: float) -> dict[str, float]:
    """Two-sided z-test of our ``mean +- se`` against a reported ``ref_mean +- ref_se``.

    ``z = (mean - ref_mean) / sqrt(se^2 + ref_se^2)``. NOTE(paper): the original DANINet-vs-Global
    z-test code was not found in the snapshot; this is the standard unpaired form (DANINet's
    per-subject errors are not available for pairing).
    """
    z = (mean - ref_mean) / math.sqrt(se**2 + ref_se**2)
    return {"z": float(z), "p_z": float(2 * stats.norm.sf(abs(z)))}


# ---------------------------------------------------------------------------------------------
# Table 1
# ---------------------------------------------------------------------------------------------
def table1(
    results: Mapping[str, Mapping[str, Sequence]],
    daninet: Mapping[str, tuple[float, float]] | None = DANINET_REFERENCE,
    regions: Sequence[str] = REGIONS,
    daninet_n: int = DANINET_N,
) -> pd.DataFrame:
    """Table 1 numbers: one row per region, columns ``<method>_mae`` and ``<method>_se``.

    Args:
        results: ``{method: {region: per-subject MAEs or (mae, dx, sid) rows}}``.
        daninet: ``{region: (mae, std)}`` reference numbers (``None`` omits the DANINet columns);
            its SE is ``std / sqrt(daninet_n)`` as in the legacy code.
    """
    rows = []
    for region in regions:
        row: dict[str, float | str] = {"region": region}
        for method, per_region in results.items():
            x = maes(per_region[region])
            row[f"{method}_mae"] = float(np.mean(x))
            row[f"{method}_se"] = standard_error(x)
        if daninet is not None:
            mean, std = daninet[region]
            row["daninet_mae"] = float(mean)
            row["daninet_se"] = float(std / math.sqrt(daninet_n))
        rows.append(row)
    return pd.DataFrame(rows).set_index("region")


def _methods_in(df: pd.DataFrame) -> list[str]:
    return [c[: -len("_mae")] for c in df.columns if c.endswith("_mae")]


def format_table1(df: pd.DataFrame, decimals: int = 3) -> str:
    """Plain-text Table 1: ``mae +- se`` per cell (pipe separated)."""
    methods = _methods_in(df)
    header = ["Region"] + [METHOD_LABELS.get(m, m) for m in methods]
    lines = [" | ".join(header)]
    for region, row in df.iterrows():
        cells = [REGION_LABELS.get(region, region)]
        cells += [
            f"{row[f'{m}_mae']:.{decimals}f} ± {row[f'{m}_se']:.{decimals}f}" for m in methods
        ]
        lines.append(" | ".join(cells))
    return "\n".join(lines) + "\n"


def table1_latex(df: pd.DataFrame, decimals: int = 3) -> str:
    """LaTeX ``tabular`` of Table 1 (``$\\pm$`` between MAE and SE)."""
    methods = _methods_in(df)
    lines = [
        "\\begin{tabular}{l" + "c" * len(methods) + "}",
        "\\toprule",
        " & ".join(["Region"] + [METHOD_LABELS.get(m, m) for m in methods]) + " \\\\",
        "\\midrule",
    ]
    for region, row in df.iterrows():
        cells = [REGION_LABELS.get(region, region)]
        cells += [
            f"${row[f'{m}_mae']:.{decimals}f} \\pm {row[f'{m}_se']:.{decimals}f}$" for m in methods
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def pairwise_tests(
    results: Mapping[str, Mapping[str, Sequence]],
    pairs: Sequence[tuple[str, str]] = (
        ("unet", "diffusion"),
        ("global", "unet"),
        ("global", "diffusion"),
    ),
    regions: Sequence[str] = REGIONS,
) -> pd.DataFrame:
    """Paired t/z tests for every method pair and region (rows aligned by subject id)."""
    rows = []
    for a, b in pairs:
        for region in regions:
            x, y = align_pair(results[a][region], results[b][region])
            rows.append({"a": a, "b": b, "region": region, "n": len(x), **paired_tests(x, y)})
    return pd.DataFrame(rows)


def daninet_tests(
    table: pd.DataFrame, methods: Sequence[str] | None = None, regions: Sequence[str] = REGIONS
) -> pd.DataFrame:
    """Unpaired z-tests of every method's MAE against the DANINet reference in ``table``."""
    methods = [m for m in (methods or _methods_in(table)) if m != "daninet"]
    rows = []
    for m in methods:
        for region in regions:
            r = table.loc[region]
            rows.append(
                {
                    "method": m,
                    "region": region,
                    **z_test_vs_reported(
                        r[f"{m}_mae"], r[f"{m}_se"], r["daninet_mae"], r["daninet_se"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_table1(cfg) -> Path:
    """Config entry point: read ``<results_dir>/<method>/results.json``, write Table 1 files.

    Writes ``tables/table1.{csv,txt,tex}`` and ``tables/table1_tests.csv`` under ``paths.run_dir``.
    """
    ev = cfg.get("eval", {})
    results_dir = Path(
        str(ev.get("results_dir", Path(str(cfg.paths.run_dir)) / "eval/single_volume"))
    )
    methods = list(ev.get("methods", ["global", "unet", "diffusion"]))
    results = {m: load_results(results_dir / m / "results.json") for m in methods}
    df = table1(results)
    out = Path(str(cfg.paths.run_dir)) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "table1.csv")
    (out / "table1.txt").write_text(format_table1(df))
    (out / "table1.tex").write_text(table1_latex(df))
    pairs = [(a, b) for i, a in enumerate(methods) for b in methods[i + 1 :]]
    pairwise_tests(results, pairs).to_csv(out / "table1_tests.csv", index=False)
    return out
