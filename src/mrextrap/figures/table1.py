"""Table 1 rendering (LaTeX / CSV / Markdown) from a table of per-method regional MAE.

Input contract (produced by `evaluation.stats.table1`): a DataFrame indexed by method name with
columns `<region>_mean` and `<region>_se` (or a two-level column index (region, "mean"|"se")).
Values are % TBV. The lowest mean per region is set in bold in the LaTeX/Markdown output.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import REGION_LABEL, REGIONS


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [f"{r}_{s}" for r, s in df.columns]
    return df


def _cells(df: pd.DataFrame, digits: int, bold_best: bool, bold: str) -> pd.DataFrame:
    df = _normalize(df)
    regions = [r for r in REGIONS if f"{r}_mean" in df.columns]
    out = pd.DataFrame(index=df.index, columns=[REGION_LABEL[r] for r in regions], dtype=object)
    for r in regions:
        best = df[f"{r}_mean"].idxmin() if bold_best else None
        for method in df.index:
            txt = (
                f"{df.loc[method, f'{r}_mean']:.{digits}f} ± {df.loc[method, f'{r}_se']:.{digits}f}"
            )
            if method == best:
                txt = bold.format(txt)
            out.loc[method, REGION_LABEL[r]] = txt
    return out


def render_table1(
    df: pd.DataFrame, fmt: str = "latex", digits: int = 3, bold_best: bool = True
) -> str:
    """Return Table 1 as text. `fmt` in {"latex", "csv", "markdown"}."""
    if fmt == "csv":
        return _cells(df, digits, False, "{}").to_csv(index_label="method")
    if fmt == "markdown":
        c = _cells(df, digits, bold_best, "**{}**")
        head = "| method | " + " | ".join(c.columns) + " |"
        rows = [f"| {m} | " + " | ".join(row) + " |" for m, row in zip(c.index, c.values)]
        return "\n".join([head, "|" + "---|" * (len(c.columns) + 1), *rows]) + "\n"
    if fmt == "latex":
        c = _cells(df, digits, bold_best, r"\textbf{{{}}}").map(lambda s: s.replace("±", r"$\pm$"))
        lines = [
            r"\begin{tabular}{l" + "c" * len(c.columns) + "}",
            r"\toprule",
            "Method & " + " & ".join(c.columns) + r" \\",
            r"\midrule",
        ]
        lines += [f"{m} & " + " & ".join(row) + r" \\" for m, row in zip(c.index, c.values)]
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines) + "\n"
    raise ValueError(f"unknown format {fmt!r}")


def write_table1(df: pd.DataFrame, out_dir: str | Path, digits: int = 3) -> list[Path]:
    """Write table1.tex, table1.csv and table1.md into `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt, ext in (("latex", "tex"), ("csv", "csv"), ("markdown", "md")):
        p = out_dir / f"table1.{ext}"
        p.write_text(render_table1(df, fmt, digits))
        paths.append(p)
    return paths
