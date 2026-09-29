"""Helpers shared by the cluster tools: hashing, report writing and a subject-ID leak guard.

The tools only ever *write reports containing counts and aggregate statistics*. Reports leave the
cluster, so ``write_report`` refuses to write anything that looks like an ADNI subject identifier.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path

SUBJECT_ID_PATTERN = re.compile(r"\b\d{3}_S_\d{4}\b")


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Hex sha256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_no_subject_ids(text: str) -> None:
    """Raise if ``text`` contains something shaped like an ADNI subject id (``<SITE>_S_<RID>``)."""
    hit = SUBJECT_ID_PATTERN.search(text)
    if hit:
        raise ValueError("refusing to write a report containing a subject identifier")


def _jsonable(obj):
    """Best-effort conversion of numpy / torch scalars and paths for ``json.dumps``."""
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and getattr(obj, "ndim", 1) == 0:
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def write_report(out_dir: str | Path, stem: str, report: dict, markdown: str) -> tuple[Path, Path]:
    """Write ``<stem>.json`` and ``<stem>.md`` into ``out_dir`` (created if needed)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, default=_jsonable, sort_keys=False)
    assert_no_subject_ids(text)
    assert_no_subject_ids(markdown)
    json_path, md_path = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    json_path.write_text(text + "\n")
    md_path.write_text(markdown if markdown.endswith("\n") else markdown + "\n")
    return json_path, md_path


def markdown_table(header: Sequence[str], rows: Sequence[Sequence]) -> str:
    """Plain GitHub-flavoured markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c).replace("|", "/") for c in row) + " |")
    return "\n".join(lines)
