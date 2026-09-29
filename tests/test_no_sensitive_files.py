"""Guard: the repository must never contain subject-level data, images, weights or notebooks.

The scan covers every file git would commit (tracked plus untracked-but-not-ignored). It fails on:

* CSV/TSV files (outside ``tests/fixtures``) whose header has a subject-identifying column
  (RID, PTID, Subject, subject_id, ...);
* ADNI subject-ID strings (``123_S_4567``, ``sub-ADNI123S4567``) in any text file;
* NIfTI/DICOM/MGZ images, model weights (*.pt, *.pth, *.safetensors, *.bin, *.ckpt), wandb
  directories, ``.ipynb`` notebooks with outputs (any notebook is also reported by name);
* hard-coded cluster paths (``/mnt/``) and institutional e-mail addresses in text files;
* files larger than 5 MB.

Allow-list: add a repo-relative path (or glob, matched with ``fnmatch``) to the relevant set
below, with a comment explaining why it is safe. Only *synthetic* or *non-personal* content
may be allow-listed. Currently allowed: the SynthSeg label ``.npy`` files (static label tables),
and everything under ``tests/fixtures/`` for the CSV column and ID-pattern checks (these must
stay tiny and synthetic). This file itself is exempt from the content scan since it spells out
the patterns.
"""

from __future__ import annotations

import csv
import fnmatch
import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve().relative_to(REPO).as_posix()

MAX_BYTES = 5 * 1024 * 1024

# ---------------------------------------------------------------- allow-lists (repo-relative globs)
# Files allowed to exceed MAX_BYTES.
ALLOW_LARGE: set[str] = set()
# Binary/weight-like extensions that are legitimate (static SynthSeg label tables, not weights).
ALLOW_BINARY: set[str] = {"src/mrextrap/segmentation/labels/*.npy"}
# Files allowed to contain subject-like CSV columns or ADNI-ID strings: synthetic fixtures only.
ALLOW_SUBJECT_CONTENT: set[str] = {"tests/fixtures/*"}
# Files allowed to mention /mnt/ or an institutional address (e.g. licence/copyright headers).
ALLOW_MISC_STRINGS: set[str] = set()

# ---------------------------------------------------------------- patterns
ADNI_ID_PATTERNS = [
    re.compile(r"\b\d{3}_S_\d{4}\b"),
    re.compile(r"sub-ADNI\d{3}S\d{4}", re.IGNORECASE),
]
MISC_PATTERNS = {
    "cluster path /mnt/": re.compile(r"(?<![A-Za-z0-9_.-])/mnt/"),
    "institutional e-mail": re.compile(r"@[\w.-]*uni-tuebingen\.de", re.IGNORECASE),
}
SUBJECT_COLUMNS = {
    "rid",
    "ptid",
    "subject",
    "subject_id",
    "subjectid",
    "subj",
    "subj_id",
    "participant_id",
    "patient_id",
    "viscode",
    "image_id",
    "imageid",
    "loni_image",
}
IMAGE_SUFFIXES = (".nii", ".nii.gz", ".mgz", ".mgh", ".dcm", ".dicom", ".ima", ".img", ".hdr")
ARRAY_SUFFIXES = (".npy", ".npz", ".pkl", ".pickle", ".parquet")
WEIGHT_SUFFIXES = (".pt", ".pth", ".safetensors", ".bin", ".ckpt", ".onnx", ".h5")


def _candidate_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=REPO,
        capture_output=True,
        check=True,
    ).stdout.decode()
    return sorted(f for f in out.split("\0") if f and (REPO / f).is_file())


FILES = _candidate_files()


def _allowed(path: str, allow: set[str]) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in allow)


def _read_text(path: str) -> str | None:
    data = (REPO / path).read_bytes()
    if b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", errors="ignore")


def test_scan_sees_files():
    assert FILES, "git ls-files returned nothing; guard would be vacuous"


def test_no_large_files():
    bad = [
        f for f in FILES if (REPO / f).stat().st_size > MAX_BYTES and not _allowed(f, ALLOW_LARGE)
    ]
    assert not bad, f"files larger than 5 MB: {bad}"


def test_no_images_or_weights():
    bad = []
    for f in FILES:
        low = f.lower()
        if _allowed(f, ALLOW_BINARY):
            continue
        if low.endswith(IMAGE_SUFFIXES + WEIGHT_SUFFIXES + ARRAY_SUFFIXES):
            bad.append(f)
    assert not bad, f"image / weight / array files: {bad}"


def test_no_wandb_dirs():
    bad = [f for f in FILES if {"wandb", ".wandb"} & set(Path(f).parts) or f.endswith(".wandb")]
    assert not bad, f"wandb artefacts: {bad}"


def test_no_notebooks():
    bad = []
    for f in FILES:
        if not f.endswith(".ipynb"):
            continue
        try:
            nb = json.loads((REPO / f).read_text())
            has_out = any(c.get("outputs") for c in nb.get("cells", []))
        except (ValueError, OSError):
            has_out = True
        bad.append(f"{f} (with outputs)" if has_out else f)
    assert not bad, f"notebooks are not shipped: {bad}"


def test_no_subject_tables():
    bad = []
    for f in FILES:
        if not f.lower().endswith((".csv", ".tsv")) or _allowed(f, ALLOW_SUBJECT_CONTENT):
            continue
        sep = "\t" if f.lower().endswith(".tsv") else ","
        try:
            with open(REPO / f, newline="", encoding="utf-8", errors="ignore") as fh:
                header = next(csv.reader(fh, delimiter=sep), [])
        except OSError:
            header = []
        cols = {c.strip().strip('"').lower() for c in header}
        hit = sorted(cols & SUBJECT_COLUMNS)
        if hit:
            bad.append(f"{f}: {hit}")
    assert not bad, f"CSV/TSV files with subject-level columns: {bad}"


def test_no_adni_subject_ids():
    bad = []
    for f in FILES:
        if f == THIS_FILE or _allowed(f, ALLOW_SUBJECT_CONTENT):
            continue
        text = _read_text(f)
        if text is None:
            continue
        for pat in ADNI_ID_PATTERNS:
            if pat.search(text):
                bad.append(f)
                break
    assert not bad, f"ADNI subject-ID patterns in: {bad}"


def test_no_cluster_paths_or_institutional_emails():
    bad = []
    for f in FILES:
        if f == THIS_FILE or _allowed(f, ALLOW_MISC_STRINGS):
            continue
        text = _read_text(f)
        if text is None:
            continue
        for name, pat in MISC_PATTERNS.items():
            if pat.search(text):
                bad.append(f"{f}: {name}")
    assert not bad, f"private paths / addresses: {bad}"


@pytest.mark.parametrize(
    "text, should_match",
    [
        ("002_S_0295", True),
        ("sub-ADNI002S0295", True),
        ("12_S_345", False),
        ("x0002_S_02951", False),
    ],
)
def test_patterns_self_check(text, should_match):
    assert any(p.search(text) for p in ADNI_ID_PATTERNS) == should_match
