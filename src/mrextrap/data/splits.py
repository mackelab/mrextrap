"""Build the subject-level train/val/test split CSVs from user-supplied ADNI tables.

No subject identifiers are shipped with this package: the ADNI metadata table (``ADNIMERGE.csv``,
downloaded from LONI IDA under the ADNI data use agreement) and an index of your preprocessed
T1 volumes are inputs, and the split CSVs are written to your own run directory.

Inputs
------
``adnimerge_csv`` (ADNIMERGE.csv, one row per visit) needs the columns

* ``PTID``    subject identifier, e.g. ``<SITE>_S_<RID>`` (becomes ``subject_id``)
* ``VISCODE`` visit code, ``bl`` or ``m06``, ``m12``, ... (converted to ``session_id`` ``M00``, ``M06``, ...)
* ``AGE``     age in years (ADNIMERGE stores the baseline age; a per-visit age also works, see below)
* ``DX``      diagnosis with values ``CN``, ``MCI``, ``Dementia`` (may be empty)

``image_index_csv`` (one row per preprocessed scan, written by you) needs

* ``filename``    path of the preprocessed NIfTI. If ``subject_id`` / ``session_id`` are absent
  they are parsed from BIDS-style names, ``sub-ADNI<SITE>S<RID>`` and ``ses-M06``.
* optional ``subject_id`` (``<SITE>_S_<RID>`` format) and ``session_id`` (``M00``, ``M06``, ...)

Output (``out_dir/{train,val,test}.csv``) columns: ``subject_id, session_id, filename, AGE, DX,
DX_filled, age_from_baseline, baseline_age, AGE_filled``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

ADNIMERGE_COLUMNS = ("PTID", "VISCODE", "AGE", "DX")
OUTPUT_COLUMNS = (
    "subject_id",
    "session_id",
    "filename",
    "AGE",
    "DX",
    "DX_filled",
    "age_from_baseline",
    "baseline_age",
    "AGE_filled",
)
SPLIT_NAMES = ("train", "val", "test")

# NOTE(paper): fractions of the original split (subjects shuffled with seed 42):
# first 90 % train, next 3 % validation, last 7 % test.
TRAIN_FRACTION = 0.90
VAL_END_FRACTION = 0.93


def diagnosis_label(dx_series) -> str:
    """Subject-level diagnosis: any Dementia -> "Dementia"; else any MCI -> "MCI"; else "CN"."""
    values = set(pd.Series(dx_series).dropna().values)
    if "Dementia" in values:
        return "Dementia"
    if "MCI" in values:
        return "MCI"
    return "CN"


def viscode_to_session_id(viscode: str) -> str:
    """ADNIMERGE visit code -> session id: ``bl`` -> ``M00``, ``m06`` -> ``M06``."""
    viscode = str(viscode)
    if viscode == "bl":
        return "M00"
    return viscode.upper()


def parse_subject_session(filename: str) -> tuple[str, str]:
    """Extract ``("<SITE>_S_<RID>", "M06")`` from a BIDS-style path."""
    subject = re.findall(r"sub-ADNI\d{3}S\d{4}", filename)
    if not subject:
        raise ValueError(f"Cannot parse subject id from filename: {filename}")
    subject_id = subject[0].replace("sub-ADNI", "").replace("S", "_S_")
    sessions = sorted(re.findall(r"ses-M\d{2,3}", filename), key=len, reverse=True)
    if not sessions:
        raise ValueError(f"Cannot parse session id from filename: {filename}")
    return subject_id, sessions[0].replace("ses-M", "M")


def read_subject_list(path: str | Path) -> list[str]:
    """Read subject ids from a text file (one per line, blank lines and ``#`` comments ignored)."""
    lines = Path(path).read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def _require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _attach_metadata(adnimerge: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Left-join ADNIMERGE ``AGE``/``DX`` onto the image index by (subject, session)."""
    _require_columns(adnimerge, ADNIMERGE_COLUMNS, "ADNIMERGE table")
    _require_columns(index, ["filename"], "image index")
    index = index.copy()
    if "subject_id" not in index.columns or "session_id" not in index.columns:
        parsed = index["filename"].apply(parse_subject_session)
        index["subject_id"] = [p[0] for p in parsed]
        index["session_id"] = [p[1] for p in parsed]

    meta = adnimerge[["PTID", "VISCODE", "AGE", "DX"]].copy()
    meta["subject_id"] = meta["PTID"].astype(str)
    meta["session_id"] = meta["VISCODE"].apply(viscode_to_session_id)
    meta = meta.drop_duplicates(subset=["subject_id", "session_id"], keep="first")
    merged = index[["subject_id", "session_id", "filename"]].merge(
        meta[["subject_id", "session_id", "AGE", "DX"]],
        on=["subject_id", "session_id"],
        how="left",
    )
    assert len(merged) == len(index)
    return merged


def _finalize_split(df: pd.DataFrame, dx_fill: str, drop_missing_files: bool) -> pd.DataFrame:
    """Derive ``DX_filled``, ``age_from_baseline``, ``baseline_age`` and ``AGE_filled``."""
    df = df.copy()
    if drop_missing_files:
        df = df[df["filename"].apply(lambda p: Path(p).exists())].copy()

    if dx_fill == "global_ffill":
        # NOTE(paper): missing DX is forward-filled over the whole table in row order, i.e. a
        # subject's leading NaNs inherit the previous subject's last diagnosis.
        df["DX_filled"] = df["DX"].ffill()
    elif dx_fill == "per_subject_ffill":
        df["DX_filled"] = df.groupby("subject_id", sort=False)["DX"].ffill()
    else:
        raise ValueError(f"dx_fill must be 'global_ffill' or 'per_subject_ffill', got {dx_fill}")

    # NOTE(paper): visit time comes from the session label (months / 12), i.e. relative to the ADNI
    # baseline visit, not to the subject's first *available* scan.
    df["age_from_baseline"] = df["session_id"].apply(lambda x: int(x[1:]) / 12.0)
    # NOTE(paper): baseline age = per-subject minimum of AGE; NaN or 0 -> mean AGE of the split.
    df["baseline_age"] = df.groupby("subject_id")["AGE"].transform("min").fillna(df["AGE"].mean())
    df["baseline_age"] = df["baseline_age"].replace(0, df["AGE"].mean())
    df["AGE_filled"] = df["baseline_age"] + df["age_from_baseline"]

    # NOTE(paper): the released "*_nona" tables had no NaN in the columns used downstream.
    df = df.dropna(subset=["DX_filled", "AGE_filled", "age_from_baseline"])
    return df[list(OUTPUT_COLUMNS)].reset_index(drop=True)


def assign_subjects(
    subjects: Sequence[str],
    seed: int = 42,
    test_subjects: Sequence[str] | None = None,
    val_subjects: Sequence[str] | None = None,
) -> dict[str, list[str]]:
    """Partition subject ids (given in table order) into train/val/test.

    Without explicit lists: shuffle with ``np.random.RandomState(seed)`` (same stream as
    ``np.random.seed(seed); np.random.shuffle``) and cut at ``int(0.90 n)`` and ``int(0.93 n)``.
    With explicit lists, those subjects form val/test (missing list = empty) and all remaining
    subjects are train.
    """
    subjects = list(dict.fromkeys(subjects))
    if test_subjects is not None or val_subjects is not None:
        test = list(dict.fromkeys(test_subjects or []))
        val = list(dict.fromkeys(val_subjects or []))
        known = set(subjects)
        unknown = (set(test) | set(val)) - known
        if unknown:
            raise ValueError(f"{len(unknown)} explicit subject(s) not present in the image index")
        if set(test) & set(val):
            raise ValueError("test_subjects and val_subjects overlap")
        held_out = set(test) | set(val)
        train = [s for s in subjects if s not in held_out]
        return {"train": train, "val": val, "test": test}

    arr = np.array(subjects, dtype=object)
    np.random.RandomState(seed).shuffle(arr)
    n = len(arr)
    n_train = int(TRAIN_FRACTION * n)
    n_val_end = int(VAL_END_FRACTION * n)
    return {
        "train": list(arr[:n_train]),
        "val": list(arr[n_train:n_val_end]),
        "test": list(arr[n_val_end:]),
    }


def build_split_csvs(
    adnimerge_csv,
    image_index_csv,
    test_subjects: Sequence[str] | None = None,
    val_subjects: Sequence[str] | None = None,
    out_dir="splits",
    seed: int = 42,
    dx_fill: str = "global_ffill",
    drop_missing_files: bool = False,
) -> dict[str, Path]:
    """Write ``out_dir/{train,val,test}.csv`` (subject-level split, paper procedure).

    Returns the paths of the written files. Guarantees that no subject occurs in two splits.
    """
    adnimerge = pd.read_csv(adnimerge_csv, low_memory=False)
    index = pd.read_csv(image_index_csv)
    table = _attach_metadata(adnimerge, index)

    # NOTE(paper): order of first appearance in the table, then shuffled (seed 42).
    subjects = table["subject_id"].unique().tolist()
    groups = assign_subjects(subjects, seed, test_subjects, val_subjects)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    seen: set[str] = set()
    for name in SPLIT_NAMES:
        members = set(groups[name])
        assert not (members & seen), "subject appears in more than one split"
        seen |= members
        part = table[table["subject_id"].isin(members)]
        part = _finalize_split(part, dx_fill, drop_missing_files)
        paths[name] = out_dir / f"{name}.csv"
        part.to_csv(paths[name], index=False)
    return paths
