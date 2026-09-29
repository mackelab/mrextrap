"""Segmentation backends and region-volume computation.

``run_synthseg`` / ``SynthSegSegmenter`` port ``eval.py::get_synthseg_seg`` of the paper code;
``read_vols_csv`` / ``region_volumes`` port ``get_real_seg_volumes`` / ``get_reg_seg_volumes``.
``FakeSegmenter`` is a weight-free stand-in for tests (never used for paper numbers).
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import nibabel as nib
import numpy as np
import pandas as pd

REGIONS = ("hippocampus", "ventricle", "cortex", "white matter")
"""Regions of the paper. NOTE(paper) (Q9): aggregation is *substring matching on the SynthSeg CSV
column names*: "hippocampus" = left + right; "ventricle" = lateral + inferior lateral + 3rd + 4th;
"cortex" = cerebral + cerebellar cortex; "white matter" = cerebral + cerebellar white matter."""


# --------------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------------
@runtime_checkable
class SegmentationBackend(Protocol):
    """Anything that turns a list of 3D volumes into a SynthSeg-style ``vols.csv``."""

    def segment(self, volumes: np.ndarray | Sequence[np.ndarray], out_dir: str | Path) -> Path:
        """Write inputs/segmentations into ``out_dir`` and return the path of ``vols.csv``."""
        ...


def _input_names(volumes) -> tuple[list[np.ndarray], list[str]]:
    # NOTE(paper): a single array is stored as ``tmp_final.nii.gz``, a list as ``tmpNNN_final.nii.gz``.
    if isinstance(volumes, np.ndarray):
        return [volumes], ["tmp_final"]
    return list(volumes), [f"tmp{i:03d}_final" for i in range(len(volumes))]


def write_input_volumes(volumes, out_dir, check_already_exists: bool = True) -> list[Path]:
    """Save volumes as NIfTI with an identity affine (``np.eye(4)``, paper behaviour).

    Existing files are kept when ``check_already_exists`` (cache semantics of the legacy code).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays, names = _input_names(volumes)
    paths = []
    for x, name in zip(arrays, names, strict=True):
        p = out_dir / f"{name}.nii.gz"
        if not (p.exists() and check_already_exists):
            nib.save(nib.Nifti1Image(x, np.eye(4)), str(p))
        paths.append(p)
    return paths


def run_synthseg(
    volumes,
    out_dir,
    device: str | None = None,
    weights=None,
    run_dir=None,
    predictor=None,
    check_already_exists: bool = True,
) -> Path:
    """Segment volumes with the torch SynthSeg (robust mode) and return ``out_dir/vols.csv``.

    Writes ``tmpNNN_final.nii.gz`` inputs and ``tmpNNN_final_synthseg_robust.nii.gz`` outputs.
    NOTE(paper): every ``*final.nii.gz`` found below ``out_dir`` is segmented (sorted by path),
    including files from earlier calls, exactly as in the legacy ``get_synthseg_seg``.
    """
    from mrextrap.segmentation.synthseg_predict import SynthSegPredictor

    out_dir = Path(out_dir)
    write_input_volumes(volumes, out_dir, check_already_exists)
    if predictor is None:
        predictor = SynthSegPredictor(weights=weights, run_dir=run_dir, device=device)
    vols_csv = out_dir / "vols.csv"
    predictor.segment_folder(out_dir, out_dir, vols_csv)
    return vols_csv


class SynthSegSegmenter:
    """``SegmentationBackend`` around the torch SynthSeg; the network is loaded once and reused."""

    def __init__(self, weights=None, run_dir=None, device: str | None = None):
        from mrextrap.segmentation.synthseg_predict import SynthSegPredictor

        self.predictor = SynthSegPredictor(weights=weights, run_dir=run_dir, device=device)

    def segment(self, volumes, out_dir) -> Path:
        return run_synthseg(volumes, out_dir, predictor=self.predictor)


def make_segmenter(cfg, device: str | None = None) -> SegmentationBackend:
    """Segmenter of a stage config: ``segmentation.backend`` is ``synthseg`` (default) or ``fake``.

    ``fake`` (``FakeSegmenter``) needs no weights and exists for the synthetic smoke run and tests;
    it is never used for paper numbers. The SynthSeg weights are ``paths.synthseg_weights`` if that
    file exists, else ``MREXTRAP_SYNTHSEG_WEIGHTS`` or the run-directory default.
    """
    seg_cfg = cfg.get("segmentation", None) or {}
    if seg_cfg.get("backend", "synthseg") == "fake":
        return FakeSegmenter()
    weights = Path(str(cfg.paths.get("synthseg_weights", "")))
    return SynthSegSegmenter(
        weights=weights if weights.is_file() else None, run_dir=cfg.paths.run_dir, device=device
    )


def load_segmentations(out_dir, n: int | None = None) -> np.ndarray | list[np.ndarray]:
    """Load ``*_synthseg_robust`` outputs written by ``run_synthseg`` (list of ``n`` arrays,
    or the single ``tmp_final`` segmentation when ``n`` is None)."""
    out_dir = Path(out_dir)
    if n is None:
        return nib.load(str(out_dir / "tmp_final_synthseg_robust.nii.gz")).get_fdata()
    return [
        nib.load(str(out_dir / f"tmp{i:03d}_final_synthseg_robust.nii.gz")).get_fdata()
        for i in range(n)
    ]


class FakeSegmenter:
    """Weight-free segmenter for tests: bands normalised intensity into the real label ids.

    Voxels below ``bg_threshold`` (after min-max scaling) are background, the rest is split into
    equal intensity bands mapped to the 31 SynthSeg foreground labels (sorted ids). Output files
    and ``vols.csv`` follow the layout of ``run_synthseg`` (volumes = voxel counts, 1 mm^3).
    """

    def __init__(self, bg_threshold: float = 0.1):
        self.bg_threshold = bg_threshold

    def label_map(self, x: np.ndarray) -> np.ndarray:
        from mrextrap.segmentation.synthseg_predict import load_label_tables

        ids = load_label_tables()["unique_labels"]
        x = np.asarray(x, dtype=np.float64)
        span = x.max() - x.min()
        v = (x - x.min()) / span if span > 0 else np.zeros_like(x)
        band = np.minimum((v * (len(ids) - 1)).astype(int), len(ids) - 2) + 1
        return np.where(v < self.bg_threshold, 0, ids[band]).astype(np.int32)

    def segment(self, volumes, out_dir) -> Path:
        from mrextrap.segmentation.synthseg_predict import load_label_tables

        tables = load_label_tables()
        ids, names = tables["unique_labels"], tables["names"][tables["unique_indices"]]
        out_dir = Path(out_dir)
        write_input_volumes(volumes, out_dir, check_already_exists=False)
        arrays, stems = _input_names(volumes)
        vols_csv = out_dir / "vols.csv"
        with open(vols_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([""] + names[1:].tolist())
            w.writerow([""] + [str(i) for i in ids[1:]])
            for x, stem in zip(arrays, stems, strict=True):
                seg = self.label_map(x)
                nib.save(
                    nib.Nifti1Image(seg, np.eye(4)), str(out_dir / f"{stem}_synthseg_robust.nii.gz")
                )
                w.writerow([stem] + [str(float((seg == i).sum())) for i in ids[1:]])
        return vols_csv


# --------------------------------------------------------------------------------------------
# Region volumes
# --------------------------------------------------------------------------------------------
def read_vols_csv(path: str | Path) -> pd.DataFrame:
    """Read a SynthSeg ``vols.csv`` (two header rows: names, label ids).

    NOTE(paper): legacy ``pd.read_csv(path)[1:]`` drops the label-id row (the index therefore
    starts at 1) and the unnamed subject column is dropped.
    """
    segs = pd.read_csv(path)[1:]
    return segs.drop(columns=["Unnamed: 0"], errors="ignore")


def region_volumes(
    df: pd.DataFrame, regions: Sequence[str] = REGIONS, tbv: float | None = None
) -> pd.DataFrame:
    """Add ``TBV``, one column per region and ``<region>_pct_tbv`` to a ``read_vols_csv`` frame.

    NOTE(paper) (Q9): ``TBV`` = sum of all label columns of each row (background excluded);
    region = sum of all columns whose name contains the region string; ``%TBV`` is relative to
    ``tbv`` if given (predicted volumes: TBV of the first *real* scan) else to the TBV of the
    first row of ``df`` (real volumes). Columns are scanned as they are added, as in the legacy code.
    """
    segs = df.copy()
    segs["TBV"] = segs.sum(axis=1)
    denom = segs["TBV"].iloc[0] if tbv is None else tbv
    for region in regions:
        segs[region] = 0
        segs[region] = sum([segs[col] for col in segs.columns if region in col])
        segs[f"{region}_pct_tbv"] = segs[region] / denom * 100
    return segs


def real_region_volumes(path: str | Path, regions: Sequence[str] = REGIONS) -> pd.DataFrame:
    """``get_real_seg_volumes``: %TBV relative to the first row's TBV."""
    return region_volumes(read_vols_csv(path), regions)


def predicted_region_volumes(
    path: str | Path, tbv: float, regions: Sequence[str] = REGIONS
) -> pd.DataFrame:
    """``get_reg_seg_volumes``: %TBV relative to the given (real, first scan) TBV."""
    return region_volumes(read_vols_csv(path), regions, tbv=tbv)


__all__ = [
    "REGIONS",
    "FakeSegmenter",
    "SegmentationBackend",
    "SynthSegSegmenter",
    "load_segmentations",
    "make_segmenter",
    "predicted_region_volumes",
    "read_vols_csv",
    "real_region_volumes",
    "region_volumes",
    "run_synthseg",
    "write_input_volumes",
]
