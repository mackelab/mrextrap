"""Config-driven stage functions that do not belong to a scientific module (``mrextrap <stage>``).

``run_make_splits`` wraps the split builder and ``run_real_volumes`` segments the real scans of a
split into ``volumes/<split>_real/<sid>/vols.csv`` (the reference volumes of every evaluation).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def run_make_splits(cfg: DictConfig) -> dict[str, Path]:
    """``mrextrap make-splits``: ADNIMERGE + image index -> ``paths.splits_dir/{train,val,test}.csv``.

    Reads ``cfg.splits``: ``adnimerge_csv`` and ``image_index_csv`` (required), ``test_subjects`` /
    ``val_subjects`` (lists or paths of text files with one id per line), ``seed`` (42),
    ``dx_fill`` and ``drop_missing_files``.
    """
    from mrextrap.data.splits import build_split_csvs, read_subject_list

    s = cfg.get("splits", None)
    if s is None or not s.get("adnimerge_csv") or not s.get("image_index_csv"):
        raise ValueError(
            "make-splits needs splits.adnimerge_csv and splits.image_index_csv "
            "(e.g. `mrextrap make-splits splits.adnimerge_csv=... splits.image_index_csv=...`)"
        )

    def subjects(key):
        value = s.get(key, None)
        if value is None:
            return None
        if isinstance(value, str):
            return read_subject_list(value)
        return list(value)

    return build_split_csvs(
        s.adnimerge_csv,
        s.image_index_csv,
        test_subjects=subjects("test_subjects"),
        val_subjects=subjects("val_subjects"),
        out_dir=cfg.paths.splits_dir,
        seed=int(s.get("seed", 42)),
        dx_fill=s.get("dx_fill", "global_ffill"),
        drop_missing_files=bool(s.get("drop_missing_files", False)),
    )


def run_real_volumes(cfg: DictConfig) -> dict[str, Path]:
    """``mrextrap real-volumes``: segment the real scans of ``real_volumes.splits`` (default test).

    Writes ``<volumes_dir>/<split>_real/<sid>/vols.csv`` with one row per scan in age order.
    Reads the split CSVs of ``paths.splits_dir`` (crop settings from ``cfg.dataset`` if present).
    """
    from einops import rearrange

    from mrextrap.data.datasets import get_longitudinal_dataloaders
    from mrextrap.segmentation.volumes import make_segmenter

    rv = cfg.get("real_volumes", None) or {}
    splits = list(rv.get("splits", None) or ["test"])
    device = rv.get("device", None) or cfg.get("eval", {}).get("device", "cpu")
    splits_dir = Path(str(cfg.paths.splits_dir))
    base = OmegaConf.create(
        {
            "dataset": {f"{s}_csv": str(splits_dir / f"{s}.csv") for s in ("train", "val", "test")},
            "training": {"batch_size": 1},
        }
    )
    dataset_cfg = OmegaConf.merge(
        base,
        {"dataset": {k: v for k, v in cfg.get("dataset", {}).items() if not k.endswith("_csv")}},
    )
    loaders = dict(
        zip(
            ("train", "val", "test"),
            get_longitudinal_dataloaders(dataset_cfg, num_workers=0),
            strict=True,
        )
    )
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    segmenter = make_segmenter(cfg, device)
    written = {}
    for split in splits:
        out_root = Path(str(cfg.paths.volumes_dir)) / f"{split}_real"
        for batch in loaders[split]:
            sid = batch["subject_id"]
            sid = str(sid[0] if isinstance(sid, (list, tuple)) else sid)
            x = rearrange(batch["vols"], "b s c d h w -> (b s) c d h w")
            if (out_root / sid).exists():  # never reuse stale segmentations
                shutil.rmtree(out_root / sid)
            segmenter.segment([v[0].numpy() for v in x], out_root / sid)
        written[split] = out_root
    return written
