"""Autoencoder evaluation (paper Sec. 3.1): SSIM and generalized Dice of reconstructions.

Ports ``eval_autoencoder_recs.py`` (SSIM, real/reconstructed SynthSeg volumes) and the GDice cells
(29-31) of ``eval_autoencoder_cross_sectional_recs.ipynb``. All metrics are computed with the paper
autoencoder (``older8``): the model is loaded from the run directory, never from a hard-coded path.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm.auto import tqdm

from mrextrap.losses import SSIM
from mrextrap.segmentation.volumes import load_segmentations

logger = logging.getLogger(__name__)

# Label ids of the regions used for the GDice (``eval.py::seg_labels`` of the research code).
GDICE_REGIONS = ("hippocampus", "amygdala", "ventricle", "cortex", "white")
SEG_LABELS: dict[str, list[float]] = {
    "ventricle": [4.0, 5.0, 14.0, 15.0, 43.0, 44.0],
    "hippocampus": [17.0, 53.0],
    "amygdala": [18.0, 54.0],
    "cortex": [3.0, 8.0, 42.0, 47.0],
    "white": [2.0, 7.0, 41.0, 46.0],
}


def extract_multiple_from_seg(
    seg: torch.Tensor, seg_labels: dict[str, list[float]]
) -> torch.Tensor:
    """``[B, D, H, W]`` label map -> ``[B, len(seg_labels), D, H, W]`` binary region masks (verbatim port)."""
    seg_shape = seg.shape
    seg = rearrange(seg, "b d h w -> (b d h w)")
    seg_channels = []
    for lbls in seg_labels.values():
        seg_channel = seg.clone()
        mask = torch.zeros_like(seg_channel, dtype=torch.bool)
        for lbl in lbls:
            mask |= seg == lbl
        seg_channel[mask] = 1.0
        seg_channel[~mask] = 0.0
        seg_channels.append(seg_channel)
    seg_channels = torch.stack(seg_channels, dim=0)
    return rearrange(
        seg_channels,
        "c (b d h w) -> b c d h w",
        c=len(seg_labels),
        b=seg_shape[0],
        d=seg_shape[1],
        h=seg_shape[2],
        w=seg_shape[3],
    )


def one_hot(tensor: torch.Tensor, num_classes: int = 60) -> torch.Tensor:
    """``(N, 1, D, H, W)`` int tensor -> ``(N, num_classes, D, H, W)`` one-hot (verbatim port).

    NOTE(paper): only ``tensor[:, 0]`` is used, i.e. the first channel.
    """
    return F.one_hot(tensor[:, 0], num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


def _region_masks(seg: np.ndarray | torch.Tensor, regions: Sequence[str]) -> torch.Tensor:
    """``[D, H, W]`` SynthSeg label map -> ``[len(regions), D, H, W]`` binary region masks."""
    labels = {r: SEG_LABELS[r] for r in regions}
    return extract_multiple_from_seg(torch.as_tensor(np.asarray(seg))[None], labels)[0]


def per_region_dice(
    seg_real: np.ndarray | torch.Tensor,
    seg_recon: np.ndarray | torch.Tensor,
    regions: Sequence[str] = GDICE_REGIONS,
) -> dict[str, float]:
    """Binary Dice per region (``nan`` if the region is empty in both maps)."""
    y, y_pred = _region_masks(seg_real, regions).bool(), _region_masks(seg_recon, regions).bool()
    out = {}
    for i, region in enumerate(regions):
        denom = float(y[i].sum() + y_pred[i].sum())
        out[region] = 2.0 * float((y[i] & y_pred[i]).sum()) / denom if denom > 0 else float("nan")
    return out


def generalized_dice(
    seg_real: np.ndarray | torch.Tensor,
    seg_recon: np.ndarray | torch.Tensor,
    regions: Sequence[str] = GDICE_REGIONS,
    mode: str = "regions",
) -> float:
    """Generalized Dice of one real vs reconstructed SynthSeg label map ``[D, H, W]``.

    ``mode="regions"`` (default, the correct 5-region score): the 5 binary region masks (hippocampus,
    amygdala, ventricle, cortex, white matter; disjoint) are combined into one label map with values
    ``1..5`` and background 0, one-hot encoded with 6 classes, and scored with
    ``monai.metrics.GeneralizedDiceScore(include_background=False)``.

    ``mode="notebook"`` reproduces the number of the preprint (0.92) exactly as the paper notebook
    computed it: ``extract_multiple_from_seg`` with the 5 regions, then
    ``one_hot(seg_extracted[i].unsqueeze(0).long(), num_classes=5)`` and ``GeneralizedDiceScore``
    with monai's defaults.

    NOTE(paper): the notebook's ``one_hot`` reads only channel 0 of the ``[1, 5, D, H, W]`` extracted
    tensor, i.e. the *first region* (hippocampus) as a 0/1 map, expanded to 5 classes of which only
    classes 0 (not hippocampus) and 1 (hippocampus) are ever populated. The "5-region" GDice of the
    paper is therefore a two-class (hippocampus vs rest, background included) generalized Dice.
    """
    from monai.metrics import GeneralizedDiceScore

    y = _region_masks(seg_real, regions)
    y_pred = _region_masks(seg_recon, regions)
    if mode == "notebook":
        y = one_hot(y.unsqueeze(0).long(), num_classes=len(regions))
        y_pred = one_hot(y_pred.unsqueeze(0).long(), num_classes=len(regions))
        return float(GeneralizedDiceScore()(y_pred=y_pred, y=y).mean())
    if mode == "regions":
        weights = torch.arange(1, len(regions) + 1).view(-1, 1, 1, 1)
        n_classes = len(regions) + 1

        def to_one_hot(masks: torch.Tensor) -> torch.Tensor:
            label_map = (masks.long() * weights).sum(dim=0)  # masks are disjoint
            return F.one_hot(label_map, n_classes).permute(3, 0, 1, 2).unsqueeze(0).float()

        score = GeneralizedDiceScore(include_background=False)
        return float(score(y_pred=to_one_hot(y_pred), y=to_one_hot(y)).mean())
    raise ValueError(f"mode must be 'regions' or 'notebook', got {mode!r}")


def _subject_id(batch) -> str:
    sid = batch["subject_id"]
    return str(sid[0] if isinstance(sid, (list, tuple)) else sid)


def _fresh(path: Path) -> Path:
    if path.exists():  # never reuse stale segmentations of an earlier run
        shutil.rmtree(path)
    return path


def reconstruction_metrics(
    vae,
    loader,
    segmenter,
    out_dir: str | Path,
    split: str = "test",
    device: str = "cpu",
    gdice_mode: str = "regions",
    autocast_bf16: bool | None = None,
) -> dict:
    """SSIM and GDice of ``vae`` reconstructions; writes real and reconstructed SynthSeg volumes.

    ``loader`` yields one subject series per batch (``batch["vols"]`` of shape ``[1, S, 1, D, H, W]``).
    Real scans are segmented into ``out_dir/<split>_real/<sid>`` (these ``vols.csv`` are the real
    volumes used by every evaluation) and reconstructions into ``out_dir/<split>_recon/<sid>``;
    ``out_dir/<split>_ae_metrics.json`` holds the summary. Reconstruction is deterministic
    (``stochastic_sampling=False``); ``autocast_bf16=None`` enables bf16 autocast on CUDA only
    (paper: Q11, bf16 autocast for ``vae(x)``).

    ``gdice_mode="regions"`` (default) is the correct 5-region generalized Dice; ``"notebook"``
    reproduces the preprint number (hippocampus channel only, see ``generalized_dice``). The
    per-region Dice is reported in ``dice_per_region`` (mean over scans) either way.

    Returns ``{"ssim", "gdice", "gdice_median", "dice_per_region", "n_scans", "n_subjects",
    "ssim_per_scan", "gdice_per_scan", "gdice_mode"}``.
    """
    out_dir = Path(out_dir)
    if autocast_bf16 is None:
        autocast_bf16 = torch.device(device).type == "cuda"
    vae = vae.to(device).eval()
    # NOTE(paper): SSIM(data_range=1.0, 3D, per-image) exactly as in eval_autoencoder_recs.py.
    ssim = SSIM(data_range=1.0, size_average=False, channel=1, spatial_dims=3)
    ssims, gdices, dices, sids = [], [], [], []
    for batch in tqdm(loader, total=len(loader)):
        x = batch["vols"]
        assert x.shape[0] == 1, "batch size must be 1 (one subject series per batch)"
        sid = _subject_id(batch)
        x = rearrange(x, "b s c d h w -> (b s) c d h w")
        xhat = []
        for x1 in x:
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=torch.device(device).type,
                    dtype=torch.bfloat16,
                    enabled=autocast_bf16,
                ),
            ):
                xhat.append(
                    vae(x1.unsqueeze(0).to(device), stochastic_sampling=False)[0].float().cpu()
                )
        xhat = torch.cat(xhat, dim=0)
        ssims.append(ssim(x.cpu(), xhat))

        real_dir = _fresh(out_dir / f"{split}_real" / sid)
        recon_dir = _fresh(out_dir / f"{split}_recon" / sid)
        segmenter.segment([x1[0].cpu().numpy() for x1 in x], real_dir)
        segmenter.segment([x1[0].numpy() for x1 in xhat], recon_dir)
        real_segs = load_segmentations(real_dir, len(x))
        recon_segs = load_segmentations(recon_dir, len(x))
        gdices += [
            generalized_dice(r, p, mode=gdice_mode)
            for r, p in zip(real_segs, recon_segs, strict=True)
        ]
        dices += [per_region_dice(r, p) for r, p in zip(real_segs, recon_segs, strict=True)]
        sids.append(sid)

    ssim_all = torch.cat(ssims)
    metrics = {
        "ssim": float(ssim_all.mean()),
        "gdice": float(np.mean(gdices)),
        "gdice_median": float(np.median(gdices)),
        "dice_per_region": {
            r: float(np.nanmean([d[r] for d in dices])) if dices else float("nan")
            for r in GDICE_REGIONS
        },
        "n_scans": len(gdices),
        "n_subjects": len(sids),
        "ssim_per_scan": [float(s) for s in ssim_all],
        "gdice_per_scan": [float(g) for g in gdices],
        "gdice_mode": gdice_mode,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{split}_ae_metrics.json").write_text(json.dumps(metrics))
    logger.info("SSIM %.4f  GDice %.4f (%d scans)", metrics["ssim"], metrics["gdice"], len(gdices))
    return metrics


def run_eval_ae(cfg) -> dict:
    """``mrextrap eval-ae``: SSIM/GDice of the autoencoder in ``<run_dir>/autoencoder``.

    Uses ``cfg.ae_eval`` (``split`` default "test", ``device``, ``gdice_mode``, ``autoencoder`` to
    override the checkpoint) and the dataloaders of the config.
    """
    from mrextrap.data.datasets import get_longitudinal_dataloaders
    from mrextrap.encoding import load_autoencoder
    from mrextrap.segmentation.volumes import make_segmenter

    ev = cfg.get("ae_eval", None) or cfg.get("eval", {})
    run_dir = Path(str(cfg.paths.run_dir))
    device = ev.get("device", "cpu")
    split = ev.get("split", "test")
    vae = load_autoencoder(
        ev.get("autoencoder", None) or run_dir / "autoencoder",
        device=device,
        model_cfg=cfg.get("model", None),
    )
    cfg = cfg.copy()
    cfg.training.batch_size = 1
    loaders = dict(
        zip(
            ("train", "val", "test"),
            get_longitudinal_dataloaders(cfg, num_workers=cfg.training.get("num_workers", 0)),
            strict=True,
        )
    )
    segmenter = make_segmenter(cfg, device)
    return reconstruction_metrics(
        vae,
        loaders[split],
        segmenter,
        Path(str(cfg.paths.volumes_dir)),
        split=split,
        device=device,
        gdice_mode=ev.get("gdice_mode", "regions"),
    )
