"""Encode scans into the latent space of the trained autoencoder.

Ported from the (commented-out) encoding loop of ``create_latent_betas.py``. Writes
``{split}_zs_ages.pt`` in the legacy-compatible schema checked by ``mrextrap.io.validate_latent_dict``.

Entry points: ``encode_split(vae, dataloader, device, dataset_type)``, ``load_autoencoder(path)``
and ``encode_splits(cfg)`` (all splits of a config; use this one for ``mrextrap encode``).
"""

from __future__ import annotations

from pathlib import Path

import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from mrextrap.io import load_state_dict_any, save_latent_dict, validate_latent_dict
from mrextrap.models.autoencoder import VAE

MODEL_FILENAME = "model.safetensors"
CONFIG_FILENAME = "config.yaml"


def _resolve_autoencoder_dir(path: Path) -> Path:
    """Accept ``run_dir``, ``run_dir/autoencoder``, a checkpoint directory or a weights file."""
    if path.is_file():
        return path.parent
    if (path / "autoencoder").is_dir():
        return path / "autoencoder"
    return path


def load_autoencoder(path, device="cpu", model_cfg=None) -> VAE:
    """Load a VAE in eval mode from ``model.safetensors`` (or another format ``load_state_dict_any`` reads).

    ``path`` is a run directory, its ``autoencoder`` directory, a checkpoint directory or a weights
    file. The architecture is read from the ``config.yaml`` next to the weights unless ``model_cfg``
    (the ``model`` block of the autoencoder config) is given.
    """
    from mrextrap.training.autoencoder import build_vae

    path = Path(path)
    directory = _resolve_autoencoder_dir(path)
    weights = path if path.is_file() else directory / MODEL_FILENAME
    if model_cfg is None:
        candidates = [directory / CONFIG_FILENAME, directory.parent.parent / CONFIG_FILENAME]
        config_path = next((c for c in candidates if c.is_file()), None)
        if config_path is None:
            raise FileNotFoundError(
                f"No {CONFIG_FILENAME} found next to {weights}; pass model_cfg=cfg.model."
            )
        model_cfg = OmegaConf.load(config_path).model
    vae = build_vae(model_cfg)
    vae.load_state_dict(load_state_dict_any(weights))
    return vae.to(device).eval()


@torch.no_grad()
def encode_split(vae: VAE, dataloader, device="cpu", dataset_type: str = "test") -> dict:
    """Encode every subject series of a longitudinal dataloader (batch size 1) into a latent dict.

    Subjects with a single scan are skipped. ``zs[i]`` is ``(N_i, C, D, H, W)``, ``ages[i]`` and
    ``ages_abs[i]`` are ``(1, N_i)``, ``subject_ids[i]`` repeats the subject id per scan.
    """
    vae = vae.to(device).eval()
    subject_ids, zs, ages_per_subject, ages_abs_per_subject = [], [], [], []
    for batch in tqdm(dataloader, total=len(dataloader)):
        x = batch["vols"]
        assert x.shape[0] == 1, "batch size must be 1 (one subject series per batch)"
        if x.shape[1] == 1:
            continue  # skip if only one timepoint
        n_scans = x.shape[1]
        subject_id = batch["subject_id"]
        subject_id = subject_id[0] if isinstance(subject_id, (list, tuple)) else subject_id
        subject_ids.append([str(subject_id)] * n_scans)
        ages_per_subject.append(batch["ages"].cpu())
        ages_abs_per_subject.append(batch["ages_abs"].cpu())
        x = rearrange(x, "b s c d h w -> (b s) c d h w")
        z = []
        for x1 in x:
            # NOTE(paper) Q11: vae.encode() is called directly (not vae()), so it runs in fp32 even
            # when the model was prepared with bf16 mixed precision; only the mean is kept.
            z1 = vae.encode(x1.unsqueeze(0).to(device=device, dtype=torch.float32))[0]
            z.append(z1)
        zs.append(torch.cat(z).cpu())
    latent_dict = {
        "zs": zs,
        "ages": ages_per_subject,
        "ages_abs": ages_abs_per_subject,
        "subject_ids": subject_ids,
        "dataset_type": dataset_type,
    }
    validate_latent_dict(latent_dict)
    return latent_dict


def encode_splits(cfg: DictConfig) -> dict[str, Path]:
    """Encode the splits in ``cfg.encoding.splits`` (default all) to ``latents_dir/{split}_zs_ages.pt``.

    Weights come from ``cfg.encoding.checkpoint`` (default: ``<paths.run_dir>/autoencoder``).
    """
    from mrextrap.data.datasets import get_longitudinal_dataloaders

    enc = cfg.get("encoding", None) or {}
    splits = list(enc.get("splits", None) or ("train", "val", "test"))
    device = enc.get("device", None) or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = enc.get("checkpoint", None) or Path(str(cfg.paths.run_dir)) / "autoencoder"
    vae = load_autoencoder(checkpoint, device=device, model_cfg=cfg.model)

    cfg = cfg.copy()
    cfg.training.batch_size = 1  # one subject series per batch, always
    loaders = dict(
        zip(
            ("train", "val", "test"),
            get_longitudinal_dataloaders(
                cfg, num_workers=cfg.training.get("num_workers", 0), shuffle_train=False
            ),
            strict=True,
        )
    )
    out_dir = Path(str(cfg.paths.latents_dir))
    written = {}
    for split in splits:
        latent_dict = encode_split(vae, loaders[split], device, dataset_type=split)
        written[split] = out_dir / f"{split}_zs_ages.pt"
        save_latent_dict(latent_dict, written[split])
    return written
