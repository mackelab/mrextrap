"""Small IO helpers: latent dictionaries and checkpoint loading/saving."""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import torch

LATENT_KEYS = ("zs", "ages", "ages_abs", "subject_ids")
_STRIP_PREFIXES = ("_orig_mod.", "module.")


def load_latent_dict(path: str | Path, weights_only: bool = True) -> dict:
    """Load a ``{split}_zs_ages.pt`` / ``{split}_latent_betas_l1.pt`` file onto the CPU.

    Paper-era files store ``subject_ids`` as ``numpy.str_``, which the weights-only unpickler
    rejects; those are retried with ``weights_only=False``. Only load files you trust.
    """
    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=weights_only)
    except pickle.UnpicklingError:
        if not weights_only:
            raise
        warnings.warn(
            f"{path}: not loadable with weights_only=True (legacy numpy objects); "
            "retrying with weights_only=False",
            stacklevel=2,
        )
        return torch.load(path, map_location="cpu", weights_only=False)


def save_latent_dict(d: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, path)


def validate_latent_dict(d: dict, require_betas: bool = False) -> None:
    """Check the legacy-compatible latent schema; raises ``ValueError`` describing the problem.

    Schema: zs, ages, ages_abs, subject_ids (one entry per subject: a list with one id per scan, or in legacy files a single id), optional betas
    (per subject, same shape as zs), dataset_type (str).
    """
    if not isinstance(d, dict):
        raise TypeError(f"Latent dict must be a dict, got {type(d).__name__}")
    required = list(LATENT_KEYS) + (["betas"] if require_betas else [])
    missing = [k for k in required if k not in d]
    if missing:
        raise ValueError(f"Latent dict is missing keys: {missing}")
    n = len(d["zs"])
    for key in [k for k in (*LATENT_KEYS, "betas") if k in d]:
        if len(d[key]) != n:
            raise ValueError(f"Latent dict key '{key}' has {len(d[key])} subjects, expected {n}")
    for i in range(n):
        z = d["zs"][i]
        if z.ndim != 5:
            raise ValueError(f"zs[{i}] must be (N,C,D,H,W), got shape {tuple(z.shape)}")
        n_scans = z.shape[0]
        ids = d["subject_ids"][i]
        # per-scan ids (list of n_scans) or legacy one id per subject (a str or a 1-element list)
        n_ids = 1 if isinstance(ids, str) else len(ids)
        if n_ids not in (1, n_scans):
            raise ValueError(f"subject_ids[{i}] has {n_ids} ids, expected 1 or {n_scans} (scans)")
        for key in ("ages", "ages_abs"):
            if d[key][i].numel() != n_scans:
                raise ValueError(f"{key}[{i}] has {d[key][i].numel()} entries, expected {n_scans}")
        if "betas" in d and tuple(d["betas"][i].shape) != tuple(z.shape):
            raise ValueError(
                f"betas[{i}] shape {tuple(d['betas'][i].shape)} != zs[{i}] shape {tuple(z.shape)}"
            )


def _strip_prefixes(sd: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in sd.items():
        changed = True
        while changed:
            changed = False
            for prefix in _STRIP_PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        out[key] = value
    return out


def load_state_dict_any(path: str | Path) -> dict:
    """Load a state dict from ``.safetensors``, ``.bin`` or ``.pt``; strip ``_orig_mod.``/``module.``."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        sd = load_file(str(path), device="cpu")
    else:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        for wrapper in ("state_dict", "model"):
            if isinstance(sd, dict) and wrapper in sd and isinstance(sd[wrapper], dict):
                sd = sd[wrapper]
                break
    return _strip_prefixes(sd)


def save_safetensors(sd: dict, path: str | Path) -> None:
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in sd.items()}, str(path))
