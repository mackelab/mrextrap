#!/usr/bin/env python
"""Convert the paper checkpoints (legacy accelerate state dirs) into the MRExtrap release layout.

Inputs are the *legacy* checkpoint locations on the cluster; nothing is modified in place.

    out_dir/
      autoencoder/model.safetensors                                  (+ config.yaml if --ae-config)
      unet_prior/checkpoints/epoch_<L>/model.safetensors
      diffusion_prior/checkpoints/epoch_<L>/ema_denoiser.safetensors
      synthseg_torch.pt                                              (copied byte for byte)
      priors/{global_params,normalization}.pt                        (optional, see below)
      conversion_manifest.{json,md}      sources, sha256, tensor counts, epoch semantics (Q4)

State dicts are loaded with ``mrextrap.io.load_state_dict_any`` (strips ``_orig_mod.`` from
torch.compile and ``module.`` from DDP), stored with safetensors and re-read to verify that every
tensor is bit-identical to the stripped source.

Example (cluster)::

    python tools/convert_checkpoints.py --out-dir "$WORK/thesis_rerender/mrextrap/release_weights" \\
        --ae-dir exp/train_autoencoder_older8/epoch_124/epoch_124 --ae-config configs/autoencoder.yaml \\
        --unet-dir exp/<unet_exp_name>/epoch_2000 --diffusion-dir exp/<diffusion_exp_name>/epoch_2000 \\
        --synthseg-weights synthseg_torch/synthseg_torch.pt \\
        --train-latents exp_eval/autoencoder_older/train_latents/latent_betas_l1.pt
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import markdown_table, sha256_file, write_report

AE_CANDIDATES = ("model.safetensors", "pytorch_model.bin", "model.bin")
UNET_CANDIDATES = ("model.safetensors", "pytorch_model.bin", "model.bin")
DIFFUSION_CANDIDATES = ("ema_denoiser.bin", "ema_denoiser.safetensors", "ema_denoiser.pt")

# NOTE(paper): Q4. The legacy label is ``(epoch + 1) // 20 * 20 + 20`` with a save every 10 epochs,
# so ``epoch_2000`` holds the weights after 1990 epochs. Q(AE): label ``epoch // 2 * 2 + 2``.
EPOCH_NOTES = {
    "unet_prior": "legacy label epoch_{label} = weights after {prev} epochs (label is "
    "(epoch+1)//20*20+20, saved every 10 epochs; Q4)",
    "diffusion_prior": "EMA denoiser; legacy label epoch_{label} = weights after {prev} epochs (Q4)",
    "autoencoder": "paper autoencoder (legacy name train_autoencoder_older8); legacy label "
    "epoch_{label} = state after {label} finished epochs",
}


def find_weights(path: str | Path, candidates: tuple[str, ...]) -> Path:
    """``path`` itself if it is a file, else the first existing candidate inside the directory."""
    path = Path(path)
    if path.is_file():
        return path
    for name in candidates:
        if (path / name).is_file():
            return path / name
    raise FileNotFoundError(f"none of {list(candidates)} found in {path}")


def convert_state_dict(src: Path, dst: Path) -> dict:
    """Legacy weights -> ``dst`` (safetensors); verify the round trip; return manifest fields."""
    from mrextrap.io import load_state_dict_any, save_safetensors

    raw_keys = _raw_keys(src)
    sd = load_state_dict_any(src)
    sd = {k: v.detach().clone().contiguous() for k, v in sd.items() if torch.is_tensor(v)}
    save_safetensors(sd, dst)
    back = load_state_dict_any(dst)
    if set(back) != set(sd):
        raise RuntimeError(f"key mismatch after writing {dst}")
    for k, v in sd.items():
        if not torch.equal(back[k], v):
            raise RuntimeError(f"tensor {k} changed in the round trip to {dst}")
    stripped = sum(1 for k in raw_keys if k not in sd)
    return {
        "source": str(src),
        "source_sha256": sha256_file(src),
        "file": str(dst),
        "sha256": sha256_file(dst),
        "n_tensors": len(sd),
        "n_parameters": int(sum(v.numel() for v in sd.values())),
        "dtypes": sorted({str(v.dtype) for v in sd.values()}),
        "n_keys_with_stripped_prefix": stripped,
    }


def _raw_keys(src: Path) -> list[str]:
    if src.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(src), framework="pt") as f:
            return list(f.keys())
    obj = torch.load(src, map_location="cpu", weights_only=True)
    for wrapper in ("state_dict", "model"):
        if isinstance(obj, dict) and isinstance(obj.get(wrapper), dict):
            obj = obj[wrapper]
            break
    return list(obj)


def _entry(rel_file: str, sha: str | None, note: str | None = None) -> dict:
    entry = {"file": rel_file, "sha256": sha}
    if note:
        entry["note"] = note
    return entry


def run(args: argparse.Namespace) -> dict:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    converted: dict[str, dict] = {}
    weights: dict[str, dict] = {}

    jobs = [
        ("autoencoder", args.ae_dir, AE_CANDIDATES, "autoencoder/model.safetensors", args.ae_epoch),
        (
            "unet_prior",
            args.unet_dir,
            UNET_CANDIDATES,
            f"unet_prior/checkpoints/epoch_{args.unet_epoch}/model.safetensors",
            args.unet_epoch,
        ),
        (
            "diffusion_prior",
            args.diffusion_dir,
            DIFFUSION_CANDIDATES,
            f"diffusion_prior/checkpoints/epoch_{args.diffusion_epoch}/ema_denoiser.safetensors",
            args.diffusion_epoch,
        ),
    ]
    for name, src_dir, candidates, rel, label in jobs:
        if src_dir is None:
            continue
        info = convert_state_dict(find_weights(src_dir, candidates), out / rel)
        info["epoch_label"] = label
        info["epoch_semantics"] = EPOCH_NOTES[name].format(label=label, prev=label - 10)
        converted[name] = info
        weights[name] = _entry(rel, info["sha256"], info["epoch_semantics"])

    if args.ae_config and args.ae_dir is not None:
        shutil.copyfile(args.ae_config, out / "autoencoder" / "config.yaml")

    if args.global_params or args.train_latents:
        (out / "priors").mkdir(exist_ok=True)
    gp_path = out / "priors" / "global_params.pt"
    if args.global_params:
        shutil.copyfile(args.global_params, gp_path)
    if args.train_latents:
        from mrextrap.io import load_latent_dict
        from mrextrap.progression import compute_global_prior, compute_normalization

        latents = load_latent_dict(args.train_latents)
        torch.save(compute_normalization(latents), out / "priors" / "normalization.pt")
        if not args.global_params:
            torch.save(compute_global_prior(latents), gp_path)
    for key, rel in (
        ("global_params", "priors/global_params.pt"),
        ("normalization", "priors/normalization.pt"),
    ):
        if (out / rel).is_file():
            weights[key] = _entry(rel, sha256_file(out / rel))
            converted[key] = {"file": str(out / rel), "sha256": weights[key]["sha256"]}

    if args.synthseg_weights:
        dst = out / "synthseg_torch.pt"
        shutil.copyfile(args.synthseg_weights, dst)
        sd = torch.load(dst, map_location="cpu", weights_only=True)  # must be a plain state dict
        note = "torch port of SynthSeg robust; check upstream terms before redistribution"
        weights["synthseg"] = _entry("synthseg_torch.pt", sha256_file(dst), note)
        converted["synthseg"] = {
            "source": str(args.synthseg_weights),
            "file": str(dst),
            "sha256": weights["synthseg"]["sha256"],
            "n_tensors": len(sd),
        }

    if not converted:
        raise SystemExit("nothing to convert: pass at least one of --ae-dir/--unet-dir/... ")

    report = {"out_dir": str(out), "converted": converted}
    rows = [
        (
            k,
            v.get("n_tensors", ""),
            v.get("n_parameters", ""),
            v.get("epoch_semantics", ""),
            v["sha256"],
        )
        for k, v in converted.items()
    ]
    md = "# Checkpoint conversion\n\n" + markdown_table(
        ["item", "tensors", "parameters", "epoch semantics", "sha256"], rows
    )
    write_report(out, "conversion_manifest", report, md)
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out-dir", required=True, help="output directory (release weights layout)")
    p.add_argument("--ae-dir", help="accelerate state dir of the AE (e.g. .../epoch_124/epoch_124)")
    p.add_argument("--ae-config", help="AE config yaml copied to autoencoder/config.yaml")
    p.add_argument("--ae-epoch", type=int, default=124, help="legacy AE epoch label")
    p.add_argument("--unet-dir", help="legacy UNet prior checkpoint dir (epoch_2000)")
    p.add_argument("--unet-epoch", type=int, default=2000, help="legacy UNet epoch label")
    p.add_argument(
        "--diffusion-dir", help="legacy diffusion checkpoint dir (holds ema_denoiser.bin)"
    )
    p.add_argument("--diffusion-epoch", type=int, default=2000, help="legacy diffusion epoch label")
    p.add_argument("--synthseg-weights", help="synthseg_torch.pt")
    p.add_argument("--global-params", help="legacy global_params.pt to copy into priors/")
    p.add_argument(
        "--train-latents",
        help="train latent_betas_l1.pt; writes priors/normalization.pt (and global_params.pt "
        "unless --global-params is given)",
    )
    return p


def main(argv=None) -> int:
    report = run(build_parser().parse_args(argv))
    for name, info in report["converted"].items():
        print(f"{name:16s} sha256 {info['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
