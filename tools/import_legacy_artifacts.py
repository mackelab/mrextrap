#!/usr/bin/env python
"""Map the legacy cluster artefacts (``exp_eval/...``, ``exp/...``, ``data/...``) onto the MRExtrap
run-directory layout (RELEASE_PLAN 2.1) by symlink (default) or copy, so the release code and
``validate_against_legacy.py`` can run on the paper artefacts without retraining.

Nothing legacy is modified. Existing targets are kept unless ``--force``. Reports only counts of
linked / missing items (no subject identifiers).

Example (cluster, from the legacy repo root)::

    python tools/import_legacy_artifacts.py --legacy-root . --run-dir "$WORK/thesis_rerender/mrextrap/run_dir" \\
        [--weights-dir "$WORK/thesis_rerender/mrextrap/release_weights"] [--dry-run]

Layout notes
* AE / diffusion legacy checkpoints are *not* usable directly (accelerate state / ``ema_denoiser.bin``):
  they are linked under ``autoencoder/legacy_state`` and ``diffusion_prior/legacy`` for the record;
  run ``convert_checkpoints.py`` and pass its output as ``--weights-dir`` (or point paths at it).
* The UNet legacy dir already holds ``model.safetensors`` and is linked as
  ``unet_prior/checkpoints/epoch_2000``.
* The diffusion single-volume results exist only as ``vols.csv`` (no results.json): linked as
  ``eval/single_volume/diffusion/legacy_segs``.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import markdown_table, write_report

SPLITS = ("train", "val", "test")
AE_LATENT_ROOT = "exp_eval/autoencoder_older"
DEFAULT_AE_EXP = "train_autoencoder_older8"
DEFAULT_UNET_EXP = "train_unet_mean_var_with_gauss_loss_lowlr_aug_warmup4000"
DEFAULT_DIFFUSION_EXP = "train_beta_diffusion_smaller64highlr3e-4longtrainlinear"


@dataclass(frozen=True)
class Mapping:
    key: str  # short id used in the report
    source: str  # relative to the source root
    target: str  # relative to run_dir
    kind: str = "file"  # "file" | "dir" | "glob" (source has wildcards, target is a directory)
    required: bool = True


def legacy_mappings(
    ae_exp: str = DEFAULT_AE_EXP,
    unet_exp: str = DEFAULT_UNET_EXP,
    diffusion_exp: str = DEFAULT_DIFFUSION_EXP,
    ae_epoch: int = 124,
    prior_epoch: int = 2000,
) -> list[Mapping]:
    """The legacy -> run_dir table (covers every legacy path of RELEASE_PLAN section 1)."""
    m: list[Mapping] = []
    for s in SPLITS:
        m += [
            Mapping(
                f"latents_{s}",
                f"{AE_LATENT_ROOT}/{s}_latents/zs_ages.pt",
                f"latents/{s}_zs_ages.pt",
                required=False,
            ),
            Mapping(
                f"betas_l1_{s}",
                f"{AE_LATENT_ROOT}/{s}_latents/latent_betas_l1.pt",
                f"latents/{s}_latent_betas_l1.pt",
            ),
            Mapping(
                f"betas_{s}",
                f"{AE_LATENT_ROOT}/{s}_latents/latent_betas.pt",
                f"latents/{s}_latent_betas.pt",
                required=False,
            ),
            Mapping(
                f"real_vols_{s}",
                f"{AE_LATENT_ROOT}/{s}_real_subjects",
                f"volumes/{s}_real",
                "dir",
                required=(s == "test"),
            ),
            Mapping(
                f"recon_vols_{s}",
                f"{AE_LATENT_ROOT}/{s}_recon_subjects",
                f"volumes/{s}_recon",
                "dir",
                required=False,
            ),
            Mapping(f"split_csv_{s}", f"data/{s}_df_nona.csv", f"splits/{s}.csv"),
        ]
    m += [
        Mapping(
            "global_params",
            f"{AE_LATENT_ROOT}/train_latents/global_params.pt",
            "priors/global_params.pt",
        ),
        Mapping(
            "single_global",
            "exp_eval/global_prior/single_volume/results.json",
            "eval/single_volume/global/results.json",
        ),
        Mapping(
            "single_global_segs",
            "exp_eval/global_prior/single_volume/temp_seg_*",
            "eval/single_volume/global",
            "glob",
            required=False,
        ),
        Mapping(
            "single_unet",
            "exp_eval/unet_heteroscedastic/single_volume/results.json",
            "eval/single_volume/unet/results.json",
        ),
        Mapping(
            "single_diffusion_segs",
            f"{AE_LATENT_ROOT}/test_daninetdiffusion_subjects",
            "eval/single_volume/diffusion/legacy_segs",
            "dir",
        ),
    ]
    for base in ("earliest", "latest"):
        m += [
            Mapping(
                f"multi_{base}_results",
                f"exp_eval/global_prior/varcond_{base}/results_var1.00.json",
                f"eval/multivolume/{base}/results_var1.00.json",
            ),
            Mapping(
                f"multi_{base}_segs",
                f"exp_eval/global_prior/varcond_{base}/temp_seg_*",
                f"eval/multivolume/{base}",
                "glob",
                required=False,
            ),
        ]
    m += [
        Mapping(
            "ae_state",
            f"exp/{ae_exp}/epoch_{ae_epoch}/epoch_{ae_epoch}",
            "autoencoder/legacy_state",
            "dir",
        ),
        Mapping(
            "unet_ckpt",
            f"exp/{unet_exp}/epoch_{prior_epoch}",
            f"unet_prior/checkpoints/epoch_{prior_epoch}",
            "dir",
        ),
        Mapping(
            "diffusion_ckpt",
            f"exp/{diffusion_exp}/epoch_{prior_epoch}",
            "diffusion_prior/legacy",
            "dir",
        ),
        Mapping(
            "synthseg_weights", "synthseg_torch/synthseg_torch.pt", "weights/synthseg_torch.pt"
        ),
    ]
    return m


def converted_mappings() -> list[Mapping]:
    """Output of ``convert_checkpoints.py`` (``--weights-dir``) -> run_dir."""
    return [
        Mapping("conv_ae", "autoencoder/model.safetensors", "autoencoder/model.safetensors"),
        Mapping(
            "conv_ae_cfg", "autoencoder/config.yaml", "autoencoder/config.yaml", required=False
        ),
        Mapping(
            "conv_unet",
            "unet_prior/checkpoints/*",
            "unet_prior/checkpoints",
            "glob",
            required=False,
        ),
        Mapping(
            "conv_diffusion",
            "diffusion_prior/checkpoints/*",
            "diffusion_prior/checkpoints",
            "glob",
            required=False,
        ),
        Mapping(
            "conv_normalization",
            "priors/normalization.pt",
            "priors/normalization.pt",
            required=False,
        ),
        Mapping("conv_synthseg", "synthseg_torch.pt", "weights/synthseg_torch.pt", required=False),
    ]


def _place(src: Path, dst: Path, mode: str, force: bool) -> str:
    if dst.exists() or dst.is_symlink():
        if not force:
            return "exists"
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return "linked" if mode == "symlink" else "copied"


def apply_mappings(
    mappings: list[Mapping], root: Path, run_dir: Path, mode="symlink", dry_run=False, force=False
) -> list[dict]:
    rows = []
    for m in mappings:
        if m.kind == "glob":
            matches = sorted(glob.glob(str(root / m.source)))
            statuses = []
            for match in matches:
                src = Path(match)
                dst = run_dir / m.target / src.name
                statuses.append("dry-run" if dry_run else _place(src, dst, mode, force))
            rows.append(
                {
                    "key": m.key,
                    "kind": m.kind,
                    "required": m.required,
                    "n_matches": len(matches),
                    "status": "missing"
                    if not matches
                    else (statuses[0] if len(set(statuses)) == 1 else "mixed"),
                }
            )
            continue
        src, dst = root / m.source, run_dir / m.target
        if not src.exists():
            rows.append(
                {
                    "key": m.key,
                    "kind": m.kind,
                    "required": m.required,
                    "n_matches": 0,
                    "status": "missing",
                }
            )
            continue
        status = "dry-run" if dry_run else _place(src, dst, mode, force)
        rows.append(
            {"key": m.key, "kind": m.kind, "required": m.required, "n_matches": 1, "status": status}
        )
    return rows


def run(args: argparse.Namespace) -> dict:
    run_dir = Path(args.run_dir)
    rows = apply_mappings(
        legacy_mappings(args.ae_exp, args.unet_exp, args.diffusion_exp),
        Path(args.legacy_root),
        run_dir,
        args.mode,
        args.dry_run,
        args.force,
    )
    if args.weights_dir:
        rows += apply_mappings(
            converted_mappings(),
            Path(args.weights_dir),
            run_dir,
            args.mode,
            args.dry_run,
            args.force,
        )
    missing_required = [r["key"] for r in rows if r["status"] == "missing" and r["required"]]
    report = {
        "run_dir": str(run_dir),
        "mode": args.mode,
        "dry_run": args.dry_run,
        "n_items": len(rows),
        "n_missing_required": len(missing_required),
        "missing_required": missing_required,
        "items": rows,
    }
    md = "# Legacy artefact import\n\n" + markdown_table(
        ["item", "kind", "required", "matches", "status"],
        [(r["key"], r["kind"], r["required"], r["n_matches"], r["status"]) for r in rows],
    )
    if not args.dry_run:
        write_report(run_dir / "import_report", "import_legacy_artifacts", report, md)
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--legacy-root", required=True, help="legacy repo root (contains exp/, exp_eval/, data/)"
    )
    p.add_argument("--run-dir", required=True, help="target run directory")
    p.add_argument(
        "--weights-dir", help="output of convert_checkpoints.py to link into run_dir as well"
    )
    p.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="replace existing targets")
    p.add_argument("--strict", action="store_true", help="exit 1 if a required item is missing")
    p.add_argument("--ae-exp", default=DEFAULT_AE_EXP)
    p.add_argument("--unet-exp", default=DEFAULT_UNET_EXP)
    p.add_argument("--diffusion-exp", default=DEFAULT_DIFFUSION_EXP)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    for r in report["items"]:
        print(f"{r['status']:8s} {r['key']}")
    print(f"{report['n_missing_required']} required item(s) missing")
    return 1 if (args.strict and report["n_missing_required"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
