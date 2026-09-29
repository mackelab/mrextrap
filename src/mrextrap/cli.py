"""Command line interface: ``mrextrap <subcommand> --config X.yaml [key=value ...]``.

Every subcommand lazily imports its stage function ``fn(cfg)``, so ``mrextrap --help`` works
without heavy imports. Without ``--config`` the merged config is ``configs/paths.yaml`` (if it exists
in the working directory) plus the dotlist overrides; ``mrextrap smoke`` uses only the overrides
(default: a temporary run directory).
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from omegaconf import OmegaConf

# subcommand -> (module, function, help); every function takes the merged config ``fn(cfg)``
SUBCOMMANDS: dict[str, tuple[str, str, str]] = {
    "make-splits": ("mrextrap.stages", "run_make_splits", "Build train/val/test split CSVs"),
    "train-ae": ("mrextrap.training.autoencoder", "train_autoencoder", "Train the autoencoder"),
    "encode": ("mrextrap.encoding", "encode_splits", "Encode scans into latents"),
    "fit-betas": (
        "mrextrap.progression",
        "run_fit_betas",
        "Fit per-subject latent progression rates",
    ),
    "global-prior": (
        "mrextrap.progression",
        "run_global_prior",
        "Compute the Global prior parameters",
    ),
    "train-unet": ("mrextrap.training.unet_prior", "train_unet_prior", "Train the UNet prior"),
    "train-diffusion": (
        "mrextrap.training.diffusion_prior",
        "train_diffusion_prior",
        "Train the diffusion prior",
    ),
    "real-volumes": ("mrextrap.stages", "run_real_volumes", "Segment the real scans (SynthSeg)"),
    "eval-single": (
        "mrextrap.evaluation.single_volume",
        "run_eval_single",
        "Single-volume evaluation (Table 1)",
    ),
    "eval-multi": (
        "mrextrap.evaluation.multivolume",
        "run_multivolume_eval",
        "Multi-volume evaluation (Fig. 3)",
    ),
    "eval-ae": (
        "mrextrap.evaluation.autoencoder",
        "run_eval_ae",
        "Autoencoder evaluation (SSIM, generalized Dice)",
    ),
    "interpolation": (
        "mrextrap.evaluation.interpolation",
        "run_interpolation",
        "Longitudinal interpolation (Fig. 2)",
    ),
    "figures": ("mrextrap.figures", "make_all_figures", "Render paper figures"),
    "table1": ("mrextrap.evaluation.stats", "run_table1", "Render Table 1"),
    "smoke": ("mrextrap.smoke", "run_smoke", "Run the whole pipeline on synthetic data"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mrextrap", description="MRExtrap command line interface."
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    for name, (_, _, help_text) in SUBCOMMANDS.items():
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.add_argument("--config", "-c", help="Stage YAML (merged with paths.yaml)")
        p.add_argument("overrides", nargs="*", help="Dotlist overrides, e.g. training.num_epochs=2")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    module_name, fn_name, _ = SUBCOMMANDS[args.command]
    try:
        module = importlib.import_module(module_name)
        fn = getattr(module, fn_name)
    except (ImportError, AttributeError) as err:
        print(f"mrextrap {args.command}: stage not available ({err})", file=sys.stderr)
        return 2
    from mrextrap.config import load_config

    if args.config:
        cfg = load_config(args.config, args.overrides)
    elif args.command != "smoke" and Path("configs/paths.yaml").is_file():
        cfg = load_config("configs/paths.yaml", args.overrides)
    else:
        cfg = OmegaConf.from_dotlist(args.overrides)
    fn(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
