"""Configuration loading: ``paths.yaml`` + stage yaml + dotlist overrides."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

PATHS_FILENAME = "paths.yaml"


def find_paths_yaml(config_path: Path) -> Path | None:
    """Look for ``paths.yaml`` next to the stage config, then one directory up (``configs/smoke``)."""
    for directory in (config_path.parent, config_path.parent.parent):
        candidate = directory / PATHS_FILENAME
        if candidate.is_file() and candidate.resolve() != config_path.resolve():
            return candidate
    return None


def load_config(
    path: str | Path, overrides: Iterable[str] = (), paths_yaml: str | Path | None = None
) -> DictConfig:
    """Merge ``paths.yaml`` (under key ``paths``), the stage yaml and dotlist overrides.

    Precedence (low to high): paths.yaml < stage yaml < overrides. The stage yaml may use
    interpolations such as ``${paths.run_dir}``; they are resolved lazily by OmegaConf.
    A stage yaml may also contain its own top-level ``paths`` block, which wins over paths.yaml.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    stage = OmegaConf.load(path)

    if paths_yaml is None:
        paths_yaml = find_paths_yaml(path)
    parts = []
    if paths_yaml is not None:
        loaded = OmegaConf.load(paths_yaml)
        # paths.yaml is either {run_dir: ...} or {paths: {run_dir: ...}}; normalise to the latter
        parts.append(loaded if "paths" in loaded else OmegaConf.create({"paths": loaded}))
    parts.append(stage)
    overrides = list(overrides)
    if overrides:
        parts.append(OmegaConf.from_dotlist(overrides))
    cfg = OmegaConf.merge(*parts)
    assert isinstance(cfg, DictConfig)
    return cfg
