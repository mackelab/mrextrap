"""End-to-end run of every pipeline stage on synthetic data (CPU, FakeSegmenter, < 2 min)."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from mrextrap.cli import SUBCOMMANDS, main
from mrextrap.io import validate_latent_dict


def test_every_subcommand_takes_a_config():
    for module_name, fn_name, _ in SUBCOMMANDS.values():
        fn = getattr(importlib.import_module(module_name), fn_name)
        params = list(inspect.signature(fn).parameters.values())
        assert params[0].name == "cfg", f"{module_name}.{fn_name} must take (cfg)"
        assert all(p.default is not inspect.Parameter.empty for p in params[1:])


def test_validate_accepts_legacy_one_id_per_subject(synthetic_latent_dict):
    d = synthetic_latent_dict["train"]
    validate_latent_dict(d)  # one id per scan
    legacy = dict(d, subject_ids=[[ids[0]] for ids in d["subject_ids"]])
    validate_latent_dict(legacy)  # legacy: one id per subject
    validate_latent_dict(dict(d, subject_ids=[ids[0] for ids in d["subject_ids"]]))
    bad = dict(d, subject_ids=[ids[:2] for ids in d["subject_ids"]])
    with pytest.raises(ValueError, match="subject_ids"):
        validate_latent_dict(bad)


def test_smoke_pipeline_end_to_end(tmp_path):
    run_dir = tmp_path / "run"
    assert main(["smoke", f"paths.run_dir={run_dir}"]) == 0

    # latents and progression
    for split in ("train", "val", "test"):
        d = torch.load(run_dir / "latents" / f"{split}_latent_betas_l1.pt")
        validate_latent_dict(d, require_betas=True)
    assert set(torch.load(run_dir / "priors" / "global_params.pt")) >= {
        "global_beta_mean",
        "global_beta_std",
    }
    # priors trained (2 epochs) and evaluated
    for prior in ("unet_prior", "diffusion_prior"):
        assert (run_dir / prior / "train_log.json").is_file()
    for method in ("global", "unet", "diffusion"):
        res = json.loads((run_dir / "eval" / "single_volume" / method / "results.json").read_text())
        assert set(res) == {"hippocampus", "ventricle", "cortex", "white matter"}
        assert all(len(rows) >= 1 for rows in res.values())
    for fmt in ("earliest", "latest"):
        summary = json.loads((run_dir / "eval" / "multivolume" / fmt / "summary.json").read_text())
        assert summary["n_subjects"] >= 2
    # autoencoder metrics: default is the 5-region generalized Dice plus per-region Dice
    metrics = json.loads((run_dir / "volumes" / "test_ae_metrics.json").read_text())
    assert metrics["gdice_mode"] == "regions"
    assert set(metrics["dice_per_region"]) == {
        "hippocampus",
        "amygdala",
        "ventricle",
        "cortex",
        "white",
    }
    # figures and Table 1
    for name in ("fig2", "fig3", "fig5"):
        assert (run_dir / "figures" / f"{name}.pdf").stat().st_size > 1000
    table = pd.read_csv(run_dir / "tables" / "table1.csv", index_col=0)
    assert table.shape[0] == 4


@pytest.mark.skipif(shutil.which("git") is None or shutil.which("tar") is None, reason="git/tar")
def test_export_script_makes_fresh_single_commit_repo(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    src = tmp_path / "src"
    (src / "tools").mkdir(parents=True)
    (src / "tests").mkdir()
    shutil.copy(repo_root / "tools" / "export_public_repo.sh", src / "tools")
    shutil.copy(repo_root / "tests" / "test_no_sensitive_files.py", src / "tests")
    (src / "README.md").write_text("hello\n")
    (src / "untracked.txt").write_text("never exported\n")
    git = ["git", "-c", "user.name=T", "-c", "user.email=t@example.org"]
    subprocess.run([*git, "init", "-q"], cwd=src, check=True)
    subprocess.run([*git, "add", "README.md", "tools", "tests"], cwd=src, check=True)
    subprocess.run([*git, "commit", "-qm", "old history"], cwd=src, check=True)
    (src / "README.md").write_text("hello again\n")
    subprocess.run([*git, "commit", "-qam", "second old commit"], cwd=src, check=True)

    dest = tmp_path / "public"
    script = src / "tools" / "export_public_repo.sh"
    env = {**os.environ, "PYTHON": sys.executable, "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig")}
    run = subprocess.run(
        ["bash", str(script), str(dest), "--author", "T <t@example.org>"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    log = subprocess.run(
        ["git", "log", "--format=%s"], cwd=dest, capture_output=True, text=True, check=True
    ).stdout.split("\n")
    assert [m for m in log if m] == ["Initial public release of MRExtrap"]
    assert (dest / "README.md").read_text() == "hello again\n"
    assert not (dest / "untracked.txt").exists()
    remotes = subprocess.run(
        ["git", "remote"], cwd=dest, capture_output=True, text=True, check=True
    ).stdout
    assert remotes.strip() == ""
    assert "git push" in run.stdout  # printed as an instruction only
    again = subprocess.run(["bash", str(script), str(dest)], capture_output=True, check=False)
    assert again.returncode != 0  # refuses a non-empty destination
