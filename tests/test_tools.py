"""Tests of the cluster tools (tools/*.py) on synthetic data."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from safetensors.torch import load_file, save_file

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import _common
import convert_checkpoints
import import_legacy_artifacts as ila
import validate_against_legacy as val

from mrextrap.data.synthetic import make_synthetic_latent_dict
from mrextrap.io import save_latent_dict
from mrextrap.progression import compute_global_prior, compute_obs_var


def _sid(site: int, rid: int, bids: bool = False) -> str:
    """Synthetic ADNI-style id, built at runtime so no id-shaped literal sits in the source."""
    return f"sub-ADNI{site:03d}S{rid:04d}" if bids else f"{site:03d}_S_{rid:04d}"


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _sd(seed, prefix=""):
    g = torch.Generator().manual_seed(seed)
    return {
        f"{prefix}layer.weight": torch.randn(3, 4, generator=g),
        f"{prefix}layer.bias": torch.randn(3, generator=g),
    }


def test_convert_round_trip(tmp_path):
    ae, unet, dif = tmp_path / "ae", tmp_path / "unet", tmp_path / "dif"
    for d in (ae, unet, dif):
        d.mkdir()
    save_file(_sd(0, "_orig_mod."), str(ae / "model.safetensors"))
    save_file(_sd(1), str(unet / "model.safetensors"))
    torch.save(_sd(2, "module."), dif / "ema_denoiser.bin")
    torch.save(_sd(3), tmp_path / "synthseg_torch.pt")
    latents = make_synthetic_latent_dict(4, seq_len_range=(4, 5), latent_shape=(4, 4, 4, 4), seed=0)
    latents["betas"] = [torch.randn_like(z) for z in latents["zs"]]
    save_latent_dict(latents, tmp_path / "train.pt")
    out = tmp_path / "release"
    rc = convert_checkpoints.main([
        "--out-dir", str(out), "--ae-dir", str(ae), "--unet-dir", str(unet), "--diffusion-dir", str(dif),
        "--synthseg-weights", str(tmp_path / "synthseg_torch.pt"), "--train-latents", str(tmp_path / "train.pt"),
    ])  # fmt: skip
    assert rc == 0
    ae_out = load_file(str(out / "autoencoder" / "model.safetensors"))
    assert set(ae_out) == {"layer.weight", "layer.bias"}
    assert torch.equal(ae_out["layer.weight"], _sd(0)["layer.weight"])
    dif_out = load_file(
        str(out / "diffusion_prior/checkpoints/epoch_2000/ema_denoiser.safetensors")
    )
    assert torch.equal(dif_out["layer.bias"], _sd(2)["layer.bias"])


def test_convert_requires_input(tmp_path):
    with pytest.raises(SystemExit):
        convert_checkpoints.main(["--out-dir", str(tmp_path)])


LEGACY_PATHS_SECTION1 = [
    "exp_eval/autoencoder_older/{s}_latents/latent_betas_l1.pt",
    "exp_eval/autoencoder_older/{s}_latents/zs_ages.pt",
    "exp_eval/autoencoder_older/{s}_real_subjects",
    "exp_eval/autoencoder_older/{s}_recon_subjects",
    "exp_eval/autoencoder_older/train_latents/global_params.pt",
    "exp_eval/autoencoder_older/test_daninetdiffusion_subjects",
    "exp_eval/global_prior/single_volume/results.json",
    "exp_eval/unet_heteroscedastic/single_volume/results.json",
    "exp_eval/global_prior/varcond_earliest/results_var1.00.json",
    "exp_eval/global_prior/varcond_latest/results_var1.00.json",
    "exp_eval/global_prior/varcond_latest/temp_seg_*",
    "exp/train_autoencoder_older8/epoch_124/epoch_124",
    "synthseg_torch/synthseg_torch.pt",
    "data/{s}_df_nona.csv",
]


def test_mapping_covers_section1():
    sources = {m.source for m in ila.legacy_mappings()}
    for tmpl in LEGACY_PATHS_SECTION1:
        for s in ("train", "val", "test"):
            assert tmpl.format(s=s) in sources, tmpl
    assert any(s.startswith("exp/train_unet_mean_var") for s in sources)
    assert any(s.startswith("exp/train_beta_diffusion") for s in sources)


def test_import_links_and_reports_missing(tmp_path):
    legacy, run = tmp_path / "legacy", tmp_path / "run"
    files = [
        "exp_eval/autoencoder_older/train_latents/latent_betas_l1.pt",
        "exp_eval/global_prior/varcond_latest/results_var1.00.json",
        "data/test_df_nona.csv",
    ]
    for f in files:
        (legacy / f).parent.mkdir(parents=True, exist_ok=True)
        (legacy / f).write_text("x")
    (legacy / "exp_eval/global_prior/varcond_latest/temp_seg_A").mkdir()
    rc = ila.main(["--legacy-root", str(legacy), "--run-dir", str(run)])
    assert rc == 0
    assert (run / "latents/train_latent_betas_l1.pt").is_symlink()
    assert (run / "splits/test.csv").read_text() == "x"
    assert (run / "eval/multivolume/latest/temp_seg_A").is_symlink()
    report = json.loads((run / "import_report/import_legacy_artifacts.json").read_text())
    assert "global_params" in report["missing_required"]
    assert ila.main(["--legacy-root", str(legacy), "--run-dir", str(run), "--strict"]) == 1
    # second run keeps existing targets
    rows = ila.apply_mappings(ila.legacy_mappings(), legacy, run)
    assert {r["status"] for r in rows if r["status"] != "missing"} == {"exists"}


def test_report_guard_rejects_subject_ids(tmp_path):
    with pytest.raises(ValueError):
        _common.write_report(tmp_path, "r", {"x": _sid(2, 9001)}, "md")
    assert val.scrub(f"path/{_sid(2, 9001)}/x") == "path/<sid>/x"


def _handmade(betas, zs, ages):
    return [betas], [zs], [ages]


def test_intended_obs_var_matches_hand_reference():
    torch.manual_seed(0)
    n = 4
    z = torch.randn(n, 2, 1, 1, 1)
    b = torch.randn(n, 2, 1, 1, 1)
    age = torch.tensor([[60.0, 61.0, 62.5, 64.0]])
    res = []
    for j in range(n):
        acc = 0
        for r in range(n):
            acc = acc + ((z[j] - z[r]) - (age[0, j] - age[0, r]) * b[r]) * n / (n - 1)
        res.append(acc / n)
    expected = torch.stack(res).var(dim=0)
    got = val.intended_obs_var(*_handmade(b, z, age))
    assert torch.allclose(got, expected, atol=1e-6)
    orig = compute_obs_var(*_handmade(b, z, age))
    assert orig.shape == got.shape and not torch.allclose(orig, got)


def _run_dir(tmp_path):
    latents = make_synthetic_latent_dict(8, seq_len_range=(4, 6), latent_shape=(4, 4, 4, 4), seed=1)
    torch.manual_seed(1)
    latents["betas"] = [0.01 * torch.randn_like(z) for z in latents["zs"]]
    run = tmp_path / "run"
    save_latent_dict(latents, run / "latents" / "train_latent_betas_l1.pt")
    (run / "priors").mkdir()
    torch.save(compute_global_prior(latents), run / "priors" / "global_params.pt")
    return run


def test_validate_cpu_checks(tmp_path):
    run = _run_dir(tmp_path)
    rc = val.main(["--run-dir", str(run), "--checks", "V2,V12,Q2a,V4", "--device", "cpu"])
    assert rc == 0
    rep = json.loads((run / "validation" / "validation_report.json").read_text())
    assert rep["checks"]["V2"]["status"] == "PASS"
    assert rep["checks"]["V12"]["status"] == "INFO"
    assert rep["checks"]["V4"]["status"] == "SKIP"  # no cached eval results in this run_dir
    q2 = json.loads((run / "validation" / "report_q2_obs_var.json").read_text())
    assert "NOT the release default" in q2["label"]
    rel = q2["Q2a_obs_var"]["data"]["relative"]
    assert rel["relative_difference_of_means"] != 0
    assert (run / "validation" / "validation_report.md").read_text().count("| V2 |") == 1


def test_validate_v2_detects_change(tmp_path):
    run = _run_dir(tmp_path)
    gp = torch.load(run / "priors" / "global_params.pt")
    gp["global_beta_mean"] = gp["global_beta_mean"] + 1e-3
    torch.save(gp, run / "priors" / "global_params.pt")
    assert val.main(["--run-dir", str(run), "--checks", "V2", "--device", "cpu"]) == 1


def _adni_tables(tmp_path, n=20):
    rows, idx = [], []
    for i in range(n):
        sid = f"002_S_{i:04d}"
        for vis in ("bl", "m06", "m12"):
            rows.append(
                {
                    "PTID": sid,
                    "VISCODE": vis,
                    "AGE": 70 + i % 5,
                    "DX": ["CN", "MCI", "Dementia"][i % 3],
                }
            )
            ses = "M00" if vis == "bl" else vis.upper()
            idx.append({"filename": f"/x/sub-ADNI002S{i:04d}_ses-{ses}.nii.gz"})
    pd.DataFrame(rows).to_csv(tmp_path / "adnimerge.csv", index=False)
    pd.DataFrame(idx).to_csv(tmp_path / "index.csv", index=False)
    return tmp_path / "adnimerge.csv", tmp_path / "index.csv"


def test_validate_splits_check(tmp_path):
    from mrextrap.data.splits import build_split_csvs

    adni, index = _adni_tables(tmp_path)
    legacy = tmp_path / "legacy_data"
    for s, p in build_split_csvs(adni, index, out_dir=tmp_path / "tmp_build").items():
        legacy.mkdir(exist_ok=True)
        (legacy / f"{s}_df_nona.csv").write_text(Path(p).read_text())
    run = tmp_path / "run"
    args = ["--run-dir", str(run), "--checks", "S1", "--adnimerge-csv", str(adni), "--image-index-csv", str(index),
            "--legacy-splits-dir", str(legacy)]  # fmt: skip
    assert val.main(args) == 0
    text = (run / "validation" / "validation_report.json").read_text()
    assert not _common.SUBJECT_ID_PATTERN.search(text)
    # perturb: move one test subject into train in the "legacy" table
    test_df = pd.read_csv(legacy / "test_df_nona.csv")
    train_df = pd.read_csv(legacy / "train_df_nona.csv")
    moved = test_df[test_df.subject_id == test_df.subject_id.iloc[0]]
    pd.concat([train_df, moved]).to_csv(legacy / "train_df_nona.csv", index=False)
    test_df.drop(moved.index).to_csv(legacy / "test_df_nona.csv", index=False)
    assert val.main(args) == 1
    rep = json.loads((run / "validation" / "validation_report.json").read_text())
    assert rep["checks"]["S1"]["data"]["per_split"]["test"]["legacy_only"] == 0
    assert rep["checks"]["S1"]["data"]["per_split"]["test"]["rebuilt_only"] == 1


def test_mae_table_and_compare():
    new = {"hippocampus": [[0, 0, 1.0, 5.0, "CN", "a"], [1, 0, 2.0, 5.0, "CN", "a"]]}
    ref = {"hippocampus": [[0, 0, 1.5, 5.0, "CN", "a"], [1, 0, 2.0, 5.0, "CN", "a"]]}
    assert val.mae_table(new) == {"hippocampus": {0: 1.0, 1: 2.0}}
    cmp = val.compare_multivolume(new, ref)
    assert cmp["n_matched"] == 2 and cmp["max_abs_diff"] == 0.5


def test_select_checks():
    assert val.select_checks("cpu") == list(val.CPU_CHECKS)
    assert "Q2b" in val.select_checks("gpu,V2")
    with pytest.raises(SystemExit):
        val.select_checks("nope")
