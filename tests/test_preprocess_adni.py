"""Tests for the reference ADNI preprocessing driver (no FSL / dcm2niix / ADNI data needed)."""

import gzip
import hashlib
import importlib.util
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest


def _sid(site: int, rid: int, bids: bool = False) -> str:
    """Synthetic ADNI-style id, built at runtime so no id-shaped literal sits in the source."""
    return f"sub-ADNI{site:03d}S{rid:04d}" if bids else f"{site:03d}_S_{rid:04d}"


ADNI_DIR = Path(__file__).resolve().parents[1] / "preprocessing" / "adni"
spec = importlib.util.spec_from_file_location("preprocess_adni", ADNI_DIR / "preprocess_adni.py")
pa = importlib.util.module_from_spec(spec)
sys.modules["preprocess_adni"] = pa
spec.loader.exec_module(pa)

# sha256 of the vendored scripts as copied from the cluster pipeline (must stay verbatim)
VERBATIM = {
    "preprocess.sh": "17c6fda297f36e1e62b50545460bb442a8a04199725ecfdcd1baf5d05414bd06",
    "_preprocess.sh": "7bbe80993ed3a1d0294e1fa1d8e469eee066eb700b17aaa378c4d30a490f0ea7",
    "optiBET.sh": "172dab9237e0f8514f972527a396d1c7eff8295d74dae4cf22f44f960c61ff7f",
}


def test_naming():
    assert pa.ptid_to_bids(_sid(2, 9001)) == _sid(2, 9001, bids=True)
    assert pa.bids_to_ptid(_sid(2, 9001, bids=True)) == _sid(2, 9001)
    assert pa.bids_to_ptid("002S9001") == _sid(2, 9001)
    with pytest.raises(ValueError):
        pa.bids_to_ptid("garbage")
    assert pa.viscode_to_session("bl") == "M00"
    assert pa.viscode_to_session("m24") == "M24"
    rel = pa.bids_t1w_relpath(_sid(2, 9001), "M00")
    assert (
        rel.as_posix()
        == f"{_sid(2, 9001, bids=True)}/ses-M00/anat/{_sid(2, 9001, bids=True)}_ses-M00_T1w.nii.gz"
    )
    assert pa.final_name("a_T1w.nii.gz") == "a_T1w_reg_stripped_final.nii.gz"
    assert pa.extract_scan_date("/x/MPRAGE/2017-05-31_13_14_01.0/I1") == "2017-05-31"
    assert pa.extract_scan_date("/x/MPRAGE/I1") is None


def test_shard_parsing(monkeypatch):
    assert pa.parse_shard(None) == (0, 1)
    assert pa.parse_shard("2/5") == (2, 5)
    with pytest.raises(ValueError):
        pa.parse_shard("5/5")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "4")
    monkeypatch.setenv("SLURM_ARRAY_TASK_MIN", "0")
    assert pa.parse_shard("slurm") == (3, 4)
    items = list(range(10))
    parts = [pa.take_shard(items, (i, 3)) for i in range(3)]
    assert sorted(x for part in parts for x in part) == items


def test_vendored_scripts_verbatim_and_flags():
    for name, digest in VERBATIM.items():
        p = ADNI_DIR / "fsl" / name
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, name
        assert p.stat().st_mode & stat.S_IXUSR
    text = (ADNI_DIR / "fsl" / "preprocess.sh").read_text()
    assert "--nosubcortseg --noseg --nononlinreg" in text
    assert "_reg_stripped_final" in text
    # licence / attribution headers kept
    assert "University of Oxford" in (ADNI_DIR / "fsl" / "_preprocess.sh").read_text()
    assert "Lutkenhoff" in (ADNI_DIR / "fsl" / "optiBET.sh").read_text()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash missing")
def test_vendored_scripts_parse():
    for name in VERBATIM:
        subprocess.run(["bash", "-n", str(ADNI_DIR / "fsl" / name)], check=True)


def test_convert_plan(tmp_path):
    d = tmp_path / "dcm" / _sid(2, 9001) / "MPRAGE" / "2011-01-01_10_00_00.0" / "I1"
    d.mkdir(parents=True)
    (d / "a.dcm").write_bytes(b"x")
    (tmp_path / "dcm" / "empty").mkdir()
    jobs = pa.plan_convert(tmp_path / "dcm", tmp_path / "nii")
    assert len(jobs) == 1
    argv = jobs[0].steps[1]
    assert argv[:4] == ["dcm2niix", "-z", "n", "-o"]
    assert argv[4] == str(tmp_path / "nii" / f"{_sid(2, 9001)}/MPRAGE/2011-01-01_10_00_00.0/I1")


def make_nifti_tree(root: Path):
    def scan(ptid, series, date, iid, name="x.nii", data=b"voxels"):
        d = root / ptid / series / f"{date}_10_00_00.0" / iid
        d.mkdir(parents=True)
        (d / name).write_bytes(data)

    scan(_sid(2, 9001), "MPRAGE", "2011-06-02", "I10")
    scan(_sid(2, 9001), "MPRAGE", "2012-06-20", "I11")
    scan(_sid(2, 9001), "Field_Mapping", "2011-06-02", "I12")  # skipped
    scan(_sid(3, 1), "MPRAGE", "2011-01-01", "I13")
    scan(_sid(999, 9999), "MPRAGE", "2011-01-01", "I14")  # not in ADNIMERGE
    merge = root.parent / "ADNIMERGE.csv"
    merge.write_text(
        "PTID,EXAMDATE,VISCODE,DX\n"
        f"{_sid(2, 9001)},2011-06-01,bl,CN\n{_sid(2, 9001)},2011-06-01,bl,CN\n"
        f"{_sid(2, 9001)},2012-06-15,m12,CN\n{_sid(3, 1)},2011-01-01,bl,MCI\n"
    )
    return merge


def test_bids_plan_and_copy(tmp_path):
    nii = tmp_path / "nii"
    merge = make_nifti_tree(nii)
    bids = tmp_path / "bids"
    jobs = pa.plan_bids(nii, merge, bids)
    assert [j.name for j in jobs] == [
        f"{_sid(2, 9001)} 2011-06-02 -> ses-M00",
        f"{_sid(2, 9001)} 2012-06-20 -> ses-M12",
        f"{_sid(3, 1)} 2011-01-01 -> ses-M00",
    ]
    assert pa.execute(jobs, dry_run=True, n_jobs=1) == 0
    assert not bids.exists()
    assert pa.execute(jobs, dry_run=False, n_jobs=2) == 0
    out = (
        bids
        / f"{_sid(2, 9001, bids=True)}/ses-M12/anat/{_sid(2, 9001, bids=True)}_ses-M12_T1w.nii.gz"
    )
    with gzip.open(out) as fh:
        assert fh.read() == b"voxels"
    only = pa.plan_bids(nii, merge, bids, subjects=[_sid(3, 1)])
    assert len(only) == 1


def test_closest_viscode():
    rows = [("2011-06-01", "bl"), ("2012-06-15", "m12")]
    assert pa.closest_viscode(rows, "2012-01-01") == "m12"
    assert pa.closest_viscode(rows, "2011-07-01") == "bl"
    assert pa.closest_viscode([], "2011-07-01") is None


def make_bids(root: Path):
    for ptid, ses in [(_sid(2, 9001), "M00"), (_sid(2, 9001), "M12"), (_sid(3, 1), "M00")]:
        p = root / pa.bids_t1w_relpath(ptid, ses)
        p.parent.mkdir(parents=True)
        p.write_bytes(b"img")


def test_preprocess_plan_command_sequence(tmp_path):
    bids, out = tmp_path / "bids", tmp_path / "out"
    make_bids(bids)
    jobs = pa.plan_preprocess(bids, out)
    assert len(jobs) == 3
    j = jobs[0]
    target = (
        out
        / f"{_sid(2, 9001, bids=True)}/ses-M00/anat/{_sid(2, 9001, bids=True)}_ses-M00_T1w.nii.gz"
    )
    assert j.steps[1] == ["bash", str(ADNI_DIR / "fsl" / "preprocess.sh"), str(target)]
    assert j.outputs == [
        target.with_name(f"{_sid(2, 9001, bids=True)}_ses-M00_T1w_reg_stripped_final.nii.gz")
    ]
    assert j.log == out / f"{_sid(2, 9001, bids=True)}/ses-M00/preprocess.log"


def fake_runner_factory(calls):
    """Stands in for FSL: creates the outputs preprocess.sh would produce."""

    def runner(argv, log, env):
        calls.append(argv)
        t1w = Path(argv[-1])
        assert t1w.read_bytes() == b"img"  # copied before the run
        stem = str(t1w)[: -len(".nii.gz")]
        Path(stem + ".anat").mkdir()
        Path(stem + "_reg_stripped_final.nii.gz").write_bytes(b"out")
        return 0

    return runner


def test_preprocess_execution_parallel_skip_and_shard(tmp_path):
    bids, out = tmp_path / "bids", tmp_path / "out"
    make_bids(bids)
    calls: list = []
    runner = fake_runner_factory(calls)
    assert pa.execute(pa.plan_preprocess(bids, out), dry_run=False, n_jobs=3, runner=runner) == 0
    assert len(calls) == 3
    # second run: everything already has its final output -> skipped
    assert pa.execute(pa.plan_preprocess(bids, out), dry_run=False, n_jobs=1, runner=runner) == 0
    assert len(calls) == 3
    # --force redoes it, wiping stale .anat first (fake runner would fail on mkdir otherwise)
    assert (
        pa.execute(
            pa.plan_preprocess(bids, out), dry_run=False, n_jobs=1, runner=runner, force=True
        )
        == 0
    )
    assert len(calls) == 6
    shards = [pa.take_shard(pa.plan_preprocess(bids, out), (i, 2)) for i in range(2)]
    assert sorted(len(s) for s in shards) == [1, 2]


def test_failure_is_reported(tmp_path):
    bids, out = tmp_path / "bids", tmp_path / "out"
    make_bids(bids)
    failed = pa.execute(
        pa.plan_preprocess(bids, out), dry_run=False, n_jobs=1, runner=lambda a, log, env: 1
    )
    assert failed == 3


def test_cli_dry_run_and_subject_file(tmp_path, capsys):
    bids, out = tmp_path / "bids", tmp_path / "out"
    make_bids(bids)
    subj = tmp_path / "subjects.txt"
    subj.write_text(f"# comment\n{_sid(3, 1)}\n")
    rc = pa.main(["preprocess", "--dry-run", "--bids-dir", str(bids), "--out-dir", str(out),
                  "--subjects", str(subj)])  # fmt: skip
    text = capsys.readouterr().out
    assert rc == 0
    assert _sid(3, 1, bids=True) in text and _sid(2, 9001, bids=True) not in text
    assert not out.exists()


def test_cli_all_dry_run_dicom(tmp_path, capsys):
    d = tmp_path / "dcm" / _sid(2, 9001) / "MPRAGE" / "2011-01-01_10_00_00.0" / "I1"
    d.mkdir(parents=True)
    (d / "a.dcm").write_bytes(b"x")
    merge = tmp_path / "m.csv"
    merge.write_text(f"PTID,EXAMDATE,VISCODE\n{_sid(2, 9001)},2011-01-01,bl\n")
    rc = pa.main(["all", "--dry-run", "--input", str(tmp_path / "dcm"), "--input-type", "dicom",
                  "--adnimerge", str(merge), "--workdir", str(tmp_path / "w")])  # fmt: skip
    assert rc == 0
    assert "dcm2niix -z n -o" in capsys.readouterr().out
    assert not (tmp_path / "w").exists()


def test_missing_tools_reporting():
    missing = pa.missing_tools(True, which=lambda t: None)
    assert "bet" in missing and "dcm2niix" in missing and "bc" in missing
    assert pa.missing_tools(False, which=lambda t: "/x") == []
