#!/usr/bin/env python3
"""Reference ADNI T1w preprocessing used for the MRExtrap paper (self-contained, stdlib only).

Pipeline (each stage can be run on its own):

1. ``convert``    DICOM -> NIfTI with ``dcm2niix`` (skipped if you already have ADNI NIfTIs).
2. ``bids``       ADNI NIfTI tree -> ``sub-ADNI<PTID>/ses-<VISCODE>/anat/*_T1w.nii.gz``; the session
                  is the ADNIMERGE ``VISCODE`` whose ``EXAMDATE`` is closest to the scan date.
3. ``preprocess`` per scan, the vendored FSL scripts (``fsl/preprocess.sh`` -> ``fsl/_preprocess.sh``
                  with ``--nosubcortseg --noseg --nononlinreg``, which calls ``fsl/optiBET.sh``):
                  reorient, crop (robustfov), bias-correct (FAST), optiBET brain extraction and
                  6-dof FLIRT to MNI152 1 mm. The model input is ``*_T1w_reg_stripped_final.nii.gz``.
``all``          runs the three stages in order.

Local parallelism: ``-j N`` (thread pool around subprocesses, equivalent to ``xargs -P N``).
Cluster arrays: ``--shard I/N`` (or SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT via ``--shard slurm``).
``--dry-run`` prints every command and touches nothing. FSL, dcm2niix, bc are external requirements.

No subject data is bundled with this repository; ADNI data must be obtained from adni.loni.usc.edu.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
FSL_SCRIPT_DIR = HERE / "fsl"
FINAL_SUFFIX = "_reg_stripped_final.nii.gz"  # replaces ".nii.gz" of the T1w file name
SKIP_DIR_KEYWORDS = ("Calibration", "Field_Mapping")  # as in the original BIDS conversion

# FSL commands used by fsl/preprocess.sh, fsl/_preprocess.sh and fsl/optiBET.sh.
FSL_TOOLS = (
    "fslmaths", "fslstats", "fslreorient2std", "robustfov", "convert_xfm", "bet", "fast",
    "fslsmoothfill", "flirt", "fnirt", "applywarp", "invwarp", "immv", "imcp", "imrm",
    "remove_ext", "imglob", "imtest",
)  # fmt: skip
SYSTEM_TOOLS = ("bc", "realpath")


# ----------------------------------------------------------------------------- naming helpers
def ptid_to_bids(ptid: str) -> str:
    """'<SITE>_S_<RID>' -> 'sub-ADNI<SITE>S<RID>'."""
    return "sub-ADNI" + ptid.replace("_", "")


def bids_to_ptid(name: str) -> str:
    """'sub-ADNI<SITE>S<RID>' (or 'ADNI<SITE>S<RID>' or '<SITE>_S_<RID>') -> '<SITE>_S_<RID>'."""
    core = name.removeprefix("sub-").removeprefix("ADNI")
    core = core.replace("_", "")
    if len(core) != 8 or core[3] != "S":
        raise ValueError(f"cannot interpret ADNI subject id: {name!r}")
    return f"{core[:3]}_S_{core[4:]}"


def viscode_to_session(viscode: str) -> str:
    """ADNIMERGE VISCODE -> BIDS session label ('bl' -> 'M00', 'm06' -> 'M06')."""
    return viscode.upper().replace("BL", "M00")


def bids_t1w_name(ptid: str, session: str) -> str:
    sub = ptid_to_bids(ptid)
    return f"{sub}_ses-{session}_T1w.nii.gz"


def bids_t1w_relpath(ptid: str, session: str) -> Path:
    return Path(ptid_to_bids(ptid)) / f"ses-{session}" / "anat" / bids_t1w_name(ptid, session)


def final_name(t1w_name: str) -> str:
    assert t1w_name.endswith(".nii.gz")
    return t1w_name[: -len(".nii.gz")] + FINAL_SUFFIX


def extract_scan_date(path: str) -> str | None:
    """Return 'YYYY-MM-DD' from a path component like '2017-05-31_13_14_01.0' (original test)."""
    for part in Path(path).parts:
        if (
            len(part) > 10
            and part[4] == "-"
            and part[7] == "-"
            and part[10] == "_"
            and part[13] == "_"
        ):
            return part[:10]
    return None


def parse_shard(spec: str | None) -> tuple[int, int]:
    """'I/N' -> (I, N); 'slurm' reads SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT; None -> (0, 1)."""
    if spec is None:
        return 0, 1
    if spec == "slurm":
        i = int(os.environ["SLURM_ARRAY_TASK_ID"])
        n = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
        min_id = int(os.environ.get("SLURM_ARRAY_TASK_MIN", "0"))
        i -= min_id
    else:
        a, b = spec.split("/")
        i, n = int(a), int(b)
    if not (n >= 1 and 0 <= i < n):
        raise ValueError(f"invalid shard {spec!r}: need 0 <= I < N")
    return i, n


def take_shard(items: Sequence, shard: tuple[int, int]) -> list:
    i, n = shard
    return [x for k, x in enumerate(items) if k % n == i]


# ----------------------------------------------------------------------------- job model
@dataclass
class Job:
    """One unit of work: an ordered list of shell steps (argv lists or python callables)."""

    name: str
    steps: list[list[str] | Callable[[], None]] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    log: Path | None = None
    skip_if_exists: Path | None = None

    def describe(self) -> list[str]:
        out = []
        for s in self.steps:
            out.append(" ".join(map(str, s)) if isinstance(s, list) else f"<python: {s.__name__}>")
        if self.log:
            out.append(f"(stdout+stderr -> {self.log})")
        return out


Runner = Callable[[list[str], Path | None, dict], int]


def default_runner(argv: list[str], log: Path | None, env: dict) -> int:
    if log is None:
        return subprocess.run(argv, env=env, check=False).returncode
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as fh:
        return subprocess.run(
            argv, env=env, stdout=fh, stderr=subprocess.STDOUT, check=False
        ).returncode


def run_job(job: Job, runner: Runner, env: dict) -> tuple[str, bool, str]:
    for step in job.steps:
        if callable(step):
            step()
            continue
        rc = runner(step, job.log, env)
        if rc != 0:
            return job.name, False, f"command failed (exit {rc}): {' '.join(step)}"
    missing = [str(p) for p in job.outputs if not p.exists()]
    if missing:
        return job.name, False, f"expected output missing: {missing}"
    return job.name, True, ""


def execute(
    jobs: Iterable[Job], *, dry_run: bool, n_jobs: int, runner: Runner = default_runner,
    env: dict | None = None, force: bool = False,
) -> int:  # fmt: skip
    """Run jobs (thread pool = local parallelism). Returns number of failed jobs."""
    env = dict(os.environ if env is None else env)
    jobs = list(jobs)
    todo = []
    for j in jobs:
        if not force and j.skip_if_exists is not None and j.skip_if_exists.exists():
            print(f"[skip] {j.name}: {j.skip_if_exists} exists")
            continue
        todo.append(j)
    if dry_run:
        for j in todo:
            print(f"[dry-run] {j.name}")
            for line in j.describe():
                print(f"    {line}")
        print(f"[dry-run] {len(todo)} job(s) would run, {len(jobs) - len(todo)} skipped")
        return 0
    failed = 0
    with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as pool:
        for name, ok, msg in pool.map(lambda j: run_job(j, runner, env), todo):
            print(f"[{'ok' if ok else 'FAIL'}] {name} {msg}")
            failed += not ok
    print(f"done: {len(todo) - failed} ok, {failed} failed, {len(jobs) - len(todo)} skipped")
    return failed


# ----------------------------------------------------------------------------- stage 1: DICOM
def plan_convert(dicom_dir: Path, nifti_dir: Path, dcm2niix: str = "dcm2niix") -> list[Job]:
    """One dcm2niix call per directory holding *.dcm (mirrors the input tree)."""
    jobs = []
    for dirpath, dirnames, filenames in os.walk(dicom_dir):
        dirnames.sort()
        if not any(f.endswith(".dcm") for f in filenames):
            continue
        out = nifti_dir / Path(dirpath).relative_to(dicom_dir)
        # NOTE(paper): the cluster call was plain `dcm2niix -o OUT DIR` and produced uncompressed
        # *.nii files named <folder>_<series>_<time>_<n>.nii; `-z n` pins that behaviour.
        argv = [dcm2niix, "-z", "n", "-o", str(out), dirpath]
        jobs.append(Job(name=str(Path(dirpath).relative_to(dicom_dir)),
                        steps=[_mkdir(out), argv]))  # fmt: skip
    return jobs


def _mkdir(path: Path) -> Callable[[], None]:
    def mkdir() -> None:
        path.mkdir(parents=True, exist_ok=True)

    return mkdir


# ----------------------------------------------------------------------------- stage 2: BIDS
def read_adnimerge(path: Path) -> dict[str, list[tuple[str, str]]]:
    """PTID -> [(EXAMDATE, VISCODE), ...] in file order (only these three columns are used)."""
    rows: dict[str, list[tuple[str, str]]] = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("PTID") and r.get("EXAMDATE") and r.get("VISCODE"):
                rows.setdefault(r["PTID"], []).append((r["EXAMDATE"], r["VISCODE"]))
    return rows


def closest_viscode(rows: Sequence[tuple[str, str]], scan_date: str) -> str | None:
    """VISCODE (first row) of the ADNIMERGE EXAMDATE closest to scan_date; ties -> file order."""
    if not rows:
        return None
    day = datetime.fromisoformat(scan_date)
    best = min(rows, key=lambda r: abs(datetime.fromisoformat(r[0]) - day))[0]
    return next(v for d, v in rows if d == best)


def find_nifti_scans(subject_dir: Path) -> dict[str, list[Path]]:
    """date -> NIfTI files below subject_dir (dcm2niix output), skipping calibration/field maps."""
    by_date: dict[str, list[Path]] = {}
    for root, dirnames, files in os.walk(subject_dir):
        dirnames.sort()
        if any(k in root for k in SKIP_DIR_KEYWORDS):
            continue
        for f in sorted(files):
            if f.endswith((".nii", ".nii.gz")):
                date = extract_scan_date(root)
                if date is not None:
                    by_date.setdefault(date, []).append(Path(root) / f)
    return by_date


def plan_bids(
    nifti_dir: Path, adnimerge: Path, bids_dir: Path, subjects: Sequence[str] | None = None
) -> list[Job]:
    """One copy job per (subject, scan date).

    Selection rule = effective behaviour of the original notebook-style script: for every scan
    date the FIRST NIfTI found is used (its ranking by series description / field strength was a
    no-op: the sort key was constant, and the +-2-month ADNIMERGE date filter was only printed).
    Several dates mapping to the same session overwrite each other, later date wins.
    The original used os.walk order (filesystem dependent); here dirs and files are visited in
    sorted order, which is deterministic but may differ on multi-series dates.
    """
    merge = read_adnimerge(adnimerge)
    wanted = None if subjects is None else {bids_to_ptid(s) for s in subjects}
    jobs = []
    for ptid in sorted(merge):
        if wanted is not None and ptid not in wanted:
            continue
        sdir = nifti_dir / ptid
        if not sdir.is_dir():
            continue
        for date, files in sorted(find_nifti_scans(sdir).items()):
            viscode = closest_viscode(merge[ptid], date)
            if viscode is None:
                continue
            session = viscode_to_session(viscode)
            src = files[0]
            dest = bids_dir / bids_t1w_relpath(ptid, session)
            jobs.append(Job(name=f"{ptid} {date} -> ses-{session}",
                            steps=[_copy_as_gz(src, dest)], outputs=[dest]))  # fmt: skip
    return jobs


def _copy_as_gz(src: Path, dest: Path) -> Callable[[], None]:
    import gzip

    def copy_scan() -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".gz":
            shutil.copyfile(src, dest)
        else:
            # NOTE(paper): the original copied the raw .nii to a file *named* .nii.gz without
            # compressing it (FSL reads both); we compress properly, voxel data are identical.
            with open(src, "rb") as fi, gzip.open(dest, "wb") as fo:
                shutil.copyfileobj(fi, fo)

    return copy_scan


# ----------------------------------------------------------------------------- stage 3: FSL
def find_bids_scans(bids_dir: Path, subjects: Sequence[str] | None = None) -> list[Path]:
    wanted = None if subjects is None else {ptid_to_bids(bids_to_ptid(s)) for s in subjects}
    out = []
    for sub in sorted(p for p in bids_dir.glob("sub-*") if p.is_dir()):
        if wanted is not None and sub.name not in wanted:
            continue
        out.extend(sorted(sub.glob("ses-*/anat/*_T1w.nii.gz")))
    return out


def plan_preprocess(
    bids_dir: Path, out_dir: Path, subjects: Sequence[str] | None = None,
    script: Path | None = None,
) -> list[Job]:  # fmt: skip
    """One job per T1w: copy into out_dir (same tree), wipe stale results, run fsl/preprocess.sh.

    The log goes to <out_dir>/sub/ses/preprocess.log as in the original driver.
    Outputs (next to the copy): *_reg[_scaled][_stripped][_scaled]_final.nii.gz and a
    *_T1w.anat directory; *_T1w_reg_stripped_final.nii.gz is the model input.
    """
    script = script or FSL_SCRIPT_DIR / "preprocess.sh"
    jobs = []
    for src in find_bids_scans(bids_dir, subjects):
        rel = src.relative_to(bids_dir)
        dest = out_dir / rel
        final = dest.with_name(final_name(dest.name))
        stem = str(dest)[: -len(".nii.gz")]

        def prepare(src=src, dest=dest, stem=stem) -> None:
            shutil.rmtree(stem + ".anat", ignore_errors=True)
            for p in dest.parent.glob(Path(stem).name + "_*"):
                p.unlink() if p.is_file() else shutil.rmtree(p)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)

        jobs.append(Job(
            name=str(rel), steps=[prepare, ["bash", str(script), str(dest)]],
            outputs=[final], log=dest.parent.parent / "preprocess.log", skip_if_exists=final,
        ))  # fmt: skip
    return jobs


# ----------------------------------------------------------------------------- requirements
def missing_tools(need_dcm2niix: bool, which=shutil.which) -> list[str]:
    tools = list(FSL_TOOLS) + list(SYSTEM_TOOLS) + (["dcm2niix"] if need_dcm2niix else [])
    return [t for t in tools if which(t) is None]


def check_requirements(need_dcm2niix: bool, strict: bool) -> bool:
    problems = []
    if not os.environ.get("FSLDIR"):
        problems.append("FSLDIR is not set (source $FSLDIR/etc/fslconf/fsl.sh)")
    problems += [f"tool not on PATH: {t}" for t in missing_tools(need_dcm2niix)]
    for p in problems:
        print(("ERROR: " if strict else "WARNING: ") + p, file=sys.stderr)
    return not problems


# ----------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dry-run", action="store_true", help="print commands, change nothing")
    common.add_argument("-j", "--jobs", type=int, default=1, help="local parallel workers")
    common.add_argument(
        "--shard", default=None, help="'I/N' or 'slurm' (array task): process every N-th job"
    )
    common.add_argument("--subjects", type=Path, help="text file with one ADNI subject id per line")
    common.add_argument("--force", action="store_true", help="redo jobs whose output exists")

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], parents=[common])
    sub = p.add_subparsers(dest="stage", required=True)

    sub.add_parser("check", help="check external requirements (FSL, dcm2niix, bc)")
    c = sub.add_parser("convert", parents=[common], help="DICOM -> NIfTI (dcm2niix)")
    c.add_argument("--dicom-dir", type=Path, required=True)
    c.add_argument("--nifti-dir", type=Path, required=True)
    b = sub.add_parser("bids", parents=[common], help="NIfTI -> BIDS layout")
    b.add_argument("--nifti-dir", type=Path, required=True)
    b.add_argument(
        "--adnimerge", type=Path, required=True, help="ADNIMERGE.csv (PTID, EXAMDATE, VISCODE)"
    )
    b.add_argument("--bids-dir", type=Path, required=True)
    f = sub.add_parser(
        "preprocess", parents=[common], help="FSL/optiBET preprocessing of a BIDS tree"
    )
    f.add_argument("--bids-dir", type=Path, required=True)
    f.add_argument("--out-dir", type=Path, required=True)
    a = sub.add_parser("all", parents=[common], help="convert (if DICOM), bids, preprocess")
    a.add_argument(
        "--input",
        type=Path,
        required=True,
        help="ADNI download root (<PTID>/<series>/<date>/<I..>/)",
    )
    a.add_argument("--input-type", choices=["dicom", "nifti"], required=True)
    a.add_argument("--adnimerge", type=Path, required=True)
    a.add_argument(
        "--workdir", type=Path, required=True, help="creates nifti/, bids/, preprocessed/ below"
    )
    return p


def read_subjects(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    lines = [ln.split("#")[0].strip() for ln in path.read_text().splitlines()]
    return [ln for ln in lines if ln]


def main(argv: Sequence[str] | None = None, runner: Runner = default_runner) -> int:
    args = build_parser().parse_args(argv)
    shard = parse_shard(args.shard) if args.stage != "check" else (0, 1)
    if args.stage == "check":
        return 0 if check_requirements(need_dcm2niix=True, strict=True) else 1
    subjects = read_subjects(args.subjects)
    os.environ.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")  # FSL default output type

    stages: list[tuple[str, list[Job]]] = []
    if args.stage == "all":
        work = args.workdir
        nifti = work / "nifti" if args.input_type == "dicom" else args.input
        if args.input_type == "dicom":
            stages.append(("convert", plan_convert(args.input, nifti)))
        stages.append(("bids", plan_bids(nifti, args.adnimerge, work / "bids", subjects)))
        stages.append(("preprocess", None))  # planned lazily: needs the BIDS tree to exist
        bids_dir, out_dir = work / "bids", work / "preprocessed"
    else:
        bids_dir = getattr(args, "bids_dir", None)
        out_dir = getattr(args, "out_dir", None)
        if args.stage == "convert":
            stages.append(("convert", plan_convert(args.dicom_dir, args.nifti_dir)))
        elif args.stage == "bids":
            stages.append(
                ("bids", plan_bids(args.nifti_dir, args.adnimerge, args.bids_dir, subjects))
            )
        else:
            stages.append(("preprocess", None))

    need_dcm = any(name == "convert" for name, _ in stages)
    if not check_requirements(need_dcm2niix=need_dcm, strict=not args.dry_run) and not args.dry_run:
        return 2
    failed = 0
    for name, jobs in stages:
        if jobs is None:
            jobs = plan_preprocess(bids_dir, out_dir, subjects)
            if args.dry_run and not bids_dir.exists():
                print("[dry-run] BIDS tree does not exist yet; preprocess commands not listed")
        print(f"== stage: {name} ({len(jobs)} jobs before sharding)")
        jobs = take_shard(jobs, shard)
        # BIDS copies are cheap and can clash between shards on shared sessions, so run them serially
        n_jobs = 1 if name == "bids" else args.jobs
        failed += execute(
            jobs, dry_run=args.dry_run, n_jobs=n_jobs, runner=runner, force=args.force
        )
        if failed:
            break
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
