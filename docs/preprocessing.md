# ADNI preprocessing (reference pipeline)

`preprocessing/adni/preprocess_adni.py` reproduces the preprocessing that produced the MRExtrap paper
inputs from an ADNI download. It is a standalone, stdlib-only script (Python >= 3.10) that drives the
vendored FSL shell scripts in `preprocessing/adni/fsl/`. **No ADNI data are included**; obtain them from
adni.loni.usc.edu (you need ADNI access and `ADNIMERGE.csv`).

## Steps

1. **convert** (DICOM only): `dcm2niix -z n -o <out> <dir>` for every directory with `*.dcm`, mirroring the
   ADNI tree `<PTID>/<series>/<date_time>/<I imageid>/`.
2. **bids**: for each subject in `ADNIMERGE.csv` and each scan date, take the first NIfTI (calibration and
   `Field_Mapping` series skipped), find the ADNIMERGE `EXAMDATE` closest to the scan date and use its
   `VISCODE` (`bl` -> `M00`) as session: `sub-ADNI<SITE>S<RID>/ses-M00/anat/sub-ADNI<SITE>S<RID>_ses-M00_T1w.nii.gz`.
3. **preprocess** (per scan, `fsl/preprocess.sh` -> `fsl/_preprocess.sh --nosubcortseg --noseg --nononlinreg`):
   ensure non-negative range, `fslreorient2std`, `robustfov`, FAST bias correction (`--weakbias`: 10 iterations,
   smoothing 20), `optiBET.sh` on the bias-corrected image (BET `-f 0.1 -B`, 2 mm FLIRT + FNIRT to MNI152 2 mm,
   back-projected MNI 1 mm brain mask), then 6-dof FLIRT (and 7-dof "scaled" variants) to
   `MNI152_T1_1mm_brain` (stripped) / `MNI152_T1_1mm` (unstripped). Final files are converted to float:
   `*_T1w_reg_stripped_final.nii.gz` is **the model input**; `*_reg_final`, `*_reg_scaled_final`,
   `*_reg_stripped_scaled_final` are by-products; a `*_T1w.anat/` directory keeps intermediates.
4. (outside this script) SynthSeg (robust) was run on `*_reg_stripped_final.nii.gz` to obtain the
   segmentations used for the volume analyses; see the segmentation module of this repository.

## Requirements

- **FSL** (bet, flirt, fnirt, fast, robustfov, fslmaths, fslstats, fslreorient2std, convert_xfm, applywarp,
  invwarp, fslsmoothfill, immv/imcp/imrm/remove_ext/imglob/imtest) with `FSLDIR` set and
  `FSLOUTPUTTYPE=NIFTI_GZ` (source `$FSLDIR/etc/fslconf/fsl.sh`). The FSL version used for the paper was not
  recorded; the scripts were written for FSL 6.
  AFNI is **not** required (only mentioned in the optiBET header; the BET path is the one used).
- `dcm2niix` (only for DICOM input), `bc`, `realpath` (coreutils), bash.
- FSL licence is non-commercial; see `preprocessing/adni/fsl/NOTICE.md`.

`python preprocessing/adni/preprocess_adni.py check` verifies the tools.

## Usage

```bash
S=preprocessing/adni/preprocess_adni.py
# everything, from raw DICOM, 8 local workers (dry run first: prints every command, changes nothing)
python $S all --input /data/ADNI --input-type dicom --adnimerge ADNIMERGE.csv --workdir /data/work --dry-run
python $S all --input /data/ADNI --input-type dicom --adnimerge ADNIMERGE.csv --workdir /data/work -j 8
# stage by stage (NIfTI already available)
python $S bids --nifti-dir /data/ADNI_NII --adnimerge ADNIMERGE.csv --bids-dir /data/bids
python $S preprocess --bids-dir /data/bids --out-dir /data/preprocessed -j 8 --subjects subjects.txt
```

`--subjects` is a text file with one ADNI id per line (`<SITE>_S_<RID>` or `sub-ADNI<SITE>S<RID>`; `#` comments).
Finished scans (final file present) are skipped; `--force` redoes them; failed scans are reported and give a
non-zero exit code. Logs: `<out>/sub-*/ses-*/preprocess.log`.
Cluster: `--shard slurm` splits scans over a SLURM array, see `preprocessing/adni/slurm_preprocess_array.sbatch.example`.
`-j N` is a thread pool around subprocesses (like `xargs -P N`); the original used Ray with 8 CPUs.

## Differences from the original cluster scripts (all intentional, listed for transparency)

- Ray/SLURM driver and hard-coded paths replaced by `-j` / `--shard`; ADNI-specific paths removed.
- The original BIDS conversion had a scan-ranking step (series description, field strength) and a +-2 month
  ADNIMERGE date filter that had **no effect** on the selection (constant sort key; filtered dict unused).
  The effective rule, first NIfTI per date, is implemented explicitly. The original script used filesystem order;
  this version sorts. This only matters if a subject has several series on the same date.
- Raw `.nii` was copied to a name ending `.nii.gz` without compression; here it is gzip-compressed
  (identical voxel data).
- Return codes and final outputs are checked; stale partial `.anat` directories are wiped before a re-run.
- `FSLOUTPUTTYPE=NIFTI_GZ` is set if unset (the default of the FSL setup; the value used for the paper was not recorded).
