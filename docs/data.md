# Data access and split files

MRExtrap was developed on the Alzheimer's Disease Neuroimaging Initiative (ADNI). **This repository ships no
ADNI data, no subject identifiers and no derived per-subject tables.** Everything below is generated locally
from data you download under your own ADNI access.

## 1. Get ADNI access

1. Apply at https://adni.loni.usc.edu (Data and Samples, Access Data). Accept the Data Use Agreement.
2. Download from the LONI Image and Data Archive (https://ida.loni.usc.edu):
   - T1-weighted MPRAGE scans of the ADNI1, ADNI-GO, ADNI2 and ADNI3 collections (DICOM or NIfTI). The
     paper used 9,200 scans from about 1,700 subjects.
   - `ADNIMERGE.csv` (study data, "ADNIMERGE" package) for ages, visit codes and diagnoses.

`preprocessing/adni/preprocess_adni.py` selects T1 MPRAGE scans with the filters described in
[preprocessing.md](preprocessing.md); it makes no further claim about the image selection.

## 2. Preprocess

Follow [preprocessing.md](preprocessing.md). The output is one skull-stripped, MNI152-registered 1 mm volume
per scan, cropped by the dataset code to 160x192x160.

## 3. Build the split CSVs

The paper splits by subject: 1411 training, 32 validation, 178 test subjects. The code in
`src/mrextrap/data/splits.py` (`mrextrap make-splits`) builds `run_dir/splits/{train,val,test}.csv` from
`ADNIMERGE.csv` and an index of your preprocessed files (`image_index_csv`, one row per scan with the
`filename` column; `subject_id` and `session_id` are parsed from BIDS-style names such as
`sub-ADNI<SITE>S<RID>_ses-M06` if not given). Subjects are shuffled with seed 42 and split 90 % / 3 % / 7 %, or
you pass explicit test/validation subject lists
(`splits.test_subjects` / `splits.val_subjects`: lists, or paths of text files with one id per line).
Example: `mrextrap make-splits paths.run_dir=/data/run splits.adnimerge_csv=ADNIMERGE.csv splits.image_index_csv=index.csv`.

Output columns: `subject_id, session_id, filename, AGE, DX, DX_filled, age_from_baseline, baseline_age,
AGE_filled`.

Diagnosis label per subject (used in Fig. 5): dementia if any visit is Dementia, else MCI if any visit is
MCI, else CN.

## 4. Test set and the DANINet pairs

The paper uses the same test subjects and the same (baseline scan, target age) pairs as DANINet
(reference "daninet" in the paper): Ravi D, Blumberg SB, Ingala S, Barkhof F, Alexander DC, Oxtoby NP.
"Degenerative adversarial neuroimage nets for brain scan simulations: Application in ageing and dementia."
Medical Image Analysis 75 (2022) 102257. doi:10.1016/j.media.2021.102257. We do not
redistribute them. Obtain the list of test subjects and pairs from the DANINet authors or their public
repository, and write them to a CSV that is read by `mrextrap eval-single`
(`eval.daninet_pairs`, default `run_dir/daninet_pairs.csv`). Two row formats are accepted (an optional
header row is skipped):

```
<subject_id>, <baseline_age>, <target_age>
<baseline_age>_..._ADNI_<subject_id>_..., <target_age>      # DANINet file-name format
```

Pass the same subjects as the test list when building the splits so that the test split contains them.
The split you build yourself will not be identical to the paper split unless you supply the same subject
lists; results then differ from the paper by sampling variation of the train/validation split. The paper
train CSV has more subjects than the 1411 quoted in the text (the code only uses subjects with two or more
scans); this is not resolved here. Subject-level split lists are not distributed (ADNI terms); the split
procedure is deterministic (seed 42) given the same inputs.

## 5. What ends up where

Nothing in `run_dir` should be committed. `.gitignore` excludes `run_dir/`, images, tables with subject rows
and weights.
