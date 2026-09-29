# Vendored FSL-based preprocessing scripts

These three files are copied **verbatim** from the pipeline used for the MRExtrap paper
(a test checks their checksums). Do not edit them; changes belong in `../preprocess_adni.py`.

| File | Origin / attribution | Licence |
|---|---|---|
| `_preprocess.sh` | Adapted from FSL's `fsl_anat` (Mark Jenkinson, FMRIB, University of Oxford, (c) 2012); the adaptation adds optiBET and linear-only registration | FMRIB Software Library licence, header retained in the file: **non-commercial use only**; modified redistribution allowed only without financial return, with the licence imposed on the receiver and all source included |
| `optiBET.sh` | Evan Lutkenhoff (Monti Lab, UCLA, 2014), simplified by Dianne Patterson (2018); header retained. Method: Lutkenhoff et al., PLoS ONE 2014 (optiBET) | No licence text in the file; kept under its upstream terms, not MIT |
| `preprocess.sh` | Project wrapper (calls `_preprocess.sh --nosubcortseg --noseg --nononlinreg`, renames outputs to `*_reg_stripped_final.nii.gz`); derived from the same lesion-mapping prep script family | as above |

FSL itself (bet, flirt, fnirt, fast, ...) is an external requirement and is not vendored.
The files in this directory ship under their own FMRIB/optiBET terms (see the table), not under the MIT
licence of the rest of the repository.
