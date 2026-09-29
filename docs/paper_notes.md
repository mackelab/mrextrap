# Notes on differences between code and paper text

This release ports the research code without changing numerics. The list below records places where the
implementation does something other than, or more specific than, what the paper text says or implies, so that
readers can interpret the code and the reported numbers. Each item is marked `NOTE(paper)` in the source. For
each, the released default reproduces the paper's numbers; where an alternative exists it is an explicit
opt-in. The Q-numbers are
the identifiers used throughout the repository.

Items that concern how numbers in the paper were produced (Q1, Q2, Q6, Q7, Q8, GDice) are listed first.

## Evaluation and reported numbers

**Q1. Prediction horizon in the single-volume Global and UNet evaluation.**
The predicted latent is `z0 + beta * dt`. For the Global and UNet priors the code uses `dt` = age of the
target scan since the subject's *first* scan, not `target - baseline`. The two are equal when the DANINet
baseline scan is the subject's first scan. The diffusion path uses `ages[target] - ages[baseline]`.
Code: `evaluation/single_volume.py`, `prediction.py`. Default `eval.delta_t_from_first_scan: true`
reproduces the paper; `false` uses `target - baseline`. The evaluation logs how many subjects have a baseline
that is not their first scan (`count_baseline_not_first`); `tools/validate_against_legacy.py` check V5a
reports the same count.

**Q2. Observation variance (`global_obs_var`).**
The paper describes it as the diagonal covariance of the residuals `z_j - (z + beta (a_j - a))` over
subjects and time points. In the code (`progression.py::compute_residuals`, `compute_obs_var`): only
subjects with at least 4 scans are used; each residual is scaled by N/(N-1); for every reference scan the
residual is computed against all scans of that subject, averaged over the reference scans, and then the
unbiased variance is taken over subjects (the mean is not subtracted separately). *Broadcast:* the term
`zs - intercept` has shape (1, N, C, D, H, W) after unsqueeze and is broadcast against `age_diff * beta` of
shape (N, 1, C, D, H, W), so the residual is an (N, N, C, D, H, W) tensor before averaging over the first
axis. This is kept exactly as in the original; the result is a per-voxel variance of shape (C, D, H, W).
In the multi-volume evaluation this variance is used as is; `obs_var_scale` (result file name
`results_var1.00.json`) only names the output file.

**Q6. Fig. 2b/c interpolation.**
The paper describes interpolation between latent codes. The research code interpolated in *voxel space* and
segmented the interpolated inputs (the autoencoder reconstruction was computed but not used), and used an
older fine-tuned autoencoder checkpoint. `evaluation/interpolation.py` implements both `mode="latent"`
(paper description, default) and `mode="voxel"` (old code). The paper autoencoder is `older8` (epoch 124);
Fig. 2 was produced with the paper autoencoder (older8, epoch 124) in latent interpolation mode
(`interp.mode=latent`).

**GDice (generalized Dice, Sec. 3.1, reported 0.92).**
In the notebook that produced the number, the 5 regional masks are extracted and then passed to a one-hot
step that reads only the first channel (hippocampus). The score is therefore a two-class (hippocampus versus
rest) generalized Dice rather than a five-region score, and it was computed with a different autoencoder
checkpoint than `older8`. `evaluation/autoencoder.py::generalized_dice` defaults to `mode="regions"`, the correct five-region score:
the five binary masks (hippocampus, amygdala, ventricle, cortex, white matter) are combined into one label map
with values 1 to 5 (background 0), one-hot encoded with six classes and scored with monai's
`GeneralizedDiceScore(include_background=False)`. `eval-ae` also reports the per-region Dice
(`dice_per_region`). `mode="notebook"` (`ae_eval.gdice_mode=notebook`) keeps the hippocampus-channel-only logic
and exists only to reproduce the number printed in the preprint (0.92). The release reports the corrected 5-region generalized Dice of the paper autoencoder (older8) on the 92
validation scans: 0.929, per region 0.924 / 0.929 / 0.965 / 0.928 / 0.945, SSIM 0.947. The preprint's
0.917 / 0.92 used only the hippocampus channel and an earlier autoencoder (acc_ft, epoch 108);
`mode="notebook"` reproduces it.

**Q7. DANINet reference row.**
The row is not recomputed: it is transcribed from Table 2 ("Proposed") of Ravi et al. (2022, Med. Image
Anal. 75:102257), mean absolute error in %TBV (`evaluation/stats.py`, `DANINET_REFERENCE`). Hippocampus =
left + right (0.029 + 0.031 = 0.060; SD sqrt(0.028^2 + 0.031^2) = 0.042; the 0.059 in one notebook is not
used), ventricles 0.257 (SD 0.222), grey matter and white matter both 0.829 (SD 0.612), as printed in that
paper. The +- values in Table 1 are SD / sqrt(178), i.e. computed with our test-set size rather than
DANINet's. The z-tests against DANINet use these constants; the code that produced the DANINet versus Global
z-test in the paper was not found, `stats.z_test_vs_reported` implements it.

**Q8. Fig. 3 subjects and baselines.**
The caption says 32 subjects, the text 24. Both cached result sets (earliest and latest baseline) contain 32
subjects, so 32 is correct. Panel b uses the results with the *earliest* baseline scan,
panels a and c the *latest* (`eval.baseline_formats`, `figures/fig3_multivol.py`). The earliest-baseline
results reproduce the correlations 0.33, 0.73, 0.68, 0.71 of panel b.

**Q9. Regional volumes.**
Regions are aggregated by substring matching on SynthSeg CSV column names: "cortex" = cerebral + cerebellar
cortex, "white matter" = cerebral + cerebellar white matter, "ventricle" = lateral + inferior lateral + 3rd +
4th ventricle. Total brain volume (TBV) is the sum of all label columns; `%TBV` is relative to the TBV of the
subject's first real scan. `vols.csv` has two header rows (names, label ids); the reader drops the second
(`segmentation/volumes.py::read_vols_csv`). Segmentation uses the SynthSeg robust mode.

**Q14. Diffusion single-volume evaluation.** K = 5 samples with generator seed 0 for every subject, the
mean of the samples is used, bf16 autocast on GPU (`prediction.py`).

**Q15. Fig. 5.** Panel a plots the first half of the subjects (by list order), with the beta at the sequence
midpoint and per-channel whitening. Panel b uses beta at the first scan (`abs().sum()`, then log10). Panel c
bins by the *mean* age over each series (the paper text says age at the first scan), bins `arange(57.5, 94.5,
5)`, mean +- SEM. The betas come from the training set (`figures/fig5_beta.py`).

## Training details

**Q3. Autoencoder warm-up.** The KL and adversarial weights ramp with the global step since the start of
training. The adversarial term starts at epoch 25 (`0.2 * num_epochs`), at which point the ramp is already at
its final value. The paper says the weight is increased linearly over one epoch after epoch 25.

**Q4. Checkpoint labels.** Prior checkpoints named `epoch_2000` hold the weights after 1990 epochs
(`training/unet_prior.py::legacy_epoch_label`; saved every 10 epochs; the final epoch is labelled
`epoch_2020`). Evaluation loads `epoch_2000`. The autoencoder is saved after every epoch under
`epoch_{epoch//2*2+2}`.

**Q5. Diffusion prior optimizer.** AdamW betas are (0.95, 0.999) in the code, (0.9, 0.999) in the paper text.
The UNet prior uses (0.95, 0.999) as in the paper. Cosine schedules use a horizon of 1.3 times the number of
steps, so the learning rate ends at about 20 % of its peak (this is in the code only).

**Q10. Progression-rate fit (`progression.py::TorchLinearRegression`).** `beta` is fitted by gradient descent
(Adam, learning rate 3e-3, 1000 iterations, L1 loss), one regression per reference scan, with a random linear
initialisation seeded once globally (seed 42). The tolerance is not updated (`tol * 10` is assigned to a local
variable). The `latent_betas_l1.pt` files are produced with this L1 fit. The legacy diffusion-prior config also
named val/test `latent_betas.pt` files from an earlier non-L1 (MSE) fit, but the training loop never
iterates the val or test loaders and checkpoints are saved on a fixed 20-epoch schedule, so these files
had no effect on the trained model. The release config points val/test at the `_l1` files instead.

**Q11. Mixed precision.** `vae.encode` and `vae.decode` run in fp32 even when the accelerator is set to bf16
(autocast covers only `forward`). Reconstruction evaluation (`vae(x)`) and the diffusion denoiser forward run
under bf16 autocast.

**Q12. UNet augmentation.** Random 90-degree rotations over two of the three spatial axes change the array
shape (20x24x20 becomes 24x20x20); the fully convolutional UNet accepts this.

**Q13. Latent datasets.** `subject_ids` are filtered for NaN sequences only in the Bayesian variant (not
released). Standardisation uses channel-wise mean and unbiased std over all scans and voxels of the *training*
latents and betas; validation and test reuse these statistics. Legacy latent files store one subject id per
subject, which the dataset expands per scan.

## Other

- The KL term of the autoencoder loss is summed over the batch and divided by `x.numel()`; `gradient_accumulation_steps`
  is accepted but not used by the autoencoder loop; the diffusion loop clips gradients at a hard-coded 2.0
  (config value 1.0 is not used there).
- `min_baseline_years` in the multi-volume configuration is accepted for compatibility and has no effect. The
  *last* n conditioning scans are used; the "latest" format anchors at the year-3 scan with earlier scans as
  conditioning (negative time differences).
- The SynthSeg network in this repository is the robust 2.0 model (no flipped second pass, no topology
  clean-up). The paper calls the segmentation model SynthSeg+; the released network is the SynthSeg 2.0 robust model it
  refers to.
- Fig. 1 is an illustration and is not produced by code.

Reference values to compare with: paper Table 1 (see the README) and the validation checks in
`tools/validate_against_legacy.py`.

**Q16. Split sizes.** The legacy split CSVs hold 1563 / 32 / 178 subjects (train / val / test; 8056 / 184 / 989
scans, no overlap). The latent files, and hence all priors, use subjects with at least two scans: 1411 / 29 / 178.
The paper's "1411 / 32 / 178" mixes the two conventions for the validation split.

**Q17. DANINet subject list.** The legacy DaniNet datapoint list has 179 subjects; one is not in the test latents
and is dropped, leaving the 178 subjects of Table 1.
