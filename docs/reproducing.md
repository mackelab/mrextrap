# Reproducing the paper

Two routes: (A) train everything from ADNI data with the commands below; (B) start from released weights
(see [weights.md](weights.md)) and skip the training steps. Retraining is not bit-exact
([reproducibility.md](reproducibility.md)); differences between code and paper text that affect numbers are in
[paper_notes.md](paper_notes.md).

Command syntax: `mrextrap <command> --config <yaml> [key=value ...]`. Paths come from `configs/paths.yaml`.
`mrextrap <command> --help` shows the options. Commands without a dedicated YAML take `configs/paths.yaml`
(read from the working directory when `--config` is omitted) plus dotlist overrides; the stage configs pass
the settings of the stage they follow (for example the model block of the autoencoder config for `encode`).
Run the commands from the repository root.

## Run-directory layout

```
run_dir/
  splits/{train,val,test}.csv           user-generated (docs/data.md)
  daninet_pairs.csv                     user-supplied DANINet test pairs
  weights/synthseg_torch.pt             or $MREXTRAP_SYNTHSEG_WEIGHTS
  autoencoder/{config.yaml, model.safetensors, checkpoints/}
  latents/{split}_zs_ages.pt            latents, ages, subject ids
  latents/{split}_latent_betas_l1.pt    + per-scan betas
  priors/global_params.pt               global_beta_mean, global_beta_std, global_obs_var
  priors/normalization.pt
  unet_prior/{config.yaml, checkpoints/epoch_XXXX/model.safetensors}
  diffusion_prior/{config.yaml, checkpoints/epoch_XXXX/ema_denoiser.safetensors}
  volumes/{split}_real/<subject>/{*_final.nii.gz, *_synthseg_robust.nii.gz, vols.csv}
  eval/single_volume/{global,unet,diffusion}/results.json
  eval/multivolume/{earliest,latest}/results_var1.00.json
  figures/{cache/figN.pkl, figN.pdf}, tables/
```

## Pipeline

```bash
# 0-1. preprocessing and splits (docs/preprocessing.md, docs/data.md)
python preprocessing/adni/preprocess_adni.py all --input ... --adnimerge ADNIMERGE.csv --workdir ...
mrextrap make-splits paths.run_dir=/data/run splits.adnimerge_csv=ADNIMERGE.csv splits.image_index_csv=index.csv
# 2-5. autoencoder, latents, progression rates, Global prior
mrextrap train-ae --config configs/autoencoder.yaml paths.run_dir=/data/run
mrextrap encode --config configs/autoencoder.yaml paths.run_dir=/data/run
mrextrap fit-betas --config configs/autoencoder.yaml paths.run_dir=/data/run
mrextrap global-prior --config configs/autoencoder.yaml paths.run_dir=/data/run
# 6-7. learned priors
mrextrap train-unet --config configs/unet_prior.yaml paths.run_dir=/data/run
mrextrap train-diffusion --config configs/diffusion_prior.yaml paths.run_dir=/data/run
# 8. segmentation of the real scans (needs the SynthSeg weights)
mrextrap real-volumes --config configs/eval_single_volume.yaml paths.run_dir=/data/run
# 9-11. evaluation
mrextrap eval-single --config configs/eval_single_volume.yaml paths.run_dir=/data/run
mrextrap eval-multi  --config configs/eval_multivolume.yaml   paths.run_dir=/data/run
mrextrap eval-ae     --config configs/autoencoder.yaml        paths.run_dir=/data/run ae_eval.device=cuda
mrextrap interpolation --config configs/autoencoder.yaml      paths.run_dir=/data/run interp.device=cuda
# 12. figures and table
mrextrap table1  --config configs/eval_single_volume.yaml paths.run_dir=/data/run
mrextrap figures --config configs/eval_multivolume.yaml   paths.run_dir=/data/run
```

`mrextrap figures` renders every `figN` whose cache file `run_dir/figures/cache/figN.pkl` exists; the cache
files are the intermediate data of each figure (documented in the docstring of `make_figN` in
`src/mrextrap/figures/`), so figures can be re-rendered without a GPU.

Config files: `configs/autoencoder.yaml` (paper autoencoder, partly reconstructed; see the comments in
the file), `configs/unet_prior.yaml`, `configs/diffusion_prior.yaml`, `configs/eval_single_volume.yaml`
(K = 5, seed 0, checkpoint label `epoch_2000`, `delta_t_from_first_scan: true`),
`configs/eval_multivolume.yaml` (`conditioning_years [1,2,3]`, `min_future_years 2`,
`baseline_formats [earliest, latest]`), `configs/smoke/*.yaml` (tiny CPU configs used by tests and
`mrextrap smoke`).

## Table and figure map

| Paper item | Steps | Result file | Paper value |
|---|---|---|---|
| Sec. 3.1 SSIM | `eval-ae` | metrics of `eval-ae` | 0.95 |
| Sec. 3.1 GDice | `eval-ae` (default `ae_eval.gdice_mode=regions`: 5-region generalized Dice, plus `dice_per_region`; `ae_eval.gdice_mode=notebook` reproduces the preprint value) | `volumes/test_ae_metrics.json` | 0.92 in the preprint (hippocampus channel only, see note) |
| Fig. 2a latent channels | `encode`, `figures` | `figures/fig2.pdf` | qualitative |
| Fig. 2b/c interpolation, linearity R^2 | `interpolation` (`interp.mode` latent or voxel), `figures` | `figures/fig2.pdf` | R^2 curves; curves with R^2 < 0.7 are dropped; scan pairs up to 4.1 years apart |
| Fig. 2d latent PCA | `encode`, `figures` | `figures/fig2.pdf` | qualitative |
| Table 1 Global | `fit-betas`, `global-prior`, `real-volumes`, `eval-single eval.methods=[global]`, `table1` | `eval/single_volume/global/results.json` | HC 0.020 +- 0.001, VT 0.322 +- 0.027, GM 0.509 +- 0.030, WM 0.399 +- 0.023 |
| Table 1 UNet | + `train-unet`, `eval-single eval.methods=[unet]` | `eval/single_volume/unet/results.json` | 0.022 +- 0.001, 0.310 +- 0.029, 0.619 +- 0.043, 0.443 +- 0.028 |
| Table 1 Diffusion | + `train-diffusion`, `eval-single eval.methods=[diffusion]` | `eval/single_volume/diffusion/` | 0.021 +- 0.001, 0.261 +- 0.019, 0.639 +- 0.032, 0.376 +- 0.022 |
| Table 1 DANINet row, p-values | `table1` | `tables/` | DANINet 0.060 +- 0.003, 0.257 +- 0.017, 0.829 +- 0.046, 0.829 +- 0.046 (constants from the DANINet paper) |
| Fig. 3 | `eval-multi` (both baseline formats), `figures` | `eval/multivolume/{earliest,latest}/results_var1.00.json`, `figures/fig3.pdf` | panel b correlations 0.33, 0.73, 0.68, 0.71 |
| Fig. 4 | `eval-multi`, `figures` | `figures/fig4.pdf` | qualitative (subject ids set in config) |
| Fig. 5 | `fit-betas`, `figures` | `figures/fig5.pdf` | qualitative |

HC hippocampus, VT ventricles, GM grey matter, WM white matter; MAE of regional volume (as %TBV, Q9) with standard error
over 178 test subjects (standard deviation with `ddof=0` divided by `sqrt(178)`). The single-volume
evaluation needs `daninet_pairs.csv` (docs/data.md).

Reproduction tolerances: from the released weights and cached latents, MAE values should agree with Table 1
to about the third decimal (SynthSeg and bf16 differences aside). Retrained models will differ; the
qualitative ordering of methods is what should be compared.

## Checking against the paper artefacts (cluster)

`tools/import_legacy_artifacts.py` maps the legacy cluster directories onto the layout above and
`tools/validate_against_legacy.py --list` lists the checks (V0 to V12 plus Q2 checks) with their tolerances.
`tools/convert_checkpoints.py` produces the weight files. All three print `--help`. These tools need the
paper artefacts and are not runnable without them.

## Smoke test

`mrextrap smoke [paths.run_dir=DIR]` runs the full pipeline on a synthetic cohort of small phantom volumes on the CPU
(configs in `configs/smoke/`; temporary directory unless `paths.run_dir` is given): autoencoder (2 steps), encoding,
progression rates, Global prior, UNet and diffusion priors (2 epochs), autoencoder metrics, real volumes,
interpolation, single- and multi-volume evaluation with the weight-free `FakeSegmenter`, figures 2, 3 and 5 and
Table 1. It takes about one minute and checks that every stage reads what the previous one wrote; its numbers
are meaningless. `pytest -q tests/test_e2e_synthetic.py` runs the same thing under pytest.
