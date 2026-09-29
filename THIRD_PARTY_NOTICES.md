# Third-party notices

The original code of this repository is MIT-licensed (see `LICENSE`). The components below are vendored,
adapted or required, and keep their own licences. Licences are
listed as declared by the upstream projects.

## Vendored or adapted code

| Component | Used in | Upstream | Licence |
|---|---|---|---|
| SynthSeg (torch port of the robust 2.0 model, pre/post-processing, label tables) | `src/mrextrap/segmentation/` | https://github.com/BBillot/SynthSeg | Apache-2.0; full text and notice in `src/mrextrap/segmentation/LICENSE-SynthSeg`. |
| Latent-diffusion / guided-diffusion style 3D UNet building blocks | `src/mrextrap/models/unet3d.py` | Rombach et al., latent-diffusion (CompVis); OpenAI guided-diffusion | MIT |
| DDPM UNet reference (N-d adaptation) | `src/mrextrap/models/unet3d.py` | https://github.com/hojonathanho/diffusion | MIT |
| Adversarial loss | `src/mrextrap/losses.py` | https://github.com/AmigoLab/SynthAnatomy (`src/losses/adversarial/adversarial.py`) | Apache-2.0 |
| PatchGAN-style baseline discriminator | `src/mrextrap/models/autoencoder.py` | https://github.com/AmigoLab/SynthAnatomy (`src/networks/discriminator/baseline.py`) | Apache-2.0 |
| LPIPS wrapper (slice-wise 3D perceptual loss) | `src/mrextrap/losses.py` | https://github.com/mueller-franzes/medfusion (`medical_diffusion/loss/perceivers.py`) | MIT |
| SSIM implementation | `src/mrextrap/losses.py` | pytorch-msssim style implementation (https://github.com/VainF/pytorch-msssim) | MIT |
| Autoencoder building blocks (ResNet blocks, `Normalize`, `Upsample`) | `src/mrextrap/models/autoencoder.py` | latent-diffusion / taming-transformers style VAE | MIT |

### SynthSeg

The segmentation code is adapted from SynthSeg. If you use the segmentation-derived volumes, please cite:

> B. Billot, D. N. Greve, O. Puonti, A. Thielscher, K. Van Leemput, B. Fischl, A. V. Dalca, J. E. Iglesias.
> SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining.
> Medical Image Analysis 86, 102789, 2023.

The network weights `synthseg_torch.pt` are not part of this repository. They are a conversion of the
official SynthSeg 2.0 (robust) weights and remain subject to the upstream terms (Apache-2.0). 

### FSL-derived and optiBET preprocessing scripts

`preprocessing/adni/fsl/` contains shell scripts under their own terms (details in
`preprocessing/adni/fsl/NOTICE.md`):

- `_preprocess.sh`: adapted from `fsl_anat` (Mark Jenkinson, FMRIB, University of Oxford). FMRIB Software
  Library licence, header retained: non-commercial use only; modified redistribution only without financial
  return, with the licence imposed on the receiver. This file and its derivative `preprocess.sh` are **not**
  under the MIT licence.
- `optiBET.sh`: E. Lutkenhoff (Monti Lab, UCLA, 2014), simplified by D. Patterson (2018). No licence text in
  the file; it ships under its upstream terms, not MIT. Method: Lutkenhoff et al., "Optimized brain extraction for
  pathological brains (optiBET)", PLoS ONE 9(12), 2014.

FSL itself (bet, flirt, fnirt, fast, robustfov, fslmaths, ...) is an **external requirement**, not
distributed here, and is licensed by FMRIB for non-commercial use. Install it from
https://fsl.fmrib.ox.ac.uk and cite the FSL papers listed on that site. The FSL version used for the
paper was not recorded; the scripts were written for FSL 6.

## Dependencies (installed by pip, not vendored)

| Package | Licence |
|---|---|
| PyTorch | BSD-3-Clause |
| Hugging Face diffusers, accelerate | Apache-2.0 |
| MONAI | Apache-2.0 |
| lpips (Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric", CVPR 2018); downloads pretrained VGG weights at run time | BSD-2-Clause (package) |
| torch-ema, einops, omegaconf, safetensors | MIT / Apache-2.0 (see each package) |
| numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, nibabel, SimpleITK, tqdm | BSD / Apache-2.0 / MIT (see each package) |

## Data

No ADNI data are included. Use of ADNI data is governed by the ADNI Data Use Agreement
(https://adni.loni.usc.edu). Acknowledgement text required by ADNI should be included in any publication
using ADNI data.
