# MRExtrap: Longitudinal Aging of Brain MRIs using Linear Modeling in Latent Space

This repository contains research code for the paper:
[***MRExtrap: Longitudinal Aging of Brain MRIs using Linear Modeling in Latent Space***](https://arxiv.org/abs/2508.19482)
by Kapoor, Macke and Baumgartner (2025).

MRExtrap predicts future brain MRIs of a subject by linear extrapolation in the latent space of a 3D
autoencoder, with the progression rate taken from a population (Global), UNet or diffusion prior, and
updated in closed form when more scans are available.

## Installation

We recommend a fresh environment with Python 3.11. A GPU is needed to train and evaluate at full size.

```bash
git clone git@github.com:mackelab/mrextrap.git
cd mrextrap
conda create --name mrextrap python=3.11
conda activate mrextrap
pip install -e .
```

Check the installation with a tiny synthetic run on CPU (no data needed, about a minute):

```bash
mrextrap smoke
```

### Data

The experiments use [ADNI](https://adni.loni.usc.edu), which cannot be redistributed. Apply for access and
download the T1 MPRAGE scans and `ADNIMERGE.csv`, then preprocess them with
[`preprocessing/adni/preprocess_adni.py`](preprocessing/adni) (needs FSL; see
[docs/preprocessing.md](docs/preprocessing.md)). Split files and the DANINet test pairs are described in
[docs/data.md](docs/data.md).

## Running the experiments

The code is in [`src/mrextrap`](src/mrextrap). Each step is one command; paths are set in
[`configs/paths.yaml`](configs/paths.yaml) and all outputs go to the run directory.

| Step | Command |
|------|---------|
| make train/val/test splits | `mrextrap make-splits` |
| train autoencoder | `mrextrap train-ae --config configs/autoencoder.yaml` |
| encode scans | `mrextrap encode --config configs/autoencoder.yaml` |
| fit progression rates | `mrextrap fit-betas` |
| Global prior | `mrextrap global-prior` |
| train UNet prior | `mrextrap train-unet --config configs/unet_prior.yaml` |
| train diffusion prior | `mrextrap train-diffusion --config configs/diffusion_prior.yaml` |
| segment real scans | `mrextrap real-volumes` |
| evaluate autoencoder (Sec. 3.1) | `mrextrap eval-ae --config configs/autoencoder.yaml` |
| single-volume prediction (Table 1) | `mrextrap eval-single --config configs/eval_single_volume.yaml` |
| multi-volume prediction (Fig. 3) | `mrextrap eval-multi --config configs/eval_multivolume.yaml` |
| latent interpolation (Fig. 2) | `mrextrap interpolation --config configs/autoencoder.yaml` |
| figures and Table 1 | `mrextrap figures`, `mrextrap table1` |

[docs/reproducing.md](docs/reproducing.md) gives the exact invocation and expected numbers for each paper
result, and [docs/paper_notes.md](docs/paper_notes.md) lists where the code differs from the paper text.

### Checkpoints

We provide the trained models in this [Google Drive folder](https://drive.google.com/drive/folders/1cYC0gQXCsxkNzTXe7-2a2uCPoX7_SpbB?usp=sharing).
Download `mrextrap_weights.zip` and unzip it into the run directory (`unzip mrextrap_weights.zip -d run_dir`).
It contains the autoencoder, the UNet and diffusion priors, the Global prior and the SynthSeg network used
for segmentation; see [docs/weights.md](docs/weights.md).

## Citation

```
@article{kapoor2025mrextrap,
  title   = {MRExtrap: Longitudinal Aging of Brain MRIs using Linear Modeling in Latent Space},
  author  = {Kapoor, Jaivardhan and Macke, Jakob H. and Baumgartner, Christian F.},
  journal = {arXiv preprint arXiv:2508.19482},
  year    = {2025}
}
```

## License

MIT (see [LICENSE](LICENSE)). Third-party code keeps its own license, see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Contact

Please open a GitHub issue for any questions.
