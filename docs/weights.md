# Model weights

Download `mrextrap_weights.zip` from the
[Google Drive folder](https://drive.google.com/drive/folders/1cYC0gQXCsxkNzTXe7-2a2uCPoX7_SpbB?usp=sharing)
and unzip it into the run directory:

```bash
unzip mrextrap_weights.zip -d run_dir
```

| File | Content |
|---|---|
| `autoencoder/model.safetensors`, `autoencoder/config.yaml` | autoencoder used for all experiments |
| `unet_prior/checkpoints/epoch_2000/model.safetensors` | UNet prior |
| `diffusion_prior/checkpoints/epoch_2000/ema_denoiser.safetensors` | diffusion prior (EMA weights) |
| `priors/global_params.pt` | Global prior |
| `priors/normalization.pt` | latent, progression-rate and age normalisation |
| `synthseg_torch.pt` | SynthSeg 2.0 robust network (PyTorch conversion, Apache-2.0) |

The models were trained on ADNI; the files contain model parameters and aggregate statistics only.
