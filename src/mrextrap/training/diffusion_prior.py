"""Training of the latent-diffusion prior over progression rates (betas).

Ported from ``train_latent_diffusion_for_betas.py`` as pinned by the W&B code snapshot of run
``train_beta_diffusion_smaller64highlr3e-4longtrainlinear`` (``bfxpfvzd``). ``configs/diffusion_prior.yaml``
holds that run's hyperparameters (not the later "WIDER" ones).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm.auto import tqdm

from mrextrap.data.datasets import get_latent_dataloaders
from mrextrap.io import load_state_dict_any, save_safetensors
from mrextrap.models.unet3d import Normalization, UNetModel
from mrextrap.training.unet_prior import (
    is_save_epoch,
    legacy_epoch_label,
    output_dir,
    save_normalization,
    seed_everything,
)

# NOTE(paper): the research code clips gradients at a hard-coded 2.0 (``clip_grad_norm_(..., 2.0)``);
# ``training.grad_clip_val: 1.0`` in the config is never read there. Preserved.
GRAD_CLIP_NORM = 2.0


def build_denoiser(model_cfg) -> UNetModel:
    return UNetModel(
        image_size=list(model_cfg.image_size),
        in_channels=model_cfg.in_channels * 2,  # noisy beta and conditioning latent, concatenated
        out_channels=model_cfg.out_channels,
        model_channels=model_cfg.model_channels,
        num_res_blocks=model_cfg.num_res_blocks,
        attention_resolutions=list(model_cfg.attention_resolutions),
        channel_mult=list(model_cfg.channel_mult),
        num_classes=1,  # hack: embeds the (normalised) age of the conditioning scan into the ResBlocks
        concat_emb=False,
    )


def build_noise_scheduler(training_cfg):
    from diffusers.schedulers import DDPMScheduler

    return DDPMScheduler(
        num_train_timesteps=training_cfg.NT,
        clip_sample=False,
        beta_schedule=training_cfg.get("noise_schedule", "linear"),
    )


def train_diffusion_prior(cfg: DictConfig) -> Path:
    """Train the diffusion prior; returns the stage output directory.

    Layout: ``<out>/config.yaml``, ``<out>/normalization.pt``,
    ``<out>/checkpoints/<legacy_epoch_label>/{ema_denoiser.safetensors, state/}``.
    ``state/`` (accelerate) holds the raw, non-averaged weights and optimiser state.
    """
    from accelerate import Accelerator
    from diffusers.optimization import get_scheduler
    from torch_ema import ExponentialMovingAverage as EMAModel

    out_dir = output_dir(cfg, "diffusion_prior")
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)
    seed_everything(cfg.training.seed)

    denoiser = build_denoiser(cfg.model)
    noise_scheduler = build_noise_scheduler(cfg.training)
    train_dataloader, val_dataloader, _ = get_latent_dataloaders(
        cfg, num_workers=cfg.training.get("num_workers", 4)
    )
    dataset = train_dataloader.dataset
    age_normalization = Normalization(dataset.ages_abs.mean(), dataset.ages_abs.std())
    save_normalization(dataset.normalization_stats(), cfg, out_dir)

    # NOTE(paper): (Q5) AdamW betas (0.95, 0.999); the paper text says (0.9, 0.999).
    optimizer = torch.optim.AdamW(
        denoiser.parameters(), lr=cfg.training.lr, betas=(0.95, 0.999), weight_decay=1e-6, eps=1e-8
    )
    # NOTE(paper): cosine horizon is 1.3x the real number of steps (lr ends at ~20% of peak).
    lr_scheduler = get_scheduler(
        cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.training.get("num_warmup_steps", 1000),
        num_training_steps=len(train_dataloader) * cfg.training.num_epochs * 1.3,
    )
    l2loss = nn.MSELoss(reduction="mean")

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with="wandb" if cfg.training.get("use_wandb", False) else None,
        cpu=bool(cfg.training.get("cpu", False)),  # force CPU (tests; avoids MPS on macOS)
        dynamo_backend="INDUCTOR" if cfg.training.torch_compile else None,
    )
    denoiser, optimizer, train_dataloader, val_dataloader, lr_scheduler, age_normalization = (
        accelerator.prepare(
            denoiser, optimizer, train_dataloader, val_dataloader, lr_scheduler, age_normalization
        )
    )
    # NOTE(paper): EMA over the prepared model's parameters, decay 0.99, updated every step.
    ema_denoiser = EMAModel(denoiser.parameters(), decay=cfg.training.ema_decay)

    use_wandb = cfg.training.get("use_wandb", False)
    if use_wandb and accelerator.is_main_process:
        accelerator.init_trackers(
            "mrextrap",
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": {"name": cfg.exp_name}},
        )

    ckpt_root = out_dir / "checkpoints"
    history: list[dict] = []
    global_step = 0
    num_epochs = cfg.training.num_epochs
    for epoch in tqdm(
        range(num_epochs), desc="Epochs", disable=not accelerator.is_local_main_process
    ):
        accelerator.wait_for_everyone()
        for batch in train_dataloader:
            x_cond = batch["zs"]
            age0_norm = age_normalization(batch["ages_abs"].unsqueeze(1).float())
            beta = batch["betas"]

            noise = torch.randn_like(beta)
            timesteps = (
                torch.randint(0, noise_scheduler.config.num_train_timesteps, (x_cond.shape[0],))
                .long()
                .to(noise.device)
            )
            zt = noise_scheduler.add_noise(beta, noise, timesteps)

            denoiser_input = torch.cat([zt, x_cond], dim=1)
            pred_noise = denoiser(denoiser_input, timestep=timesteps, y=age0_norm).sample

            loss = l2loss(noise, pred_noise)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(denoiser.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "epoch": epoch,
            }
            history.append(logs)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                ema_denoiser.update()
            if global_step % 5 == 0 and use_wandb:
                accelerator.log(logs, step=global_step)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process and is_save_epoch(epoch, num_epochs):
            if torch.isnan(loss).any():
                raise ValueError("Loss is NaN. Training failed.")
            # NOTE(paper): (Q4) the label is not the number of completed epochs.
            save_dir = ckpt_root / legacy_epoch_label(epoch)
            save_dir.mkdir(parents=True, exist_ok=True)
            accelerator.save_state(output_dir=str(save_dir / "state"))
            with ema_denoiser.average_parameters():
                save_safetensors(
                    accelerator.unwrap_model(denoiser, keep_fp32_wrapper=False).state_dict(),
                    save_dir / "ema_denoiser.safetensors",
                )
        accelerator.wait_for_everyone()

    accelerator.end_training()
    (out_dir / "train_log.json").write_text(json.dumps(history))
    return out_dir


def load_diffusion_prior(cfg: DictConfig, checkpoint_dir: str | Path) -> UNetModel:
    """Build the denoiser from ``cfg.model`` and load the EMA weights of a checkpoint directory."""
    denoiser = build_denoiser(cfg.model)
    denoiser.load_state_dict(load_state_dict_any(Path(checkpoint_dir) / "ema_denoiser.safetensors"))
    return denoiser.eval()


__all__ = [
    "GRAD_CLIP_NORM",
    "build_denoiser",
    "build_noise_scheduler",
    "load_diffusion_prior",
    "train_diffusion_prior",
]
