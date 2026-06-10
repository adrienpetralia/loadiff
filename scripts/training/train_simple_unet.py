import logging
import os
import pickle
from typing import Tuple

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.baselines import SimpleUnet
from src.helpers.dataset import CdCDataset
from src.loadit.diffusion import create_diffusion
from src.loadit.diffusion.timestep_sampler import create_named_schedule_sampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def evaluate(
    model: SimpleUnet,
    diffusion,
    dataloader: DataLoader,
    sampler,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in dataloader:
        values = batch[0].to(device)
        values = values.flatten(start_dim=1).unsqueeze(1)
        timesteps, weights = sampler.sample(values.shape[0], device)
        losses = diffusion.training_losses(model, values, timesteps)
        loss = (losses["loss"] * weights).mean()
        total_loss += loss.item()
        n_batches += 1
    if n_batches == 0:
        return 0.0
    return total_loss / n_batches


@hydra.main(version_base=None, config_path="../../configs", config_name="diffusion_ts")
def train(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.training.global_seed)

    log_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(cfg.data.path_client_split, "rb") as f:
        splits = pickle.load(f)
    train_clients = splits["train"]
    val_clients = splits["val"]

    train_dataset = CdCDataset(
        path_parquet_part=cfg.data.data_path,
        list_pdl=train_clients,
        nb_days=cfg.data.nb_days,
        patch_length_day=cfg.data.patch_length_day,
        scale_param1=cfg.data.value_scale_min,
        scale_param2=cfg.data.value_scale_max,
        random_window=cfg.data.random_window,
    )

    val_dataset = CdCDataset(
        path_parquet_part=cfg.data.data_path,
        list_pdl=val_clients,
        nb_days=cfg.data.nb_days,
        patch_length_day=cfg.data.patch_length_day,
        scale_param1=cfg.data.value_scale_min,
        scale_param2=cfg.data.value_scale_max,
        random_window=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.valid.batch_size,
        shuffle=False,
        num_workers=cfg.valid.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    input_length = train_dataset.nb_days * train_dataset.patch_length
    model = SimpleUnet(
        input_length=input_length,
        in_channels=cfg.model.in_channels,
        base_channels=cfg.model.base_channels,
        channel_mults=cfg.model.channel_mults,
        num_res_blocks=cfg.model.num_res_blocks,
        time_emb_dim=cfg.model.time_emb_dim,
        dropout=cfg.model.dropout,
    ).to(device)

    diffusion = create_diffusion(
        timestep_respacing="",
        diffusion_steps=cfg.training.diffusion_steps,
        learn_sigma=False,
    )
    sampler = create_named_schedule_sampler(cfg.training.schedule_sampler, diffusion)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    best_val_loss = float("inf")
    global_step = 0
    checkpoint_every = cfg.training.get("checkpoint_every", cfg.training.log_every)

    for epoch in range(cfg.training.max_epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            global_step += 1
            values = batch[0].to(device)
            values = values.flatten(start_dim=1).unsqueeze(1)

            timesteps, weights = sampler.sample(values.shape[0], device)
            losses = diffusion.training_losses(model, values, timesteps)
            loss = (losses["loss"] * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % cfg.training.log_every == 0:
                logger.info("Epoch %s Step %s: loss=%.5f", epoch, step, total_loss / step)

            if checkpoint_every and global_step % checkpoint_every == 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "step": global_step,
                }
                torch.save(
                    checkpoint,
                    os.path.join(checkpoint_dir, f"diffusion_ts_step_{global_step:07d}.pt"),
                )

        avg_loss = total_loss / max(len(train_loader), 1)
        writer.add_scalar("train/loss", avg_loss, epoch)

        val_loss = evaluate(model, diffusion, val_loader, sampler, device)
        writer.add_scalar("val/loss", val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "model": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "step": global_step,
                "val_loss": val_loss,
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, "diffusion_ts_best.pt"))

    writer.close()


if __name__ == "__main__":
    train()