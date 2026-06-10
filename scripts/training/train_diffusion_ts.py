import logging
import os
import pickle
from typing import Tuple

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.helpers.label_filter import filter_dataset_by_label
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.baselines import Diffusion_TS
from src.helpers.dataset import SmachDataset, CERDataset, CERBisDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def evaluate(
    model: Diffusion_TS,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in dataloader:
        values = batch[0].to(device)
        values = values.flatten(start_dim=1).unsqueeze(-1)
        loss = model(values)
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

    if cfg.data.dataset=="smach":
        CdCDataset = SmachDataset
    elif cfg.data.dataset=="cer":
        CdCDataset = CERDataset
    elif cfg.data.dataset=="cer_bis":
        CdCDataset = CERBisDataset
    else:
        raise ValueError(
            f"Unknown data.dataset={cfg.data.dataset!r}. Expected one of: smach, cer, cer_bis."
        )

    # Optional per-appliance filtering: train a dedicated model on the subset of
    # clients matching training.filter_by_label (e.g. {heater: 1}). Disabled when empty.
    filter_by_label = OmegaConf.select(cfg, "training.filter_by_label", default=None)
    filter_by_label = dict(filter_by_label) if filter_by_label else {}

    def _make_dataset(list_pdl, random_window):
        kwargs = dict(
            path_load_curves=cfg.data.data_path,
            list_pdl=list_pdl,
            nb_days=cfg.data.nb_days,
            patch_length_day=cfg.data.patch_length_day,
            scale_param1=cfg.data.value_scale_min,
            scale_param2=cfg.data.value_scale_max,
            random_window=random_window,
        )
        if filter_by_label:
            # Only the columns we actually filter on are needed (the baselines do
            # not condition on labels). Deriving them from filter_by_label keeps
            # this dataset-agnostic: no need for data.bool_col_names to match.
            path_metadata = OmegaConf.select(cfg, "data.path_parquet_part_metadata", default=None)
            if not path_metadata:
                raise ValueError(
                    "training.filter_by_label requires data.path_parquet_part_metadata to be set."
                )
            kwargs["bool_col_names"] = list(filter_by_label.keys())
            kwargs["path_metadata"] = path_metadata
        dataset = CdCDataset(**kwargs)
        if filter_by_label:
            dataset = filter_dataset_by_label(dataset, filter_by_label)
        return dataset

    train_dataset = _make_dataset(train_clients, cfg.data.random_window)
    val_dataset = _make_dataset(val_clients, False)

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
    model = Diffusion_TS(
        seq_length=input_length,
        feature_size=cfg.model.feature_size,
        n_layer_enc=cfg.model.n_layer_enc,
        n_layer_dec=cfg.model.n_layer_dec,
        d_model=cfg.model.d_model,
        timesteps=cfg.model.timesteps,
        sampling_timesteps=cfg.model.sampling_timesteps,
        loss_type=cfg.model.loss_type,
        beta_schedule=cfg.model.beta_schedule,
        n_heads=cfg.model.n_heads,
        mlp_hidden_times=cfg.model.mlp_hidden_times,
        eta=cfg.model.eta,
        attn_pd=cfg.model.attn_pd,
        resid_pd=cfg.model.resid_pd,
        kernel_size=cfg.model.kernel_size,
        padding_size=cfg.model.padding_size,
        use_ff=cfg.model.use_ff,
        reg_weight=cfg.model.reg_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    best_val_loss = float("inf")
    epoch = 0
    global_step = 0
    checkpoint_every = cfg.training.get("checkpoint_every", cfg.training.log_every)

    for epoch in range(cfg.training.max_epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            global_step += 1
            values = batch[0].to(device)
            values = values.flatten(start_dim=1).unsqueeze(-1)

            loss = model(values)

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

        val_loss = evaluate(model, val_loader, device)
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