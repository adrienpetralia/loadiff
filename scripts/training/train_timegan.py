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

from src.baselines import TimeGAN
from src.helpers.dataset import SmachDataset, CERDataset, CERBisDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _supervised_loss(h: torch.Tensor, h_hat: torch.Tensor) -> torch.Tensor:
    return torch.mean((h[:, 1:, :] - h_hat[:, :-1, :]) ** 2)


def _moment_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    mean_diff = torch.mean(x_hat, dim=0) - torch.mean(x, dim=0)
    var_diff = torch.var(x_hat, dim=0, unbiased=False) - torch.var(x, dim=0, unbiased=False)
    return torch.mean(torch.abs(mean_diff)) + torch.mean(torch.abs(var_diff))


@torch.no_grad()
def evaluate(
    model: TimeGAN,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_recon = 0.0
    total_sup = 0.0
    n_batches = 0
    for batch in dataloader:
        values = batch[0].to(device)
        values = values.flatten(start_dim=1).unsqueeze(-1)
        h = model.embed(values)
        x_tilde = model.recover(h)
        h_sup = model.supervise(h)
        recon_loss = torch.mean((values - x_tilde) ** 2)
        sup_loss = _supervised_loss(h, h_sup)
        total_recon += recon_loss.item()
        total_sup += sup_loss.item()
        n_batches += 1
    if n_batches == 0:
        return 0.0, 0.0
    return total_recon / n_batches, total_sup / n_batches


@hydra.main(version_base=None, config_path="../../configs", config_name="timegan")
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
    model = TimeGAN(
        input_length=input_length,
        input_dim=cfg.model.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        z_dim=cfg.model.z_dim,
        dropout=cfg.model.dropout,
    ).to(device)

    embedder_params = list(model.embedder.parameters()) + list(model.recovery.parameters())
    supervisor_params = list(model.supervisor.parameters())
    generator_params = list(model.generator.parameters())
    discriminator_params = list(model.discriminator.parameters())

    embedder_opt = torch.optim.Adam(embedder_params + supervisor_params, lr=cfg.training.lr)
    generator_opt = torch.optim.Adam(generator_params + supervisor_params, lr=cfg.training.lr)
    discriminator_opt = torch.optim.Adam(discriminator_params, lr=cfg.training.lr)

    bce_loss = torch.nn.BCEWithLogitsLoss()
    best_val_loss = float("inf")

    for epoch in range(cfg.training.max_epochs):
        model.train()
        total_recon = 0.0
        total_sup = 0.0
        total_gen = 0.0
        total_disc = 0.0

        for step, batch in enumerate(train_loader, start=1):
            values = batch[0].to(device)
            values = values.flatten(start_dim=1).unsqueeze(-1)

            h = model.embed(values)
            x_tilde = model.recover(h)
            h_sup = model.supervise(h)

            recon_loss = torch.mean((values - x_tilde) ** 2)
            sup_loss = _supervised_loss(h, h_sup)
            e_loss = recon_loss + cfg.training.gamma * sup_loss

            embedder_opt.zero_grad()
            e_loss.backward()
            embedder_opt.step()

            z = torch.randn(values.shape[0], input_length, cfg.model.z_dim, device=device)
            h_hat = model.generate(z)
            h_hat_sup = model.supervise(h_hat)
            x_hat = model.recover(h_hat_sup)

            g_loss_adv = bce_loss(model.discriminate(h_hat_sup), torch.ones(values.shape[0], input_length, 1, device=device))
            g_loss_sup = _supervised_loss(h_hat, h_hat_sup)
            g_loss_moment = _moment_loss(values, x_hat)
            g_loss = g_loss_adv + cfg.training.gamma * g_loss_sup + cfg.training.moment_loss_weight * g_loss_moment

            generator_opt.zero_grad()
            g_loss.backward()
            generator_opt.step()

            d_loss_real = bce_loss(model.discriminate(h.detach()), torch.ones(values.shape[0], input_length, 1, device=device))
            d_loss_fake = bce_loss(model.discriminate(h_hat_sup.detach()), torch.zeros(values.shape[0], input_length, 1, device=device))
            d_loss = d_loss_real + d_loss_fake

            if d_loss.item() > cfg.training.discriminator_threshold:
                discriminator_opt.zero_grad()
                d_loss.backward()
                discriminator_opt.step()

            total_recon += recon_loss.item()
            total_sup += sup_loss.item()
            total_gen += g_loss.item()
            total_disc += d_loss.item()

            if step % cfg.training.log_every == 0:
                logger.info(
                    "Epoch %s Step %s: recon=%.5f sup=%.5f gen=%.5f disc=%.5f",
                    epoch,
                    step,
                    total_recon / step,
                    total_sup / step,
                    total_gen / step,
                    total_disc / step,
                )

        avg_recon = total_recon / max(len(train_loader), 1)
        avg_sup = total_sup / max(len(train_loader), 1)
        avg_gen = total_gen / max(len(train_loader), 1)
        avg_disc = total_disc / max(len(train_loader), 1)

        writer.add_scalar("train/recon", avg_recon, epoch)
        writer.add_scalar("train/supervised", avg_sup, epoch)
        writer.add_scalar("train/generator", avg_gen, epoch)
        writer.add_scalar("train/discriminator", avg_disc, epoch)

        val_recon, val_sup = evaluate(model, val_loader, device)
        writer.add_scalar("val/recon", val_recon, epoch)
        writer.add_scalar("val/supervised", val_sup, epoch)

        checkpoint = {
            "model": model.state_dict(),
            "config": cfg,
            "epoch": epoch,
            "val_loss": val_recon,
        }
        # torch.save(checkpoint, os.path.join(checkpoint_dir, f"timegan_epoch_{epoch:04d}.pt"))

        if val_recon < best_val_loss:
            best_val_loss = val_recon
            torch.save(checkpoint, os.path.join(checkpoint_dir, "timegan_best.pt"))

    writer.close()


if __name__ == "__main__":
    train()