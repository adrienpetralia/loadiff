import math
from typing import Iterable, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _conv1d_out_length(length: int, kernel_size: int, stride: int, padding: int) -> int:
    return math.floor((length + 2 * padding - kernel_size) / stride) + 1


class TimeVAE(nn.Module):
    """TimeVAE adaptation for 1D time-series.

    Inputs are expected as (batch, time) or (batch, 1, time).
    """

    def __init__(
        self,
        input_length: int,
        latent_dim: int = 16,
        hidden_channels: Iterable[int] = (32, 64, 128),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_length <= 0:
            raise ValueError("input_length must be positive.")

        self.input_length = input_length
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.hidden_channels = list(hidden_channels)

        padding = kernel_size // 2
        lengths = [input_length]
        encoder_layers = []
        in_channels = 1

        for out_channels in self.hidden_channels:
            encoder_layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                )
            )
            encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            lengths.append(_conv1d_out_length(lengths[-1], kernel_size, stride=2, padding=padding))
            in_channels = out_channels

        self.encoder = nn.Sequential(*encoder_layers)
        self.lengths = lengths

        reduced_length = lengths[-1]
        flattened_dim = self.hidden_channels[-1] * reduced_length
        self.fc_mu = nn.Linear(flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(flattened_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flattened_dim)

        decoder_channels = list(reversed(self.hidden_channels))
        self.decoder_convs = nn.ModuleList(
            [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
                for in_ch, out_ch in zip(decoder_channels[:-1], decoder_channels[1:])
            ]
        )
        self.final_conv = nn.Conv1d(
            self.hidden_channels[0], 1, kernel_size=kernel_size, padding=padding
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.encoder(x)
        h = h.flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, self.hidden_channels[-1], self.lengths[-1])

        for idx in range(len(self.hidden_channels) - 1):
            target_length = self.lengths[-(idx + 2)]
            h = F.interpolate(h, size=target_length, mode="linear", align_corners=False)
            h = self.decoder_convs[idx](h)
            h = F.leaky_relu(h, 0.2)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)

        h = F.interpolate(h, size=self.input_length, mode="linear", align_corners=False)
        h = self.final_conv(h)
        return h

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss, kl_loss

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z)
        return samples.squeeze(1)
