from typing import Tuple

import torch
from torch import nn


class _RNNBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float,
        apply_sigmoid: bool,
    ) -> None:
        super().__init__()
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.apply_sigmoid = apply_sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x)
        h = self.fc(h)
        if self.apply_sigmoid:
            h = torch.sigmoid(h)
        return h


class TimeGAN(nn.Module):
    """TimeGAN baseline adapted from the official implementation.

    Inputs are expected as (batch, seq_len, features) with features=1.
    """

    def __init__(
        self,
        input_length: int,
        input_dim: int = 1,
        hidden_dim: int = 24,
        num_layers: int = 3,
        z_dim: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_length <= 0:
            raise ValueError("input_length must be positive.")

        self.input_length = input_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.z_dim = z_dim
        self.dropout = dropout

        self.embedder = _RNNBlock(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=hidden_dim,
            dropout=dropout,
            apply_sigmoid=True,
        )
        self.recovery = _RNNBlock(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=input_dim,
            dropout=dropout,
            apply_sigmoid=True,
        )
        self.generator = _RNNBlock(
            input_dim=z_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=hidden_dim,
            dropout=dropout,
            apply_sigmoid=True,
        )
        self.supervisor = _RNNBlock(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=max(1, num_layers - 1),
            output_dim=hidden_dim,
            dropout=dropout,
            apply_sigmoid=True,
        )
        self.discriminator = _RNNBlock(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=1,
            dropout=dropout,
            apply_sigmoid=False,
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedder(x)

    def recover(self, h: torch.Tensor) -> torch.Tensor:
        return self.recovery(h)

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)

    def supervise(self, h: torch.Tensor) -> torch.Tensor:
        return self.supervisor(h)

    def discriminate(self, h: torch.Tensor) -> torch.Tensor:
        return self.discriminator(h)

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.input_length, self.z_dim, device=device)
        h = self.generate(z)
        h_hat = self.supervise(h)
        x_hat = self.recover(h_hat)
        return x_hat.squeeze(-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.embed(x)
        x_tilde = self.recover(h)
        return x_tilde, h
