import math
from typing import Iterable, List

import torch
from torch import nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.
    """
    if timesteps.dim() != 1:
        timesteps = timesteps.view(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def _pick_groups(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float,
        max_groups: int = 8,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_pick_groups(in_channels, max_groups), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_pick_groups(out_channels, max_groups), out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(time_emb))[:, :, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.residual(x)


class Upsample1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        x = F.interpolate(x, size=target_length, mode="linear", align_corners=False)
        return self.conv(x)


class SimpleUnet(nn.Module):
    """U-Net LDM inspired 1D UNet for time-series generation."""

    def __init__(
        self,
        input_length: int,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Iterable[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_length <= 0:
            raise ValueError("input_length must be positive.")

        self.input_length = input_length
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_mults = list(channel_mults)
        self.num_res_blocks = num_res_blocks
        self.time_emb_dim = time_emb_dim
        self.dropout = dropout

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        channels = [base_channels * mult for mult in self.channel_mults]
        self.input_conv = nn.Conv1d(in_channels, channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        for idx, ch in enumerate(channels):
            resblocks = nn.ModuleList()
            for block_idx in range(num_res_blocks):
                resblocks.append(
                    ResBlock1D(
                        in_channels=channels[idx - 1] if block_idx == 0 and idx > 0 else ch,
                        out_channels=ch,
                        time_dim=time_emb_dim,
                        dropout=dropout,
                    )
                )
            downsample = None
            if idx < len(channels) - 1:
                downsample = nn.Conv1d(ch, ch, kernel_size=4, stride=2, padding=1)
            self.down_blocks.append(nn.ModuleDict({"resblocks": resblocks, "downsample": downsample}))

        mid_channels = channels[-1]
        self.mid_block1 = ResBlock1D(mid_channels, mid_channels, time_emb_dim, dropout)
        self.mid_block2 = ResBlock1D(mid_channels, mid_channels, time_emb_dim, dropout)

        self.up_blocks = nn.ModuleList()
        for idx in range(len(channels) - 1, 0, -1):
            in_ch = channels[idx]
            out_ch = channels[idx - 1]
            resblocks = nn.ModuleList()
            resblocks.append(ResBlock1D(in_ch + out_ch, out_ch, time_emb_dim, dropout))
            for _ in range(num_res_blocks - 1):
                resblocks.append(ResBlock1D(out_ch, out_ch, time_emb_dim, dropout))
            self.up_blocks.append(
                nn.ModuleDict({"upsample": Upsample1D(in_ch), "resblocks": resblocks})
            )

        self.output_norm = nn.GroupNorm(_pick_groups(channels[0]), channels[0])
        self.output_conv = nn.Conv1d(channels[0], in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        time_emb = self.time_mlp(timestep_embedding(timesteps, self.time_emb_dim))

        h = self.input_conv(x)
        skips: List[torch.Tensor] = []
        for block in self.down_blocks:
            for resblock in block["resblocks"]:
                h = resblock(h, time_emb)
            skips.append(h)
            if block["downsample"] is not None:
                h = block["downsample"](h)

        h = self.mid_block1(h, time_emb)
        h = self.mid_block2(h, time_emb)

        for idx, block in enumerate(self.up_blocks):
            skip = skips[-(idx + 2)]
            h = block["upsample"](h, target_length=skip.shape[-1])
            h = torch.cat([h, skip], dim=1)
            for resblock in block["resblocks"]:
                h = resblock(h, time_emb)

        h = self.output_conv(F.silu(self.output_norm(h)))
        return h
