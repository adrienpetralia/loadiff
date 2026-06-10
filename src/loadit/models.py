# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

    ##########################################################
# Modèle opérant sans espace latent,                            #
# Prends en entrée une donnée de forme : [B, C=1, N=365, D=48]  #
# Permet d'intégrer le conditionnement calendaire               #
    ##########################################################

import torch
import torch.nn as nn
import numpy as np
import math
import timm
# from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from timm.models.vision_transformer import Attention, Mlp
import collections.abc
from typing import Union, Optional


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
    

class MultiLabelEmbedder_old(nn.Module):
    """
    Multi-label conditioning embedder with classifier-free guidance support.

    Inputs:
      - labels can be:
          * multi_hot:  (B, C) float/bool/int in {0,1}
          * indices:    (B, K) Long with -1 as pad (variable set size)
    """
    def __init__(self,
                 num_classes: int,
                 hidden_size: int,
                 p_uncond: float = 0.1,
                 p_partial: float = 0.0,
                 pooling: str = "sum", # "mean" or "sum"
                ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.p_uncond = float(p_uncond)
        self.p_partial = float(p_partial)
        self.pooling = pooling

        # +1 for a learned "unconditional" token at index num_classes
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        nn.init.normal_(self.embedding_table.weight, std=0.02)

    def _apply_partial_dropout(self, multi_hot: torch.Tensor) -> torch.Tensor:
        """
        Per-label dropout on active entries with probability p_partial.
        multi_hot: (B, C) in {0,1}
        """
        if self.p_partial <= 0:
            return multi_hot
        # Drop only active labels
        drop_mask = (torch.rand_like(multi_hot.float()) < self.p_partial) & (multi_hot > 0)
        return multi_hot * (~drop_mask)

    def _labels_to_multi_hot(self, labels) -> torch.Tensor:
        """
        Normalize inputs to multi-hot (B, C) in {0,1}.
        - If already (B, C): pass through (cast to long/bool as needed)
        - If (B, K) with -1 pads: scatter to (B, C)
        """
        if labels.dim() == 2 and labels.size(1) == self.num_classes:
            # Already multi-hot
            mh = (labels > 0).long()
            return mh
        elif labels.dim() == 2:
            # (B, K) indices with -1 padding
            B, K = labels.shape
            mh = labels.new_zeros((B, self.num_classes), dtype=torch.long)
            valid = labels >= 0
            if valid.any():
                idx_b = torch.nonzero(valid, as_tuple=False)[:, 0]
                idx_c = labels[valid]
                mh[idx_b, idx_c] = 1
            return mh
        else:
            raise ValueError("labels must be (B, C) multi-hot or (B, K) indices with -1 padding")

    def forward(self, labels, train: bool, force_drop_ids: Optional[torch.Tensor] = None):
        """
        Returns:
          cond_emb: (B, hidden_size)
        Classifier-free guidance:
          - With prob p_uncond (during train, or when force_drop_ids==1) -> unconditional token
          - Optional per-label dropout p_partial on remaining labels (train only)
        """
        device = next(self.parameters()).device
        B = labels.shape[0]

        # Determine unconditional drops
        if force_drop_ids is not None:
            # force_drop_ids: (B,) with {0,1}, where 1 => uncond
            drop_all = (force_drop_ids == 1)
        else:
            drop_all = torch.zeros(B, dtype=torch.bool, device=device)
            if train and self.p_uncond > 0:
                drop_all = (torch.rand(B, device=device) < self.p_uncond)

        # If we’re unconditional for a sample, return the learned uncond embedding
        if drop_all.any():
            # Build unconditional embedding for dropped rows
            uncond_idx = torch.full((drop_all.sum().item(),), self.num_classes, dtype=torch.long, device=device)
            uncond_emb = self.embedding_table(uncond_idx)  # (n_drop, D)

        # For conditional rows: build multi-hot, optional partial drop, then pool
        cond_mask = ~drop_all
        cond_emb = torch.zeros((B, self.hidden_size), device=device)

        if cond_mask.any():
            labels_cond = labels[cond_mask]
            multi_hot = self._labels_to_multi_hot(labels_cond).to(device)

            # Optional per-label dropout (only during training)
            if train and self.p_partial > 0:
                multi_hot = self._apply_partial_dropout(multi_hot)

            # If after dropout there are zero active labels, fall back to unconditional
            none_active = (multi_hot.sum(dim=1) == 0)
            if none_active.any():
                # Replace those with the unconditional token
                uidx = torch.full((none_active.sum().item(),), self.num_classes, dtype=torch.long, device=device)
                cond_emb[cond_mask.nonzero(as_tuple=False)[none_active, 0]] = self.embedding_table(uidx)

            still_mask = cond_mask.clone()
            still_mask[cond_mask.clone()] &= ~none_active  # rows that still have labels

            if still_mask.any():
                # Gather per-class embeddings for active labels
                mh = multi_hot[~none_active]  # (B', C)
                # Fast linear combine: (B', C) @ (C, D) = (B', D)
                # Equivalent to sum of embeddings of active ids
                E = self.embedding_table.weight[:self.num_classes]  # (C, D)
                summed = mh.float() @ E  # (B', D)

                if self.pooling == "sum":
                    pooled = summed
                elif self.pooling == "mean":
                    denom = mh.sum(dim=1, keepdim=True).clamp_min(1.0)
                    pooled = summed / denom
                else:
                    raise ValueError(f"Unknown pooling: {self.pooling}")

                # Write back
                idx_rows = still_mask.nonzero(as_tuple=False).squeeze(1)
                cond_emb[idx_rows] = pooled

        # Stitch unconditional rows back
        if drop_all.any():
            cond_emb[drop_all] = uncond_emb

        return cond_emb


class MultiLabelEmbedder(nn.Module):
    """
    Fixed-length ternary label embedder.

    Expected input:
      labels: (B, C) or (C,)
      where C == num_classes, and each entry is:
        -1 -> unknown
         0 -> absent
         1 -> present

    Each label slot has its own 3-state embedding:
      slot i, state unknown
      slot i, state absent
      slot i, state present

    These per-slot embeddings are pooled into one conditioning vector.
    Classifier-free guidance is handled with a separate unconditional token.
    """
    def __init__(
        self,
        num_classes: int,
        hidden_size: int,
        p_uncond: float = 0.1,
        p_partial: float = 0.0,
        pooling: str = "sum",   # "sum" or "mean"
    ):
        super().__init__()
        if pooling not in {"sum", "mean"}:
            raise ValueError(f"Unknown pooling: {pooling}")

        self.num_classes = num_classes          # number of label slots
        self.hidden_size = hidden_size
        self.p_uncond = float(p_uncond)
        self.p_partial = float(p_partial)
        self.pooling = pooling

        # For each slot we store 3 embeddings:
        # state 0 -> unknown  (original label -1)
        # state 1 -> absent   (original label  0)
        # state 2 -> present  (original label  1)
        #
        # Total = num_classes * 3 embeddings
        # Plus 1 extra learned unconditional token for CFG
        self.uncond_idx = num_classes * 3
        self.embedding_table = nn.Embedding(num_classes * 3 + 1, hidden_size)
        nn.init.normal_(self.embedding_table.weight, std=0.02)

    def _normalize_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Ensure labels are shape (B, C) with values in {-1, 0, 1}.
        """
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        if labels.dim() != 2:
            raise ValueError(
                f"labels must have shape (B, C) or (C,), got {tuple(labels.shape)}"
            )

        if labels.size(1) != self.num_classes:
            raise ValueError(
                f"Expected labels.shape[1] == num_classes == {self.num_classes}, "
                f"got {labels.size(1)}"
            )

        labels = labels.long()
        if ((labels < -1) | (labels > 1)).any():
            raise ValueError("labels must only contain values in {-1, 0, 1}")

        return labels

    def _apply_partial_dropout(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Randomly hide observed labels by converting them to unknown (-1).
        This applies to both absent (0) and present (1), but not to already unknown labels.
        """
        if self.p_partial <= 0:
            return labels

        observed = labels >= 0
        drop_mask = (torch.rand(labels.shape, device=labels.device) < self.p_partial) & observed
        return torch.where(drop_mask, torch.full_like(labels, -1), labels)

    def _labels_to_token_ids(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Map labels in {-1, 0, 1} to embedding ids.

        For slot i:
          unknown -> 3*i + 0
          absent  -> 3*i + 1
          present -> 3*i + 2
        """
        # map {-1,0,1} -> {0,1,2}
        state_ids = labels + 1  # -1->0, 0->1, 1->2

        slot_offsets = (
            torch.arange(self.num_classes, device=labels.device, dtype=torch.long)
            .unsqueeze(0) * 3
        )  # (1, C)

        token_ids = slot_offsets + state_ids  # (B, C)
        return token_ids

    def forward(
        self,
        labels: torch.Tensor,
        train: bool,
        force_drop_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Returns:
          cond_emb: (B, hidden_size)

        CFG behavior:
          - with prob p_uncond during training, or when force_drop_ids==1,
            returns a learned unconditional embedding
          - otherwise embeds the full ternary label vector

        Note:
          an all-unknown label vector [-1, -1, ..., -1] is NOT the same as CFG-unconditional.
          It is a valid structured condition meaning "all labels unknown".
        """
        device = next(self.parameters()).device
        labels = self._normalize_labels(labels).to(device)
        B = labels.shape[0]

        # Whole-sample unconditional drop for CFG
        if force_drop_ids is not None:
            drop_all = (force_drop_ids == 1).to(device=device, dtype=torch.bool)
        else:
            drop_all = torch.zeros(B, dtype=torch.bool, device=device)
            if train and self.p_uncond > 0:
                drop_all = torch.rand(B, device=device) < self.p_uncond

        cond_emb = torch.zeros((B, self.hidden_size), device=device)

        # Unconditional rows
        if drop_all.any():
            uncond_idx = torch.full(
                (drop_all.sum().item(),),
                self.uncond_idx,
                dtype=torch.long,
                device=device
            )
            cond_emb[drop_all] = self.embedding_table(uncond_idx)

        # Conditional rows
        cond_mask = ~drop_all
        if cond_mask.any():
            labels_cond = labels[cond_mask]

            # Optional partial masking: turn some observed labels into unknown
            if train and self.p_partial > 0:
                labels_cond = self._apply_partial_dropout(labels_cond)

            token_ids = self._labels_to_token_ids(labels_cond)   # (B', C)
            token_embs = self.embedding_table(token_ids)         # (B', C, D)

            if self.pooling == "sum":
                pooled = token_embs.sum(dim=1)                   # (B', D)
            else:  # mean
                pooled = token_embs.mean(dim=1)                  # (B', D)

            cond_emb[cond_mask] = pooled

        return cond_emb


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    

class FinalLayer(nn.Module):
    """
    The final layer of DiT (adapted for rectangular patches).
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        if isinstance(patch_size, int):
            ph, pw = patch_size, patch_size
        else:
            ph, pw = patch_size
        self.patch_size = (ph, pw)
        self.out_channels = out_channels

        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, ph * pw * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # x: [B, N, hidden_size]
        # c: [B, hidden_size]  (e.g. timestep embedding)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)   # [B, N, hidden_size]
        x = self.linear(x)                               # [B, N, ph*pw*out_channels]
        return x
    
def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


class PatchEmbed(nn.Module): 
    """ 2D (365 x 48) to Patch Embedding
    modified function (based on original PatchEmbed from timm) """
    def __init__(self, 
                 img_size=(365, 48),
                 patch_size=(1, 48),  
                 in_chans=1,      
                 embed_dim=768,
                 norm_layer=None,
                 flatten=True):
        super().__init__()
       
        self.img_size = img_size
        self.patch_size = patch_size

        # grid size = number of patches along each axis
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # Conv2d does patchification
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                              kernel_size=patch_size, 
                              stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H//Ph, W//Pw]
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim], N = num_patches
        x = self.norm(x)
        return x
    

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=(365, 48),
        patch_size=(1, 48),
        in_channels=1,
        hidden_size=192,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        temp_dropout_prob=0.1,
        num_classes=0,
        multilabels=True,
        learn_sigma=True,
        n_exo_var = 0,
        temperature = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        
        self.x_embedder = PatchEmbed(
        img_size=input_size,  
        patch_size=patch_size,   
        in_chans=in_channels,         
        embed_dim=hidden_size,        
        norm_layer=nn.LayerNorm, 
        flatten=True
    )
        self.t_embedder = TimestepEmbedder(hidden_size)

        if num_classes > 0:
            if multilabels:
                self.y_embedder = MultiLabelEmbedder(num_classes, hidden_size, class_dropout_prob)
            else:
                self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)


        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        

        self.n_exo_var  = n_exo_var

        if self.n_exo_var > 0:
            self.proj_fourier_features_enc = nn.Linear(
                int(2 * self.n_exo_var),  
                hidden_size, bias=True
                )
        else:
            grid_h, grid_w = self.x_embedder.grid_size
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim=hidden_size,
                cls_token=False,
                extra_tokens=0,
                grid_size=(grid_h, grid_w)   
            )
            self.pos_embed = nn.Parameter(
                torch.from_numpy(pos_embed).float().unsqueeze(0), 
                requires_grad=False
            )

        self.temperature = temperature

        if self.temperature:
            self.temp_dropout_prob = temp_dropout_prob
            self.null_temp_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.proj_temperature = nn.Sequential(
                 nn.Linear(1,  hidden_size, bias=True),
                 nn.LayerNorm(hidden_size)
             )

        self.initialize_weights()


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        if self.n_exo_var==0:
            grid_h, grid_w = self.x_embedder.grid_size
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size=(grid_h, grid_w))
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if self.num_classes > 0:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def fourierfeatures(self, pos_encoding):  
        sin_cos = torch.cat([pos_encoding.sin(), pos_encoding.cos()], dim=-1)  
        pos_encoding_encoder = self.proj_fourier_features_enc(sin_cos)  # [B, D, hidden_dim]

        return pos_encoding_encoder
    
    def encodetemperature(self, temp):  
        temp = self.proj_temperature(temp)  # [B, D, hidden_dim]

        return temp

    def unpatchify(self, x):
        """
        x: (N, T, patch_h*patch_w*C)
        returns: (N, C, H, W)
        """
        c = self.out_channels
        ph, pw = self.patch_size
        N, T, _ = x.shape
        grid_h = self.x_embedder.grid_size[0]
        grid_w = self.x_embedder.grid_size[1]
        x = x.reshape(N, grid_h, grid_w, ph, pw, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(N, c, grid_h*ph, grid_w*pw)
        return imgs

    def forward(self, 
                x: torch.Tensor, 
                t: torch.Tensor,
                exog: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                force_temp_drop: Optional[torch.Tensor] = None):
        """
        force_temp_drop: boolean tensor of shape (B,) or None. 
                         If True for a batch item, drops temperature conditioning.
        """
        x_embed = self.x_embedder(x)

        if exog is not None:
            if self.temperature:
                # Assuming exog is [B, N, Features], temp is last feature
                calendar, temp = exog[..., :-1], exog[..., -1:]
                
                # 1. Project Calendar
                proj_calendar = self.fourierfeatures(calendar)
                
                # 2. Project Temperature
                proj_temp = self.encodetemperature(temp)

                # 3. Handle Temperature Dropout (CFG)
                B = x.shape[0]
                
                # Determine drop mask
                if force_temp_drop is not None:
                    # User forced drop (inference)
                    # Ensure shape is broadcastable: (B, 1, 1)
                    drop_mask = force_temp_drop.view(B, 1, 1).to(dtype=torch.bool, device=x.device)
                elif self.training and self.temp_dropout_prob > 0:
                    # Random drop (training)
                    drop_mask = (torch.rand(B, device=x.device) < self.temp_dropout_prob).view(B, 1, 1)
                else:
                    drop_mask = None

                # Apply Dropout
                if drop_mask is not None:
                    # null_temp_embed is (1, 1, hidden), it will broadcast to (B, N, hidden)
                    proj_temp = torch.where(drop_mask, self.null_temp_embed.to(x.device), proj_temp)

                x = x_embed + proj_calendar + proj_temp
            else:
                proj_calendar = self.fourierfeatures(exog)
                x = x_embed + proj_calendar
        else:
            x = x_embed + self.pos_embed

        t = self.t_embedder(t) # (N, D)
        
        if y is not None:
            # Assuming y_embedder handles its own CFG logic via internal force_drop_ids if needed
            y = self.y_embedder(y, self.training) 
            c = t + y
        else:
            c = t 

        for block in self.blocks:
            x = block(x, c) 

        x = self.final_layer(x, c) 
        x = self.unpatchify(x) 

        return x

    def forward_with_cfg(self, x, t, exog, y, cfg_scale): 
        """
        Runs the forward pass with Classifier-Free Guidance on temperature.
        Assumes input 'x' is already a double-batch (cond + uncond) or 
        you should restructure inputs before calling this.
        
        This implementation assumes you pass standard (B) sized inputs, 
        and it handles the doubling internally for easier usage.
        """
        
        # 1. Duplicate inputs for CFG (Conditional + Unconditional)
        x_in = torch.cat([x, x], dim=0)
        t_in = torch.cat([t, t], dim=0)
        exog_in = torch.cat([exog, exog], dim=0) if exog is not None else None
        
        # Handle labels y if they exist
        if y is not None:
             y_in = torch.cat([y, y], dim=0)
             # If using MultiLabelEmbedder or LabelEmbedder, they usually have their own
             # CFG logic. If you want to ONLY guide on temperature, keep Y as is.
             # If you want to guide on both, you might need to pass force_drop to y_embedder too.
        else:
            y_in = None

        # 2. Create Drop Mask for Temperature
        # First half: Keep Temp (False). Second half: Drop Temp (True).
        B = x.shape[0]
        force_temp_drop = torch.cat([
            torch.zeros(B, dtype=torch.bool, device=x.device), # Conditional
            torch.ones(B, dtype=torch.bool, device=x.device)   # Unconditional
        ], dim=0)

        # 3. Forward Pass
        # We use the modified forward that accepts force_temp_drop
        model_out = self.forward(x_in, t_in, exog=exog_in, y=y_in, force_temp_drop=force_temp_drop)

        # 4. Split and Guide
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, B, dim=0)
        
        # Guidance Formula: eps = uncond + scale * (cond - uncond)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        
        # If you need to return the full batch (doubled), concat. 
        # Usually for inference you only want the guided half.
        # But to match your original function signature which seemed to return doubled:
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: tuple (grid_h, grid_w)
    return:
    pos_embed: [grid_h*grid_w, embed_dim] or [1+grid_h*grid_w, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  
    grid = np.stack(grid, axis=0)  # (2, H, W)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def DiT_S_2(**kwargs): 
    return DiT(depth=12, hidden_size=384, num_heads=6, **kwargs) 

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=192, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=96, num_heads=6, **kwargs)


DiT_models = {
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
