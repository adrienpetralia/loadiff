from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Union, Iterable, Tuple, Dict, List, Optional

from abc import ABC
from contextlib import contextmanager


ArrayLike = np.ndarray | torch.Tensor

class BaseFeaturesExtractor(nn.Module, ABC):
    """
    Reusable base class for feature extractors.

    Key responsibilities:
    - Accepts inputs as np.ndarray or torch.Tensor
    - Converts numpy -> torch and moves inputs to the extractor device
    - Runs feature extraction in inference mode (no grad), preserving the module's training state
    - Optionally returns NumPy outputs (CPU) for metric computation pipelines

    Subclasses should implement `forward(self, x: torch.Tensor) -> torch.Tensor`.
    """

    def __init__(
        self,
        *,
        input_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.input_dtype = input_dtype

        # Ensures the module always has a device anchor even if it has no parameters.
        # Calling `.to(device)` on the module will move this buffer too.
        self.register_buffer("_device_ref", torch.empty(0), persistent=False)

    @property
    def device(self) -> torch.device:
        return self._device_ref.device

    def _to_tensor(self, x: ArrayLike) -> torch.Tensor:
        """
        Convert x to a torch.Tensor on `self.device` with dtype `self.input_dtype`.
        """
        if isinstance(x, np.ndarray):
            if x.dtype == np.object_:
                raise TypeError("numpy array has dtype=object; cannot convert safely to torch.Tensor.")
            t = torch.from_numpy(x)
        elif torch.is_tensor(x):
            t = x
        else:
            raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")

        # Cast + move to the extractor device
        # (also handles cases where the input tensor is already on GPU/CPU, etc.)
        t = t.to(device=self.device, dtype=self.input_dtype, non_blocking=True)

        # Conv/linear ops are generally happier with contiguous tensors
        if not t.is_contiguous():
            t = t.contiguous()

        return t

    @contextmanager
    def _inference_context(self):
        """
        Inference context that:
        - preserves the current `.training` state
        - runs with `torch.inference_mode()`
        """
        was_training = self.training
        try:
            self.eval()
            with torch.inference_mode():
                yield
        finally:
            if was_training:
                self.train()

    def extract(self, x: ArrayLike, *, as_numpy: bool = False) -> np.ndarray | torch.Tensor:
        """
        Extract features for a single batch/population.

        Args:
            x: np.ndarray or torch.Tensor (batch-first)
            as_numpy: if True, returns CPU np.ndarray; otherwise returns torch.Tensor on the extractor device.
        """
        x_t = self._to_tensor(x)
        with self._inference_context():
            feats = self(x_t)

        if as_numpy:
            return feats.detach().cpu().numpy()
        return feats

    def get_features(
        self,
        real_data: ArrayLike,
        synth_data: ArrayLike,
        *,
        as_numpy: bool = True,
    ) -> Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        """
        Extract features for real and synthetic data.

        Args:
            real_data: np.ndarray or torch.Tensor (batch-first)
            synth_data: np.ndarray or torch.Tensor (batch-first)
            as_numpy: if True, returns (np.ndarray, np.ndarray) on CPU; else returns torch tensors on extractor device.
        """
        feat_real = self.extract(real_data, as_numpy=as_numpy)
        feat_synth = self.extract(synth_data, as_numpy=as_numpy)
        return feat_real, feat_synth



class ROCKET(BaseFeaturesExtractor):
    """
    Efficient ROCKET: groups kernels by (kernel_size, dilation, padding)
    and applies each group in a single conv call.

    Input:  x  (B, C, L)
    Output: feats (B, 2 * n_kernels)  -> [MAX, PPV] per kernel
    """
    def __init__(
        self,
        seq_len: int = 17520,
        c_in: int = 1,
        n_kernels: int = 1_000,
        kss: Iterable[int] = (7, 9, 11),
        seed: int = 0,
        normalize_features: bool = True, 
        weight_unit_norm: bool = False,
        channel_subsample: bool = False,  # if True, sample a random subset of channels per kernel
    ):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.normalize_features = normalize_features

        kss = [int(k) for k in kss if int(k) < seq_len]
        if not kss:
            raise ValueError("All kernel sizes are >= seq_len; provide smaller values.")

        # --- Sample kernel hyperparams for each kernel ---
        ks_arr = np.random.choice(kss, size=n_kernels)

        # sample dilation as powers of two that fit the sequence length
        dilations = np.empty(n_kernels, dtype=np.int64)
        paddings  = np.empty(n_kernels, dtype=np.int64)
        for i, ks in enumerate(ks_arr):
            max_dil = max(1, (seq_len - 1) // (ks - 1))
            max_pow = int(math.floor(math.log2(max_dil))) if max_dil > 0 else 0
            dil = 2 ** np.random.randint(0, max_pow + 1)
            pad = ((ks - 1) * dil) // 2 if np.random.randint(2) == 1 else 0
            dilations[i] = dil
            paddings[i]  = pad

        # optional channel subsampling (close to some ROCKET variants)
        if channel_subsample and c_in > 1:
            # choose a random number of channels per kernel in [1, c_in], bias toward more channels
            ch_counts = np.random.randint(1, c_in + 1, size=n_kernels)
            ch_indices = [np.random.choice(c_in, size=k, replace=False) for k in ch_counts]
        else:
            ch_indices = [None] * n_kernels

        # --- Build weight & bias per kernel ---
        # We'll group them by (ks, dil, pad) to batch the conv calls.
        groups: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
        for i, (ks, dil, pad) in enumerate(zip(ks_arr, dilations, paddings)):
            groups[(int(ks), int(dil), int(pad))].append(i)

        self._groups_meta = []  # list of (ks, dil, pad, n_group)
        self._weights = nn.ParameterList()  # store as Parameters with requires_grad=False
        self._biases  = nn.ParameterList()
        self._ch_masks: List[torch.Tensor | None] = []  # (n_group, c_in) masks or None

        for (ks, dil, pad), idxs in groups.items():
            n_g = len(idxs)

            # weights: (n_group, C_in, ks)
            W = torch.randn(n_g, c_in, ks)
            # zero-mean per kernel
            W -= W.mean(dim=(1, 2), keepdim=True)

            if weight_unit_norm:
                # L2-normalize per kernel
                W = W / (W.flatten(1).norm(p=2, dim=1, keepdim=True).clamp_min(1e-12).view(n_g, 1, 1))

            # optional channel masks if channel_subsample is used
            if ch_indices[0] is None:
                ch_mask = None
            else:
                ch_mask = torch.zeros(n_g, c_in)
                for row, k_idx in enumerate(idxs):
                    ch_mask[row, ch_indices[k_idx]] = 1.0
                # Apply mask by zeroing unused channels (cheaper than per-kernel slicing)
                W = W * ch_mask.unsqueeze(-1)

            # biases U(-1, 1)
            b = 2 * (torch.rand(n_g) - 0.5)

            # freeze
            W = nn.Parameter(W, requires_grad=False)
            b = nn.Parameter(b, requires_grad=False)

            self._weights.append(W)
            self._biases.append(b)
            self._groups_meta.append((ks, dil, pad, n_g))
            self._ch_masks.append(ch_mask)  # keep for info; not used in forward

        # total kernels must match
        assert sum(n_g for *_, n_g in self._groups_meta) == n_kernels
        self.n_kernels = n_kernels

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
        - (B, L) or (B, C, L)
        returns:
        - (B, 2 * n_kernels)  -> interleaved [MAX, PPV] per kernel group-wise concatenated
        """
        # If input is (B, L), assume single channel and convert to (B, 1, L)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3:
            raise ValueError(f"Expected x to have shape (B, L) or (B, C, L), got {tuple(x.shape)}")

        B = x.size(0)
        feats = []
        for (ks, dil, pad, _), W, b in zip(self._groups_meta, self._weights, self._biases):
            # W: (n_group, C, ks) acts like out_channels = n_group
            # b: (n_group,)
            y = F.conv1d(x, W, b, stride=1, padding=pad, dilation=dil)  # (B, n_group, L')
            # MAX and PPV per kernel
            y_max = y.amax(dim=-1)                     # (B, n_group)
            y_ppv = (y > 0).float().mean(dim=-1)       # (B, n_group)
            feats.append(y_max)
            feats.append(y_ppv)

        # Concatenate across all groups, then interleave already done by appending [max, ppv] per group
        out = torch.cat(feats, dim=1)  # (B, 2 * n_kernels)

        if self.normalize_features:
            out = F.normalize(out, p=2, dim=1)

        return out

    def extra_repr(self) -> str:
        gs = len(self._groups_meta)
        return f"n_kernels={self.n_kernels}, groups={gs}, " + \
               "specs=" + ", ".join([f"(ks={ks},dil={d},pad={p},n={n})" for (ks,d,p,n) in self._groups_meta[:5]]) + \
               ("..." if len(self._groups_meta) > 5 else "")


@torch.no_grad()
def create_rocket_features(
    dataloader: torch.utils.data.DataLoader, 
    model: nn.Module
    ):
    """
    Args:
        dataloader: yields (xb, yb) where xb is (B, C, L)
        model     : ROCKET instance (put it on the same device as xb)
    Returns:
        X_feat: np.ndarray of shape (N, 2 * n_kernels)
        y:      np.ndarray of shape (N,)
    """
    X_list, y_list = [], []
    for xb, yb in dataloader:
        xb = xb.to(next(model.parameters(), model._dummy).device)
        feats = model(xb)            # (B, 2K)
        X_list.append(feats.cpu())
        y_list.append(yb.cpu())
    X = torch.cat(X_list, dim=0).numpy()
    y = torch.cat(y_list, dim=0).numpy()
    return X, y
