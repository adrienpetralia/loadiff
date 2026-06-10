from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities: shapes, patching
# -----------------------------
def _to_3d_time_series(x: np.ndarray) -> np.ndarray:
    """
    Ensure x is (N, C, L).
    Accepts (N, L) or (N, C, L).
    """
    x = np.asarray(x)
    if x.ndim == 2:
        return x[:, None, :]
    if x.ndim == 3:
        return x
    raise ValueError(f"x must be 2D or 3D, got shape={x.shape}")


def _to_3d_metadata(c: np.ndarray, L: int) -> np.ndarray:
    """
    Ensure c is (N, F, L).
    Accepts (N, F) (static -> broadcast over time) or (N, F, L).
    """
    c = np.asarray(c)
    if c.ndim == 2:
        # static metadata, broadcast
        return np.repeat(c[:, :, None], repeats=L, axis=2)
    if c.ndim == 3:
        if c.shape[2] != L:
            raise ValueError(f"metadata length mismatch: c.shape[2]={c.shape[2]} vs L={L}")
        return c
    raise ValueError(f"c must be 2D or 3D, got shape={c.shape}")


def _extract_patches(
    x: torch.Tensor,  # (B, Cin, L)
    starts: torch.Tensor,  # (B, P) start indices
    patch_len: int
) -> torch.Tensor:
    """
    Extract aligned patches.
    Returns (B, P, Cin, patch_len)
    """
    B, Cin, L = x.shape
    P = starts.shape[1]
    # build indices: (B, P, patch_len)
    idx = starts.unsqueeze(-1) + torch.arange(patch_len, device=x.device).view(1, 1, -1)
    # idx in [0, L-1]
    # gather along time dim
    x_exp = x.unsqueeze(1).expand(B, P, Cin, L)           # (B, P, Cin, L)
    idx_exp = idx.unsqueeze(2).expand(B, P, Cin, patch_len)  # (B, P, Cin, patch_len)
    patches = torch.gather(x_exp, dim=3, index=idx_exp)   # (B, P, Cin, patch_len)
    return patches


# -----------------------------
# Encoders (simple but strong)
# -----------------------------
class ConvTransformerEncoder(nn.Module):
    """
    Lightweight sequence encoder:
      Conv1d -> TransformerEncoder -> global average pool -> Linear projection
    Works for both time series and metadata sequences of shape (B, Cin, L).
    """
    def __init__(
        self,
        in_channels: int,
        emb_dim: int = 128,
        model_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, model_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(model_dim, model_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.proj = nn.Linear(model_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Cin, L)
        returns: (B, emb_dim)
        """
        h = self.conv(x)          # (B, model_dim, L)
        h = h.transpose(1, 2)     # (B, L, model_dim)
        h = self.tr(h)            # (B, L, model_dim)
        h = h.mean(dim=1)         # (B, model_dim)
        z = self.proj(h)          # (B, emb_dim)
        return z


# -----------------------------
# Fréchet distance on embeddings
# -----------------------------
def _covariance(x: np.ndarray) -> np.ndarray:
    """
    x: (N, D)
    Returns (D, D) covariance (unbiased).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must be (N, D)")
    return np.cov(x, rowvar=False, bias=False)


def _sqrtm_product(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    """
    Compute matrix square root of (c1 @ c2) via scipy if available.
    Falls back to eigen-based approximation on a symmetrized product.
    """
    c1 = np.asarray(c1, dtype=np.float64)
    c2 = np.asarray(c2, dtype=np.float64)

    try:
        from scipy.linalg import sqrtm
        m = sqrtm(c1 @ c2)
        # numerical noise can introduce tiny imaginary parts
        if np.iscomplexobj(m):
            if np.max(np.abs(np.imag(m))) < 1e-6:
                m = np.real(m)
            else:
                raise ValueError("Large imaginary component in sqrtm result.")
        return m
    except Exception:
        # fallback: symmetrize the product and take PSD sqrt
        a = (c1 @ c2 + (c1 @ c2).T) / 2.0
        w, v = np.linalg.eigh(a)
        w = np.clip(w, 0.0, None)
        return (v * np.sqrt(w)) @ v.T


def frechet_distance_from_embeddings(
    z_real: np.ndarray,
    z_gen: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Standard Fréchet distance between Gaussians fitted to embeddings.

    z_real, z_gen: (N, D)
    """
    z_real = np.asarray(z_real, dtype=np.float64)
    z_gen  = np.asarray(z_gen, dtype=np.float64)

    if z_real.ndim != 2 or z_gen.ndim != 2:
        raise ValueError("Embeddings must be 2D arrays (N, D).")

    mu_r = z_real.mean(axis=0)
    mu_g = z_gen.mean(axis=0)

    c_r = _covariance(z_real)
    c_g = _covariance(z_gen)

    # stabilize
    c_r = c_r + eps * np.eye(c_r.shape[0])
    c_g = c_g + eps * np.eye(c_g.shape[0])

    diff = mu_r - mu_g
    covmean = _sqrtm_product(c_r, c_g)

    fid = float(diff @ diff + np.trace(c_r + c_g - 2.0 * covmean))
    # small negative due to numeric precision
    return max(0.0, fid)


# -----------------------------
# J-FTSD: training + scoring
# -----------------------------
@dataclass
class JFTSDConfig:
    emb_dim: int = 128
    model_dim: int = 128
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1

    # CLIP-like training
    patch_len: int = 64
    patches_per_sample: int = 4
    temperature: float = 0.07

    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class JFTSD:
    """
    End-to-end J-FTSD:
      - fit feature extractors (time series + metadata) using CLIP-like patch contrastive loss
      - compute joint embeddings for real vs generated, then Fréchet distance

    Notes:
      - Fit encoders on REAL paired data (typically train split), then score on eval split.
      - This matches the paper’s intent: a dataset-specific feature space learned from real (x, c).
    """
    def __init__(self, cfg: JFTSDConfig):
        self.cfg = cfg
        self.fx: Optional[nn.Module] = None
        self.fc: Optional[nn.Module] = None

    def fit(self, x_real: np.ndarray, c_real: np.ndarray) -> "JFTSD":
        x3 = _to_3d_time_series(x_real)              # (N, C, L)
        N, C, L = x3.shape
        c3 = _to_3d_metadata(c_real, L=L)            # (N, F, L)
        Fm = c3.shape[1]

        cfg = self.cfg
        device = torch.device(cfg.device)

        self.fx = ConvTransformerEncoder(
            in_channels=C,
            emb_dim=cfg.emb_dim,
            model_dim=cfg.model_dim,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
        ).to(device)

        self.fc = ConvTransformerEncoder(
            in_channels=Fm,
            emb_dim=cfg.emb_dim,
            model_dim=cfg.model_dim,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
        ).to(device)

        opt = torch.optim.AdamW(
            list(self.fx.parameters()) + list(self.fc.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        # tensors on CPU; move per batch (helps memory)
        x_t = torch.from_numpy(x3.astype(np.float32))   # (N, C, L)
        c_t = torch.from_numpy(c3.astype(np.float32))   # (N, F, L)

        max_start = L - cfg.patch_len
        if max_start < 0:
            raise ValueError(f"patch_len={cfg.patch_len} longer than series length L={L}")

        n_steps = int(np.ceil(N / cfg.batch_size))

        self.fx.train()
        self.fc.train()

        for epoch in range(cfg.epochs):
            # shuffle indices each epoch
            perm = torch.randperm(N)
            for step in range(n_steps):
                idx = perm[step * cfg.batch_size : (step + 1) * cfg.batch_size]
                xb = x_t[idx].to(device)  # (B, C, L)
                cb = c_t[idx].to(device)  # (B, F, L)
                B = xb.shape[0]

                # sample aligned patch starts: (B, P)
                starts = torch.randint(
                    low=0, high=max_start + 1, size=(B, cfg.patches_per_sample), device=device
                )

                x_p = _extract_patches(xb, starts, cfg.patch_len)  # (B, P, C, pl)
                c_p = _extract_patches(cb, starts, cfg.patch_len)  # (B, P, F, pl)

                # flatten patches into batch dimension
                x_p = x_p.reshape(B * cfg.patches_per_sample, C, cfg.patch_len)
                c_p = c_p.reshape(B * cfg.patches_per_sample, Fm, cfg.patch_len)

                zx = self.fx(x_p)  # (BP, d)
                zc = self.fc(c_p)  # (BP, d)

                # CLIP-style normalization
                zx = F.normalize(zx, dim=1)
                zc = F.normalize(zc, dim=1)

                logits = (zx @ zc.T) / cfg.temperature   # (BP, BP)
                labels = torch.arange(logits.shape[0], device=device)

                loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        return self

    @torch.no_grad()
    def _joint_embeddings(self, x: np.ndarray, c: np.ndarray, batch_size: int = 512) -> np.ndarray:
        if self.fx is None or self.fc is None:
            raise RuntimeError("Call fit() before computing embeddings.")

        x3 = _to_3d_time_series(x)
        N, C, L = x3.shape
        c3 = _to_3d_metadata(c, L=L)
        Fm = c3.shape[1]

        device = torch.device(self.cfg.device)
        self.fx.eval()
        self.fc.eval()

        out = []
        for i in range(0, N, batch_size):
            xb = torch.from_numpy(x3[i:i+batch_size].astype(np.float32)).to(device)  # (B, C, L)
            cb = torch.from_numpy(c3[i:i+batch_size].astype(np.float32)).to(device)  # (B, F, L)

            zx = self.fx(xb)
            zc = self.fc(cb)

            # The paper concatenates embeddings to form a joint space. :contentReference[oaicite:6]{index=6}
            z = torch.cat([zx, zc], dim=1)  # (B, 2d)
            out.append(z.cpu().numpy())

        return np.concatenate(out, axis=0)

    def score(
        self,
        x_real_eval: np.ndarray,
        c_real_eval: np.ndarray,
        x_gen: np.ndarray,
        c_gen: np.ndarray,
        eps: float = 1e-6,
    ) -> float:
        """
        Compute J-FTSD between (x_real_eval, c_real_eval) and (x_gen, c_gen)
        using the learned encoders.
        """
        z_real = self._joint_embeddings(x_real_eval, c_real_eval)
        z_gen  = self._joint_embeddings(x_gen, c_gen)
        return frechet_distance_from_embeddings(z_real, z_gen, eps=eps)


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Toy example shapes:
    # x: (N, L) univariate series
    # c: (N, F) static metadata (broadcast over time internally)
    N, L, Fm = 2000, 96, 10
    x_real = np.random.randn(N, L)
    c_real = np.random.randn(N, Fm)

    # pretend generator outputs:
    x_gen = np.random.randn(N, L) * 1.1
    c_gen = c_real.copy()

    # train encoders on real (train split); here we just reuse all as a demo
    cfg = JFTSDConfig(emb_dim=64, patch_len=32, patches_per_sample=4, epochs=5, batch_size=256)
    metric = JFTSD(cfg).fit(x_real, c_real)

    # score (eval split vs generated)
    jftsd_value = metric.score(x_real, c_real, x_gen, c_gen)
    print("J-FTSD:", jftsd_value)
