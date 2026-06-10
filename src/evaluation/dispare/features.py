from omegaconf import OmegaConf
import torchaudio.transforms as T
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sktime.transformations.panel.rocket import MiniRocket
from numpy.typing import NDArray
from torch import Tensor

from abc import abstractmethod
from typing import Union


class FeaturesExtractor:
    """Base class for features extraction"""

    @abstractmethod
    def fit(self, x: Tensor, y=None) -> Tensor:
        """Fit features extractor on given data."""

    @abstractmethod
    def __call__(self, x: Tensor, y=None) -> Tensor:
        """Extract features of given data

        Args:
            x (NDArray): Data
            y (NDArray): Metadata or classes

        Returns:
            NDArray: Features
        """


class FEComposition(FeaturesExtractor):
    """Compose different features extractors"""

    def __init__(self, fe: list[FeaturesExtractor]) -> None:
        super().__init__()
        self.fe = fe

    def fit(self, x: Tensor, y=None):
        for f in self.fe:
            x = f.fit(x, y)
        return x

    def __call__(self, x: Tensor, y=None):
        for f in self.fe:
            x = f(x, y)
        return x


class RawFeatures(FeaturesExtractor):
    """Extract nothing, it is the identity function."""

    def fit(self, x, y=None):
        return x

    def __call__(self, x, y=None):
        return x


class SeasonExtractor(FeaturesExtractor):
    """Mask the data."""

    def __init__(self, mask: NDArray[np.bool_]) -> None:
        """Initialize season extractor with given mask"""
        self.mask = torch.tensor(mask, dtype=torch.bool)

    def fit(self, x, y=None):
        return x

    def __call__(self, x, y=None):
        if x.device != self.mask.device:
            self.mask = self.mask.to(x.device)
        return x[:, self.mask]


class ClassExtractor(FeaturesExtractor):
    """Extract data that has given class."""

    def __init__(self, class_name: str, class_value) -> None:
        """Initialize class extractor with given class name and required
        value"""
        self.class_name = class_name
        self.class_value = class_value

    def fit(self, x, y):
        mask = y[self.class_name] == self.class_value
        mask = mask.to(x.device)
        return x[mask]

    def __call__(self, x, y):
        mask = y[self.class_name] == self.class_value
        mask = mask.to(x.device)
        return x[mask]


class MiniRExtractor(FeaturesExtractor):
    """Mini Rocket"""

    def __init__(
        self,
        num_kernels=500,
        max_dilations_per_kernel=32,
        n_jobs=1,
        random_state: Union[int, None] = None,
    ):
        """Initialize features extractor by MiniRocket

        Args:
            see sktime's minirocket documentation
        """
        self.num_kernels = num_kernels
        self.max_dilations_per_kernels = max_dilations_per_kernel
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.mini_r = MiniRocket(
            num_kernels=num_kernels,
            max_dilations_per_kernel=max_dilations_per_kernel,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def fit(self, x, y=None):
        x = x.cpu().numpy()
        if len(x.shape) == 2:
            x = np.expand_dims(x, 1)
        else:
            x = np.swapaxes(x, 1, 2)
        x = self.mini_r.fit_transform(x)
        return torch.tensor(x.values)

    def __call__(self, x, y=None):
        self.mini_r.check_is_fitted()
        x = x.cpu().numpy()
        if len(x.shape) == 2:
            x = np.expand_dims(x, 1)
        else:
            x = np.swapaxes(x, 1, 2)
        x = self.mini_r.transform(x)
        return torch.tensor(x.values)


######################################################
#        Implémentation d'un Auto-Encodeur           #
#       L'implé vient du stage de génération         #
#             de courbes de charge                   #
######################################################


class Unsqueeze(nn.Module):
    """Unsqueeze layer"""

    def __init__(self, dim: int):
        """Initialize module

        Args:
            dim (int): dim to be unsqueezed
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return torch.unsqueeze(x, self.dim)


class Squeeze(nn.Module):
    """Unsqueeze layer"""

    def __init__(self, dim: int | None = None):
        """Initialize module

        Args:
            dim (int): dim to be squeezed
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return torch.squeeze(x, self.dim)


class Spectrogram(nn.Module):
    def __init__(self, n_fft=96, win_length=48, hop_length=12, device=None):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.device = device
        self.transform = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        ).to(device)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(1)
        return self.transform(x)


def bf_hf_split(x: Tensor, threshold=0.2) -> tuple[Tensor, Tensor]:
    thresh = int(threshold * x.size(1))
    return x[:, :thresh], x[:, thresh:]


def periodic_pad_1d(x: Tensor, pad: int | tuple[int, int]) -> Tensor:
    """Periodic padding for 1D sequences

    Args:
        x (Tensor): input tensor to pad. It must be of shape (N, T, ...) and
            the padding will occur along the dimension 1.
        pad (int | tuple[int, int]): Number of values that will be padded on
            the left and on the right.

    Returns:
        Tensor: Padded input
    """
    # If pad is an int, apply same padding left and right
    if isinstance(pad, int):
        pad = pad, pad
    assert pad[0] <= x.size(
        1
    ), f""""preriodic_pad_1d" not supported for pad >= x.size(1). Got pad={pad[0]}, size={x.size(1)}"""  # noqa
    assert pad[1] <= x.size(
        1
    ), f""""preriodic_pad_1d" not supported for pad >= x.size(1). Got pad={pad[1]}, size={x.size(1)}"""  # noqa

    # If pad[0] == 0 then x[:, -pad[0]:] is the hole sequence
    if pad[0] > 0:
        x = torch.concatenate((x[:, -pad[0] :], x, x[:, : pad[1]]), dim=1)
    else:
        x = torch.concatenate((x, x[:, : pad[1]]), dim=1)
    return x


def periodic_pad_2d(
    x: Tensor, pad: tuple[int, int] | tuple[int, int, int, int]
) -> Tensor:
    """Periodic padding for 2D sequences

    Args:
        x (Tensor): input tensor to pad. It must be of shape (N, T, T', ...)
            and the padding will occur along the T, T' dimensions. Dimension T
            is periodic.
        pad (tuple[int, int]): Number of values that will be padded on the
            left, right, top and bottom.

    Returns:
        Tensor: padded tensor
    """
    if len(pad) == 2:
        pad = pad[0], pad[0], pad[1], pad[1]
    # Pad 1D is done 1 time more than needed for the later shift
    x = periodic_pad_1d(x, (pad[0] + 1, pad[1] + 1))
    assert pad[2] <= x.size(
        2
    ), """"preriodic_pad_2d" not supported for pad > x.size(2)."""  # noqa
    assert pad[3] <= x.size(
        2
    ), """"preriodic_pad_2d" not supported for pad > x.size(2)."""  # noqa
    # Shift
    if pad[2] != 0:
        c = x[:, :-2, -pad[2] :], x[:, 1:-1], x[:, 2:, : pad[3]]
    else:
        c = x[:, 1:-1], x[:, 2:, : pad[3]]
    x = torch.concatenate(c, dim=2)
    return x


class PeriodicPad2d(nn.Module):
    """2D periodic padding layer

         L1 M1 m1 J1 V1 S1 D1
         L2 M2 m2 J2 V2 S2 D2
         L3 M3 m3 J3 V3 S3 D3

    is padded to

      S3 D3 L3 M3 m3 J3 V3 S3 D3
      D1 L1 M1 m1 J1 V1 S1 D1 L1
      D2 L2 M2 m2 J2 V2 S2 D2 L2
      D3 L3 M3 m3 J3 V3 S3 D3 L3
      L1 M1 m1 J1 V1 S1 D1 L1 M1

    """

    def __init__(
        self,
        pad: tuple[int, int] | tuple[int, int, int, int],
    ) -> None:
        super().__init__()
        self.pad = pad

    def forward(self, x: Tensor) -> Tensor:
        """Pad the input Tensor along time dim T, T'

        Args:
            x (Tensor): Tensor of shape (N, C, T, T').

        Returns:
            Tensor: output of the pass forward
        """
        x = x.transpose(1, 3)  # C, T, T' -> T', T, C
        x = x.transpose(1, 2)  # T', T, C -> T, T', C
        x = periodic_pad_2d(x, self.pad)
        x = x.transpose(1, 3)  # T, T', C -> C, T', T
        x = x.transpose(2, 3)  # C, T', T -> C, T, T'
        return x


class AdaptativeCrop2d(nn.Module):

    def __init__(self, size: tuple[int, int] | None = None) -> None:
        super().__init__()
        self.size = size

    def forward(self, x: Tensor, size: tuple[int, int] | None = None) -> Tensor:
        """Pad/Crop the input Tensor along time dim T to obtain the
        target size.

        Args:
            x (Tensor): Tensor of shape (N, C, T, T').

        Returns:
            Tensor: output of the pass forward
        """
        # Get target size
        if size is None:
            if self.size is None:
                raise ValueError("You must give a size")
            size = self.size
        # Crop to right size
        w, h = x.size(2), x.size(3)
        sw = max((w - size[0]) // 2, 0)
        sh = max((h - size[1]) // 2, 0)
        x = x[:, :, sw : sw + size[0], sh : sh + size[1]]
        return x


class Conv2dBlock(nn.Module):
    """2D convolution block"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_layers: int,
        padding="periodic",
        device=None,
        dtype=None,
    ):
        """Initialize Module and internal states.

        Args:
            in_channels (int): number of channels of the input
            out_channels (int): number of channels of the output
            kernel_size (int): Duh
            num_layers (int): number of convolution layer
            padding (str, optional): "periodic" or "zeros".
                Defaults to "periodic".
            device (_type_, optional): Module's device. Defaults to None.
            dtype (_type_, optional): Module's dtype. Defaults to None.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = (out_channels,)
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        pad = (
            kernel_size // 2,
            (kernel_size // 2) + (kernel_size % 2) - 1,
            kernel_size // 2,
            (kernel_size // 2) + (kernel_size % 2) - 1,
        )
        if padding == "periodic":
            self.padding = PeriodicPad2d(pad=pad)
        elif padding == "zeros":
            self.padding = nn.ZeroPad2d(pad)
        else:
            raise NotImplementedError(f"Padding {padding} not implemented")
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, device=device, dtype=dtype
                )
            ]
        )
        for _ in range(num_layers - 1):
            self.conv_layers.append(
                nn.Conv2d(
                    out_channels, out_channels, kernel_size, device=device, dtype=dtype
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        """Pass forward

        Args:
            x (Tensor): Tensor of shape (N, C, T, T').

        Returns:
            Tensor: output of the pass forward
        """
        for cl in self.conv_layers:
            old_x, shape = x, x.shape
            x = self.padding(x)
            x = cl(x)
            x = F.relu(x)
            if x.shape == shape:
                x = old_x + x
        return x


class DailyDiscriminator(nn.Module):
    """Discriminator classifying days."""

    def __init__(self, num_layers: int, device=None):
        super().__init__()
        self.device = device

        self.norm = nn.InstanceNorm1d(48)
        self.conv_start = nn.Conv1d(48, 256, 1).to(device)
        self.conv = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.conv.append(nn.Conv1d(256, 256, 1).to(device))
        self.conv_end = nn.Conv1d(256, 1, 1).to(device)

    def forward(self, x: Tensor) -> Tensor:
        """Pass forward

        Args:
            x (Tensor): Time series of shape (N, T, 48)

        Returns:
            Tensor: Predictions of shape (N, T, 1)
        """
        x = self.norm(x)
        x = F.silu(self.conv_start(x))
        for c in self.conv:
            x = x + F.silu(c(x))
        return self.conv_end(x)


class AutoEncoder(nn.Module):
    """Auto-Encoder base class"""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        h_dim: tuple,
        num_preds=1,
        perceptual_loss=False,
        device=None,
    ):
        """Initialize AutoEncoder and internal states

        Args:
            enc (Module): The encoder network
            dec (Module): The decoder network
            h_dim (int): dimension of the latent space
            num_preds (int, optional): Number of timesteps the latent space
                should represent. Defaults to 1.
            perceptual_loss (bool optional): True if another loss than
                reconstruction loss should be used. Defaults to False.
                Other losses are HF reconstruction and adversarial loss.
        """
        super().__init__()
        self.encoder = encoder.to(device)
        """The encoder network"""
        self.decoder = decoder.to(device)
        """The decoder network"""
        self.h_dim = h_dim
        """Latent space dimension"""
        self.num_preds = num_preds
        """Number of timesteps the latent space should represent"""

        self.perceptual_loss = perceptual_loss
        if perceptual_loss:
            self.dis = DailyDiscriminator(3, device=device)
            self.spec = Spectrogram(device=device)

        self.device = device
        """Module's device"""

    def encode(self, x: Tensor) -> Tensor:
        """Encode into latent space"""
        return self.encoder(x)

    def decode(self, h: Tensor) -> Tensor:
        """Decode from latent space"""
        return self.decoder(h)

    def forward(self, x: Tensor) -> Tensor:
        """Encode then decode"""
        x = self.encode(x)
        return self.decode(x)

    def rec_loss(self, pred: Tensor, batch: Tensor) -> Tensor:
        """Reconstruction loss"""
        in_dim = batch.size(2)
        loss = F.mse_loss(pred[:, :, :in_dim], batch[:, :, :in_dim])
        for i in range(1, self.num_preds):
            loss = loss + (
                F.mse_loss(pred[:, i:, i * in_dim : (i + 1) * in_dim], batch[:, :-i])
                / self.num_preds
            )
        return loss

    def variation_loss(self, pred: Tensor, batch: Tensor) -> Tensor:
        """Variation loss"""
        pred = pred.flatten(1)
        batch = batch.flatten(1)
        var_pred = torch.abs(pred[:, :1] - pred[:, 1:])
        var_batch = torch.abs(batch[:, :1] - batch[:, 1:])
        return F.mse_loss(var_pred, var_batch)

    def hf_rec_loss(self, pred: Tensor, batch: Tensor) -> Tensor:
        """High Frequency reconstruction loss"""
        if not self.perceptual_loss:
            return torch.tensor(0)
        # Spectrograms
        batch = self.spec(batch)
        pred = self.spec(pred)
        # High frequencies
        _, batch_hf = bf_hf_split(batch)
        _, pred_hf = bf_hf_split(pred)
        return F.mse_loss(pred_hf, batch_hf)

    def dis_loss(self, pred: Tensor, batch: Tensor) -> tuple[Tensor, Tensor]:
        """Adversarial loss"""
        if not self.perceptual_loss:
            return torch.tensor(0), torch.tensor(0)
        # Patch data = pick some random days
        r_patch = batch.transpose(1, 2)
        f_patch = pred.transpose(1, 2)

        # Discriminator predictions
        f_pred = self.dis(f_patch).squeeze()
        f_pred_d = self.dis(f_patch.detach()).squeeze()
        r_pred = self.dis(r_patch).squeeze()
        r_pred_d = self.dis(r_patch).squeeze()

        # Losses
        # # BPR Loss
        g_loss = F.binary_cross_entropy_with_logits(
            f_pred - r_pred, torch.ones_like(f_pred)
        )
        d_loss = F.binary_cross_entropy_with_logits(
            f_pred_d - r_pred_d, torch.zeros_like(f_pred_d)
        )
        return g_loss, d_loss

    def loss(self, batch: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Computes all losses"""
        pred = self(batch)
        in_dim = batch.size(2)
        rec_loss = self.rec_loss(pred, batch)
        val_loss = self.variation_loss(pred[:, :, :in_dim], batch)
        gen_loss, dis_loss = self.dis_loss(pred[:, :, :in_dim], batch)
        hf_loss = self.hf_rec_loss(pred[:, :, :in_dim], batch)
        return rec_loss, val_loss, gen_loss, dis_loss, hf_loss


class ConvAutoEncoder(AutoEncoder):
    """Time-Series auto-encoder with 2D CNNs as encoder and decoder"""

    def __init__(
        self,
        in_channels: int,
        sample_size: int,
        num_layers: int,
        num_blocks: int,
        hid_channels: int,
        perceptual_loss=False,
        device=None,
        dtype=None,
    ):
        """Initialize internal Module state, shared by both nn.Module
        and ScriptModule.

        Args:
            in_channels (int): Dimension of the input.
            sample_size (int): Number of samples.
            num_layers (int): Number of convolution laer in each block.
            num_blocks (int): Number of blocks
            hid_channels (int): number of channels after the first block.
                It will be doubled after each block.
            perceptual_loss (bool optional): True if another loss than
                reconstruction loss should be used. Defaults to False.
                Other losses are HF reconstruction and adversarial loss.
            device (_type_, optional): Network device. Defaults to None.
            dtype (_type_, optional): Network dtype. Defaults to None.
        """
        self.in_channels = in_channels
        self.sample_size = sample_size
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.hid_channels = hid_channels
        self.device = device
        self.dtype = dtype

        # Compute h_dim
        h_dim_1 = [365, 183, 92, 47, 25, 13, 7, 4][num_blocks]
        h_dim_2 = [48, 48, 48, 48, 25, 13, 7, 4][num_blocks]
        self.h_dim = h_dim_1, h_dim_2, 4

        # Encoder
        encoder = []
        encoder.append(Unsqueeze(1))
        nbc = hid_channels
        for i in range(num_blocks):
            in_dim = 1 if i == 0 else nbc
            out_dim = 4 if i == num_blocks - 1 else nbc * 2
            if i <= 2:
                pad = (1, 0)
                stride = (2, 1)
            elif i == 3:
                pad = (2, 1)
                stride = (2, 2)
            else:
                pad = (1, 1)
                stride = (2, 2)
            encoder.append(
                Conv2dBlock(in_dim, nbc, 3, num_layers, device=device, dtype=dtype)
            )
            if i == 0:
                encoder.append(PeriodicPad2d(pad))
            else:
                encoder.append(nn.ZeroPad2d((pad[1], pad[1], pad[0], pad[0])))
            encoder.append(
                nn.Conv2d(nbc, out_dim, stride, stride, device=device, dtype=dtype)
            )
            nbc = nbc * 2
        encoder.append(nn.Tanh())
        encoder.append(Squeeze(1))

        # Decoder
        decoder = []
        for i in range(num_blocks):
            in_dim = 4 if i == 0 else nbc
            if i <= 2:
                stride = (2, 1)
            else:
                stride = (2, 2)
            decoder.append(
                nn.ConvTranspose2d(
                    in_dim, nbc // 2, 2, stride, device=device, dtype=dtype
                )
            )
            decoder.append(
                Conv2dBlock(
                    nbc // 2,
                    nbc // 2,
                    3,
                    num_layers - int(i == num_blocks - 1),
                    device=device,
                    dtype=dtype,
                )
            )
            nbc = nbc // 2
        decoder.append(nn.Conv2d(nbc, 1, 1, 1, device=device, dtype=dtype))
        decoder.append(AdaptativeCrop2d((sample_size, in_channels)))
        decoder.append(nn.Tanh())
        decoder.append(Squeeze(1))
        encoder = nn.Sequential(*encoder)
        decoder = nn.Sequential(*decoder)
        super().__init__(
            encoder, decoder, self.h_dim, perceptual_loss=perceptual_loss, device=device
        )


class AEExtractor(FeaturesExtractor):
    """Auto-encoder"""

    def __init__(
        self,
        cfg_file: Union[str, os.PathLike],
        params_file: Union[str, os.PathLike],
        device="cpu",
    ):
        cfg = OmegaConf.load(cfg_file)
        params = torch.load(params_file, map_location=device)
        self.model: torch.nn.Module = ConvAutoEncoder(
            in_channels=cfg.dataset.dim, **cfg.model.kwargs, device=device
        )

        self.model.load_state_dict(params)

    @torch.no_grad()
    def fit(self, x, y=None):
        # t = torch.tensor(x, device=self.model.device)
        t = x.to(self.model.device)
        # t = x
        t = t.flatten(1)
        t = torch.split(t, self.model.in_channels)
        t = torch.stack(t, 1)
        # return self.model.encode(t).detach().cpu().numpy()
        t_l = []
        for batch in torch.split(t, 256, 0):
            t_l.append(self.model.encode(batch))
        t = torch.concat(t_l, 0)
        return t

    @torch.no_grad()
    def __call__(self, x, y=None):
        # t = torch.tensor(x, device=self.model.device)
        t = x.to(self.model.device)
        # t = x
        if t.ndim != 3 or t.size(2) != self.model.in_channels:
            t = t.flatten(1)
            t = torch.split(t, self.model.in_channels, dim=1)
            t = torch.stack(t, 1)
        # return self.model.encode(t).detach().cpu().numpy()
        t_l = []
        for batch in torch.split(t, 256, 0):
            t_l.append(self.model.encode(batch))
        t = torch.concat(t_l, 0)
        return t
