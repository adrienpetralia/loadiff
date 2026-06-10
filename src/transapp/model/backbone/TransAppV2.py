import torch
import torch.nn as nn
import numpy as np

from dataclasses import dataclass, field

from module.transformer import EncoderLayer
from module.embedding import DilatedBlock


@dataclass
class TransAppV2Config:
    c_in: int = 1
    n_exogene_var: int = 6  # e.g., minute, hour, dayofweek, dayofmonth, dayofyear, month
    nb_class: int = 2
    kernel_size: int = 3
    kernel_size_pt_head: int = 3
    dilations: list = field(default_factory=lambda: [1, 2, 4, 8])
    conv_bias: bool = True
    n_encoder_layers: int = 3
    d_model: int = 128
    dp_rate: float = 0.1
    activation: str = 'gelu'
    pffn_ratio: int = 4
    n_head: int = 8
    prenorm: bool = True
    norm: str = "LayerNorm"
    attn_dp_rate: float = 0.2
    att_param: dict = field(default_factory=dict)
    masking_type: str = 'subseq'
    mask_ratio: float = 0.3
    mask_mean_length: int = 24
    output_mask: bool = False
    loss_pretraining: nn.Module = field(default_factory=lambda: nn.SmoothL1Loss())

    def __post_init__(self):
        if not self.att_param:
            self.att_param = {
                'attenc_mask_diag': True,
                'attenc_mask_flag': False,
                'learnable_scale_enc': False
            }


class TransAppV2Base(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.loss_pretraining = config.loss_pretraining
        self.mask_ratio = config.mask_ratio
        self.mask_mean_length = config.mask_mean_length
        self.masking_type = config.masking_type
        self.output_mask = config.output_mask

        assert config.d_model % 4 == 0, 'd_model must be divisible by 4.'

        d_model_ = int(3 * config.d_model // 4)

        # ===== Embedding =====
        self.EmbedBlock = DilatedBlock(
            c_in=config.c_in,
            c_out=d_model_,
            kernel_size=config.kernel_size,
            dilation_list=config.dilations,
            bias=config.conv_bias
        )
        self.ProjEmbedding = nn.Sequential(
            nn.Linear(config.n_exogene_var * 2, config.d_model // 4), 
            nn.ReLU()
        )

        self.ProjStats = nn.Linear(2, config.d_model)

        # ===== Transformer Encoder =====
        layers = [
            EncoderLayer(
                config.d_model,
                config.d_model * config.pffn_ratio,
                config.n_head,
                dp_rate=config.dp_rate,
                attn_dp_rate=config.attn_dp_rate,
                att_mask_diag=config.att_param['attenc_mask_diag'],
                att_mask_flag=config.att_param['attenc_mask_flag'],
                learnable_scale=config.att_param['learnable_scale_enc'],
                norm=config.norm,
                prenorm=config.prenorm,
                activation=config.activation
            ) for _ in range(config.n_encoder_layers)
        ]
        layers.append(nn.LayerNorm(config.d_model))
        self.EncoderBlock = nn.Sequential(*layers)

        # ===== Heads =====
        self.PredHead = nn.Conv1d(
            in_channels=config.d_model,
            out_channels=config.c_in,
            kernel_size=config.kernel_size_pt_head,
            padding=config.kernel_size_pt_head // 2
        )

        self.ClassifHead = torch.nn.Sequential(*[
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(config.d_model, config.nb_class, bias=True),
            nn.Dropout(config.dp_rate)
            ]
        )

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def instance_norm(self, x):
        inst_mean = x.mean(dim=-1, keepdim=True).detach()
        inst_std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + 1e-6).detach()
        return (x - inst_mean) / inst_std, inst_mean, inst_std

    def embed(self, x, encoding):
        x = self.EmbedBlock(x) # [batch, 3 * (d//4), seq_length]
        encoding = encoding.permute(0, 2, 1)
        encoding = self.ProjEmbedding(torch.cat([encoding.sin(), encoding.cos()], dim=-1)) # [batch, seq_length, dim_pe]
        x = torch.cat([x.permute(0, 2, 1), encoding], dim=-1)  # [batch, seq_length, d]
        return x


class TransAppV2Pretrain(TransAppV2Base):
    def __init__(self, config):
        super().__init__(config)

    def get_mask(self, device, N, L):
        if self.masking_type == 'subseq':
            mask = np.ones(L, dtype=bool)
            p_m = 1 / self.mask_mean_length
            p_u = p_m * self.mask_ratio / (1 - self.mask_ratio)
            p = [p_m, p_u]
            state = int(np.random.rand() > self.mask_ratio)
            for i in range(L):
                mask[i] = state
                if np.random.rand() < p[state]:
                    state = 1 - state
            mask = torch.tensor(mask).int().to(device)
            ids_keep = torch.nonzero(mask, as_tuple=True)[0]
            ids_restore = torch.argsort(torch.cat((ids_keep, torch.nonzero(~mask.bool(), as_tuple=True)[0])))
            return (~mask.bool()).int().unsqueeze(0).repeat(N, 1), ids_keep.unsqueeze(0).repeat(N, 1), ids_restore.unsqueeze(0).repeat(N, 1)
        else:
            len_keep = int(L * (1 - self.mask_ratio))
            noise = torch.rand(N, L, device=device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :len_keep]
            mask = torch.ones([N, L], device=device)
            mask[:, :len_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)
            return mask, ids_keep, ids_restore

    def forward(self, x, encoding):
        x_input = x.clone()
        x, mean, std = self.instance_norm(x)

        mask, _, _ = self.get_mask(x.device, x.size(0), x.size(-1))
        x = x * (~mask.bool()).int().unsqueeze(1)

        x = self.embed(x, encoding)
        stats_token = self.ProjStats(torch.cat([mean, std], dim=1).permute(0, 2, 1))
        x = torch.cat([stats_token, x], dim=1)
        x = self.EncoderBlock(x)
        x = x[:, 1:] # Remove stats token

        x = self.PredHead(x.permute(0, 2, 1))
        x = x * std + mean
        x_input = x_input * mask.unsqueeze(1)
        x = x * mask.unsqueeze(1)
        loss = self.loss_pretraining(x, x_input)
        return (x, loss, mask) if self.output_mask else (x, loss)


class TransAppV2Classif(TransAppV2Base):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, x, encoding):
        x, mean, std = self.instance_norm(x)
        x = self.embed(x, encoding)

        stats_token = self.ProjStats(torch.cat([mean, std], dim=1).permute(0, 2, 1))
        x = torch.cat([stats_token, x], dim=1)

        x = self.EncoderBlock(x)
        x = self.ClassifHead(x.permute(0, 2, 1))

        return x


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_exogene_var = 6

    cdc = torch.randn((3, 1, 1024)).to(device)
    encoding = torch.randn((3, 6, 1024)).to(device)

    print("Check TransApp V2 Pretraining")
    model = TransAppV2Pretrain(TransAppV2Config(n_exogene_var=n_exogene_var)).to(device)
    out, _ = model(cdc, encoding)
    del model
    print(out.shape)

    print("Check TransApp V2 Classif")
    model = TransAppV2Classif(TransAppV2Config(n_exogene_var=n_exogene_var)).to(device)
    out = model(cdc, encoding)
    del model
    print(out.shape)