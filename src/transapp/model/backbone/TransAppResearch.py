import torch
import torch.nn as nn

from dataclasses import dataclass

from module.transformer import EncoderLayer, Transpose
from module.embedding import DilatedBlock
from module.positional_encoding import PositionalEncoding1D, LearnablePositionalEncoding1D


@dataclass
class TransAppConfig:
    max_len: int = 1024
    c_in: int = 1
    mode: str = "classif"
    n_embed_blocks: int = 1
    encoding_type: str = "noencoding"
    n_encoder_layers: int = 3
    kernel_size: int = 5
    d_model: int = 64
    pffn_ratio: int = 2
    n_head: int = 4
    prenorm: bool = True
    norm: str = "LayerNorm"
    activation: str = 'gelu'
    store_att: bool = False
    attn_dp_rate: float = 0.2
    head_dp_rate: float = 0.1
    dp_rate: float = 0.2
    att_param: dict = None
    c_reconstruct: int = 1
    apply_gap: bool = False
    nb_class: int = 2

    def __post_init__(self):
        if self.att_param is None:
            self.att_param = {'attenc_mask_diag': True, 'attenc_mask_flag': False, 'learnable_scale_enc': False}
            

class TransAppResearch(nn.Module):
    def __init__(self, config: TransAppConfig):
        super().__init__()

        self.c_in = config.c_in
        self.d_model = config.d_model
        self.mode = config.mode
        self.nb_class = config.nb_class

        # ============ Dilated Conv Embedding ============#
        layers = []
        for i in range(config.n_embed_blocks):
            layers.append(DilatedBlock(c_in=config.c_in if i == 0 else config.d_model,
                                       c_out=config.d_model, kernel_size=config.kernel_size))
        layers.append(Transpose(1, 2))
        self.EmbedBlock = torch.nn.Sequential(*layers)

        # ============ Encoding ============#
        if config.encoding_type == 'learnable':
            self.PosEncoding = LearnablePositionalEncoding1D(config.d_model, max_len=config.max_len)
        elif config.encoding_type == 'fixed':
            self.PosEncoding = PositionalEncoding1D(config.d_model)
        elif config.encoding_type == 'noencoding':
            self.PosEncoding = None
        else:
            raise ValueError(f'Type of encoding {config.encoding_type} unknown, only "learnable", "fixed" or "noencoding" supported.')

        # ============ Encoder ============#
        layers = []
        for i in range(config.n_encoder_layers):
            layers.append(EncoderLayer(config.d_model, config.d_model * config.pffn_ratio, config.n_head,
                                       dp_rate=config.dp_rate, attn_dp_rate=config.attn_dp_rate,
                                       att_mask_diag=config.att_param.get('attenc_mask_diag', True),
                                       att_mask_flag=config.att_param.get('attenc_mask_flag', False),
                                       learnable_scale=config.att_param.get('learnable_scale_enc', False),
                                       store_att=config.store_att, norm=config.norm, prenorm=config.prenorm, activation=config.activation))
        layers.append(nn.LayerNorm(config.d_model))
        self.EncoderBlock = torch.nn.Sequential(*layers)

        # ============ Pretraining Head ============#
        layers = []
        layers.append(nn.Linear(config.d_model, config.c_reconstruct, bias=True))
        layers.append(nn.Dropout(config.head_dp_rate))
        self.PredHead = torch.nn.Sequential(*layers)

        # ============ Classif Head ============#
        layers = []
        if config.apply_gap:
            layers.append(Transpose(1, 2))
            layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten(start_dim=1))
        if config.apply_gap:
            layers.append(nn.Linear(config.d_model, config.nb_class, bias=True))
        else:
            layers.append(nn.Linear(config.max_len * config.d_model, config.nb_class, bias=True))
        layers.append(nn.Dropout(config.head_dp_rate))
        self.ClassifHead = torch.nn.Sequential(*layers)

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def freeze_params(self, model_part, rq_grad=False):
        for _, child in model_part.named_children():
            for param in child.parameters():
                param.requires_grad = rq_grad
            self.freeze_params(child)

    def forward(self, x) -> torch.Tensor:
        # Dilated Conv Embedding Block
        x = self.EmbedBlock(x)

        # Add Pos. Encoding (if any)
        if self.PosEncoding is not None:
            x = x + self.PosEncoding(x)

        # Forward Encoder
        x = self.EncoderBlock(x)

        # Forward Head
        if self.mode == "pretraining":
            x = self.PredHead(x).permute(0, 2, 1)
        else:
            x = self.ClassifHead(x)

        return x