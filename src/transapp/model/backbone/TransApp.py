import torch
import torch.nn as nn

from dataclasses import dataclass

from module.transformer import EncoderLayer, Transpose
from module.embedding import DilatedBlock


@dataclass
class TransAppConfig:
    # Config Large
    n_encoder_layers: int = 5
    d_model: int = 96

    c_in: int = 5
    mode: str = "classif"
    n_embed_blocks: int = 1
    kernel_size: int = 5
    pffn_ratio: int = 2
    n_head: int = 4
    prenorm: bool = True
    norm: str = "LayerNorm"
    activation: str = 'gelu'
    store_att: bool = False
    attn_dp_rate: float = 0.2
    head_dp_rate: float = 0.2
    dp_rate: float = 0.1
    att_param: dict = None
    c_reconstruct: int = 1
    nb_class: int = 2

    def __post_init__(self):
        if self.att_param is None:
            self.att_param = {'attenc_mask_diag': True, 'attenc_mask_flag': False, 'learnable_scale_enc': False}


class TransAppBase(nn.Module):
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
        self.PosEncoding = None

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
        self.PredHead = torch.nn.Sequential(*[
            nn.Linear(config.d_model, config.c_reconstruct, bias=True),
            nn.Dropout(config.head_dp_rate)]
            )

        # ============ Classif Head ============#
        self.ClassifHead = torch.nn.Sequential(*[
            Transpose(1, 2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(config.d_model, config.nb_class, bias=True),
            nn.Dropout(config.head_dp_rate)
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
        inst_std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + 1e-6).detach() # Not biased norm as such sklearn StandardScaler
        return (x - inst_mean) / inst_std, inst_mean, inst_std
    
    def embed(self, x, encoding):
        # x and encoding as [B, C, seq_length] (C=1 and C=n_exo_var, respectively)
        # compute sin & cos
        encoding = encoding.contiguous()
        sin_enc = encoding.sin()
        cos_enc = encoding.cos()
        
        # interleave sin & cos along the channel axis
        # stack → [B, C, 2, L] → flatten (merge C and 2) → [B, 2*C, seq_length]
        encoding = (
            torch.stack((sin_enc, cos_enc), dim=2)   # add a tiny axis
                .flatten(1, 2)                     # interleave & double channels
        )
        # concat
        x = torch.cat([x, encoding], dim=1)  # [batch, 1+2*n_exo_var, seq_length]
        x = self.EmbedBlock(x) # [batch, seq_length, d]
        return x
    

class TransAppPretrain(TransAppBase):
    def __init__(self, config: TransAppConfig):
        super().__init__(config)

    def get_mask(self, device, N, L):
        len_keep = int(L * (1 - 0.3))
        noise = torch.rand(N, L, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        mask = torch.ones([N, L], device=device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask, ids_keep, ids_restore
    
    def forward(self, x, encoding) -> torch.Tensor:
        # x and encoding as [B, C, seq_length] (C=1 and C=n_exo_var, respectively)

        # Instance Norm and clone 
        x, _, _ = self.instance_norm(x)
        x_input_norm = x.clone()

        mask, _, _ = self.get_mask(x.device, x.size(0), x.size(-1))
        x = x * (~mask.bool()).int().unsqueeze(1)

        # Embedding
        x = self.embed(x, encoding)

        # Forward Encoder
        x = self.EncoderBlock(x)

        # Forward Head
        x = self.PredHead(x).permute(0, 2, 1)

        x_input_norm = x_input_norm * mask.unsqueeze(1)
        x = x * mask.unsqueeze(1)
        loss = nn.L1Loss()(x, x_input_norm)
        return x, loss
    

class TransAppClassif(TransAppBase):
    def __init__(self, config: TransAppConfig):
        super().__init__(config)
    
    def forward(self, x, encoding) -> torch.Tensor:
        # x and encoding as [B, C, seq_length] (C=1 and C=n_exo_var, respectively)

        # Embedding
        x, _, _ = self.instance_norm(x)
        x = self.embed(x, encoding)

        # Forward Encoder
        x = self.EncoderBlock(x)

        # Forward Head
        x = self.ClassifHead(x)

        return x


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cdc = torch.randn((3, 1, 1024)).to(device)
    encoding = torch.randn((3, 2, 1024)).to(device) # days and hours exo variable

    print("Check TransApp Pretraining")
    model = TransAppPretrain(TransAppConfig()).to(device)
    out, _ = model(cdc, encoding)
    del model
    print(out.shape)

    print("Check TransApp Classif")
    model = TransAppClassif(TransAppConfig()).to(device)
    out = model(cdc, encoding)
    del model
    print(out.shape)