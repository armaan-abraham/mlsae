import torch
import torch.nn as nn

from mlsae.model.model import DeepSAE, TopKActivation


class DeepSAEReLUEncoder(DeepSAE):
    def _init_encoder_params(self):
        self.dense_encoder_blocks = torch.nn.ModuleList()
        in_dim = self.act_size

        for dim in self.encoder_dims:
            linear_layer = self._create_linear_layer(
                in_dim, dim, apply_weight_decay=True
            )
            self.dense_encoder_blocks.append(
                torch.nn.Sequential(linear_layer, nn.ReLU(), nn.LayerNorm(dim))
            )
            in_dim = dim

        self.sparse_encoder_block = torch.nn.Sequential(
            self._create_linear_layer(
                in_dim, self.sparse_dim, apply_weight_decay=False
            ),
            nn.ReLU(),
            TopKActivation(self.topk),
        )


class DeepSAE7(DeepSAEReLUEncoder):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[1],
            sparse_dim_mult=16,
            decoder_dim_mults=[1],
            name="1-1-relu-encoder",
            enc_dtype="fp32",
            device=device,
            topk=16,
            act_l2_coeff=0.2,
            weight_decay=2e-4,
            lr=1e-4,
        )
