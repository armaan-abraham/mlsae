from mlsae.model.model import DeepSAE
import torch.nn as nn


class DeepSAE5(DeepSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[1],
            sparse_dim_mult=16,
            decoder_dim_mults=[1],
            name="1-1-wd-td",
            enc_dtype="fp32",
            device=device,
            topk=16,
            act_l2_coeff=0.0,
            weight_decay=1e-3,
            lr=1e-4,
        )

    def _init_decoder_params(self):
        self.decoder_blocks = nn.ModuleList()
        in_dim = self.sparse_dim

        for dim in self.decoder_dims:
            linear_layer = self._create_linear_layer(
                in_dim, dim, apply_weight_decay=True
            )
            self.decoder_blocks.append(linear_layer)
            self.decoder_blocks.append(nn.ReLU())
            in_dim = dim

        self.decoder_blocks.append(
            self._create_linear_layer(in_dim, self.act_size, apply_weight_decay=True)
        )
