import torch
import torch.nn as nn

from mlsae.model.model import DeepSAE


class DeepSAE0(DeepSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[1],
            sparse_dim_mult=16,
            decoder_dim_mults=[1],
            name="1-1-unit-first-layer",
            enc_dtype="fp32",
            device=device,
            topk=16,
            act_l2_coeff=0.2,
            weight_decay=2e-4,
            lr=1e-4,
        )

    def _init_decoder_params(self):
        self.decoder_blocks = nn.ModuleList()

        self.decoder_blocks.append(
            self._create_linear_layer(
                self.sparse_dim, self.decoder_dims[0], apply_weight_decay=False
            )
        )
        self.decoder_blocks.append(nn.ReLU())

        self.decoder_blocks.append(
            self._create_linear_layer(
                self.decoder_dims[0], self.act_size, apply_weight_decay=True
            )
        )

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        w = self.decoder_blocks[0].weight
        w_normed = w / w.norm(dim=-1, keepdim=True)
        if w.grad is not None:
            w_dec_grad_proj = (w.grad * w_normed).sum(-1, keepdim=True) * w_normed
            w.grad -= w_dec_grad_proj
        w.data = w_normed
