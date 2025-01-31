import torch
import torch.nn as nn
from einops import einsum

from mlsae.model.model import DeepSAE, TopKActivation


class DeepSAENormalizeInputs(DeepSAE):
    def _forward(self, x):
        x = x - x.mean(dim=1, keepdim=True)
        x = x / torch.linalg.norm(x, dim=1, keepdim=True)
        return super()._forward(x)


class DeepSAE10(DeepSAENormalizeInputs):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[1],
            sparse_dim_mult=16,
            decoder_dim_mults=[1],
            name="1-1-normalize-inputs",
            enc_dtype="fp32",
            device=device,
            topk=16,
            act_l2_coeff=0.2,
            weight_decay=2e-4,
            lr=1e-4,
        )
