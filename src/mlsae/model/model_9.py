import torch

from mlsae.model.model import DeepSAE


class DeepSAE9(DeepSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[1],
            sparse_dim_mult=16,
            decoder_dim_mults=[1],
            name="1-1-act-weight-L2-lg",
            enc_dtype="fp32",
            device=device,
            topk=16,
            act_l2_coeff=0.3,
            weight_decay=6e-4,
            lr=1e-4,
        )
