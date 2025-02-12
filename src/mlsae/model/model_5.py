from mlsae.model.model_1 import DeepSAENoScale
import torch


class DeepSAE5(DeepSAENoScale):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[1],
            sparse_dim_mult=8,
            decoder_dim_mults=[1],
            name="1-1-5",
            enc_dtype="fp32",
            device=device,
            topk=4,
            act_decay=1e-1,
            lr=2e-4,
        )

