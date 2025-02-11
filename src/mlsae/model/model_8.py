from mlsae.model.model import DeepSAE
import torch.nn as nn

class DeepSAE8(DeepSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[],
            sparse_dim_mult=8,
            decoder_dim_mults=[],
            name="0-0-1",
            enc_dtype="fp32",
            device=device,
            topk=4,
            act_decay=1e-4,
            lr=2e-4,
        )
