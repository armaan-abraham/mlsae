from mlsae.model.model import DeepSAE


class DeepSAE6(DeepSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[1],
            sparse_dim_mult=16,
            decoder_dim_mults=[1],
            name="1-1-6",
            enc_dtype="fp32",
            device=device,
            topk=16,
            act_decay=1e-4,
            weight_decay=1e-4,
            lr=2e-4,
        )
