from mlsae.model.model import DeepSAE


class DeepSAE17(DeepSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[1, 1],
            sparse_dim_mult=8,
            decoder_dim_mults=[1, 1],
            name="1-1-1-1-0",
            enc_dtype="fp32",
            device=device,
            topk=4,
            act_decay=0,
            lr=2e-4,
        )
