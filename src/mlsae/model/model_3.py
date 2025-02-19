from mlsae.model.model import DeepSAE


class DeepSAE3(DeepSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[],
            sparse_dim_mult=32,
            decoder_dim_mults=[],
            name="0-0_512",
            enc_dtype="fp32",
            device=device,
            topk=512,
            act_decay=1e-3,
            lr=2e-4,
        )