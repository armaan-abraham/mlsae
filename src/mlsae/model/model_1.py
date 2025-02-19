from mlsae.model.model import DeepSAE


class DeepSAE1(DeepSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[2],
            sparse_dim_mult=16,
            decoder_dim_mults=[2],
            name="2-2_128",
            enc_dtype="fp32",
            device=device,
            topk=128,
            act_decay=1e-3,
            lr=2e-4,
        )