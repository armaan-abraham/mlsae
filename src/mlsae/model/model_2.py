from mlsae.model.model import DeepSAE


class DeepSAE2(DeepSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[],
            sparse_dim_mult=16,
            decoder_dim_mults=[],
            name="0-0-wd",
            enc_dtype="fp32",
            device=device,
            topk=16,
            act_l2_coeff=0.0,
            weight_decay=1e-3,
            lr=1e-4,
        )
