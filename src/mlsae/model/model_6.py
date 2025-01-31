from mlsae.model.model_1 import DeepSAEClip


class DeepSAE6(DeepSAEClip):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[],
            sparse_dim_mult=16,
            decoder_dim_mults=[],
            name="0-0-clip",
            enc_dtype="fp32",
            device=device,
            topk=16,
            act_l2_coeff=0.2,
            weight_decay=1e-4,
            lr=1e-4,
        )
