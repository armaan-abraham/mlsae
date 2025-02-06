from mlsae.model.model import DeepSAE


class DeepSAE6(DeepSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[1],
            sparse_dim_mult=16,
            decoder_dim_mults=[1],
            name="1-1-3",
            enc_dtype="fp32",
            device=device,
            topk=16,
            act_decay_start=8e-4,
            act_decay_end=8e-7,
            act_decay_tau=3000,
            weight_decay=2e-4,
            lr=1e-4,
        )
