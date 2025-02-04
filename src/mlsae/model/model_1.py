from mlsae.model.model import DeepSAE


class DeepSAE1(DeepSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[1],
            sparse_dim_mult=16,
            decoder_dim_mults=[1],
            name="1-1-1",
            enc_dtype="fp32",
            device=device,
            topk=16,
            act_decay_start=8,
            act_decay_end=1e-3,
            act_decay_tau=1500,
            weight_decay=2e-4,
            lr=1e-4,
        )
