from mlsae.model.model import DeepSAE


class DeepSAE2(DeepSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[],
            sparse_dim_mult=8,
            decoder_dim_mults=[],
            name="SAE2",
            device=device,
            lr=2e-4,
            act_decay=0,
            topk=128,
        )
