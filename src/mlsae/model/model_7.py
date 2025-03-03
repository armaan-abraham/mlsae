from mlsae.model.model import DeepSAE


class DeepSAE7(DeepSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[],
            sparse_dim_mult=8,
            decoder_dim_mults=[],
            name="SAE55",
            device=device,
            lr=2e-4,
            topk=55,
            act_decay=0,
        )
