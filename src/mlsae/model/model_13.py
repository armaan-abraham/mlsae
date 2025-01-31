from mlsae.model.model import DeepSAE


class DeepSAE13(DeepSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[],
            sparse_dim_mult=16,
            decoder_dim_mults=[],
            name="0-0-resample",
            enc_dtype="fp32",
            device=device,
            topk=16,
            act_l2_coeff=0,
            weight_decay=1e-4,
            lr=1e-4,
        )

    def should_resample_sparse_features(self, idx):
        return True
