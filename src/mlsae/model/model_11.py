from mlsae.model.model_10 import DeepSAENormalizeInputs


class DeepSAE11(DeepSAENormalizeInputs):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[1],
            sparse_dim_mult=16,
            decoder_dim_mults=[],
            name="1-0-normalize-inputs",
            enc_dtype="fp32",
            device=device,
            topk=16,
            act_l2_coeff=0.2,
            weight_decay=1e-4,
            lr=1e-4,
        )
