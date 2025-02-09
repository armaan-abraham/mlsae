from mlsae.model.model_3 import ReLUEncoderSAE

class DeepSAE4(ReLUEncoderSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[1],
            sparse_dim_mult=8,
            decoder_dim_mults=[1],
            name="1-1-4",
            enc_dtype="fp32",
            device=device,
            topk=4,
            act_decay=1e-5,
            lr=2e-4,
        )
    
