import torch

from mlsae.model.model import DeepSAE


class DeepSAEClip(DeepSAE):
    def process_gradients(self):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.make_decoder_weights_and_grad_unit_norm()


class DeepSAE1(DeepSAEClip):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[1],
            sparse_dim_mult=16,
            decoder_dim_mults=[1],
            name="1-1-act-weight-L2-clip",
            enc_dtype="fp32",
            device=device,
            topk=16,
            act_l2_coeff=0.2,
            weight_decay=2e-4,
            lr=1e-4,
        )
