from mlsae.model.model import DeepSAE
import torch


class DeepSAE2(DeepSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[1],
            sparse_dim_mult=8,
            decoder_dim_mults=[1],
            name="1-1-2",
            enc_dtype="fp32",
            device=device,
            topk=4,
            act_decay=1e-4,
            lr=2e-4,
        )

    @torch.no_grad()
    def process_gradients(self):
        self.make_decoder_weights_and_grad_unit_norm()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.25)
