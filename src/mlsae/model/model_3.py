import torch.nn as nn
from einops import einsum

from mlsae.model.model import DeepSAE, TopKActivation


class DeepSAE3(DeepSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[],
            sparse_dim_mult=16,
            decoder_dim_mults=[],
            name="0-0-canon-bias",
            enc_dtype="fp32",
            device=device,
            topk=16,
            act_l2_coeff=0.2,
            weight_decay=1e-4,
            lr=1e-4,
        )

    def _forward(self, x):
        W_e = self.sparse_encoder_block[0].weight
        b_e = self.sparse_encoder_block[0].bias
        W_d = self.decoder_blocks[0].weight
        b_d = self.decoder_blocks[0].bias

        x_hat = x - b_d
        feature_acts = nn.ReLU()(
            einsum(x_hat, W_e, "batch d_in, d_out d_in -> batch d_out") + b_e
        )
        feature_acts = TopKActivation(self.topk)(feature_acts)
        reconstructed = einsum(feature_acts, W_d, "batch d_in, d_out d_in -> batch d_out") + b_d

        mse_loss = (reconstructed.float() - x.float()).pow(2).mean()
        l2_loss = (feature_acts**2).mean() * (self.act_l2_coeff / self.topk)

        loss = mse_loss + l2_loss

        return loss, l2_loss, mse_loss, feature_acts, reconstructed
