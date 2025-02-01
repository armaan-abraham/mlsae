import torch
import torch.nn as nn
from einops import einsum

from mlsae.model.model import DeepSAE, TopKActivation


class DeepSAEStandardize(DeepSAE):
    def _forward(self, x):
        if self.encoder_dims:
            resid = x
            for block in self.dense_encoder_blocks:
                resid = block(resid)
            resid += x
        else:
            resid = x

        resid = feature_acts = self.sparse_encoder_block(resid)

        for block in self.decoder_blocks[:-1]:
            resid = block(resid)

        final_layer = self.decoder_blocks[-1]
        final_layer_weight_unit = final_layer.weight / final_layer.weight.norm(
            dim=1, keepdim=True
        )
        reconstructed = (
            einsum(
                resid, final_layer_weight_unit, "batch d_in, d_out d_in -> batch d_out"
            )
            + final_layer.bias
        )

        # MSE reconstruction loss
        mse_loss = (reconstructed.float() - x.float()).pow(2).mean()
        l2_loss = (feature_acts**2).mean() * (self.act_l2_coeff / self.topk)

        loss = mse_loss + l2_loss

        return loss, l2_loss, mse_loss, feature_acts, reconstructed


class DeepSAE2(DeepSAEStandardize):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[1],
            sparse_dim_mult=16,
            decoder_dim_mults=[1],
            name="1-1-standardize",
            enc_dtype="fp32",
            device=device,
            topk=16,
            act_l2_coeff=0.2,
            weight_decay=2e-4,
            lr=1e-4,
        )
