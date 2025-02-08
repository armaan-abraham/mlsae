from mlsae.model.model import DeepSAE, TopKActivation
import torch

class InflateSAE(DeepSAE):
    def _forward(self, x):
        # Encode
        if self.encoder_dims:
            resid = x
            for block in self.dense_encoder_blocks:
                resid = block(resid)
            resid += x
        else:
            resid = x

        resid = feature_acts_full = self.sparse_encoder_block(resid)
        feature_acts = TopKActivation(self.topk)(feature_acts_full)

        for block in self.decoder_blocks:
            resid = block(resid)

        reconstructed = resid

        # MSE reconstruction loss
        mse_loss = (reconstructed.float() - x.float()).pow(2).mean()

        # Regularization
        deflate_act_mag = torch.abs(feature_acts).mean()
        deflate_loss = deflate_act_mag * self.act_decay * 1.5
        inflate_act_mag = torch.abs(feature_acts_full).mean()
        inflate_loss = -inflate_act_mag * self.act_decay
        act_mag_loss = deflate_loss + inflate_loss

        loss = mse_loss + act_mag_loss

        return loss, deflate_act_mag, mse_loss, feature_acts, reconstructed

class DeepSAE0(InflateSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[1],
            sparse_dim_mult=16,
            decoder_dim_mults=[1],
            name="1-1-0",
            enc_dtype="fp32",
            device=device,
            topk=16,
            act_decay=1e-2,
            weight_decay=1e-4,
            lr=2e-4,
        )
