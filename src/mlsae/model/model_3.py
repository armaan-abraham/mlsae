from mlsae.model.model import DeepSAE, TopKActivation

class L2ActDecaySAE(DeepSAE):
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
        act_mag = feature_acts.pow(2).mean()
        act_mag_loss = act_mag * self.act_decay

        loss = mse_loss + act_mag_loss

        return loss, act_mag, mse_loss, feature_acts, reconstructed

class DeepSAE3(L2ActDecaySAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[1],
            sparse_dim_mult=16,
            decoder_dim_mults=[1],
            name="1-1-3",
            enc_dtype="fp32",
            device=device,
            topk=16,
            act_decay=1e2,
            weight_decay=1e-4,
            lr=2e-4,
        )
