from mlsae.model.model import DeepSAE


class DeepSAE16(DeepSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[1],
            sparse_dim_mult=8,
            decoder_dim_mults=[1],
            name="1-1-6",
            enc_dtype="fp32",
            device=device,
            topk=4,
            act_decay=0,
            lr=2e-4,
        )

    def _forward(self, x):
        # Encode
        if self.encoder_dims:
            resid = x
            for block in self.dense_encoder_blocks:
                resid = block(resid)
            resid += x
        else:
            resid = x
        
        resid_scaled = resid * self.sparse_scalar

        resid = feature_acts = self.sparse_encoder_block(resid_scaled)
        assert ((feature_acts == 0).float().sum(dim=-1) >= (self.sparse_dim - self.topk)).all()

        for block in self.decoder_blocks:
            resid = block(resid)
        
        resid_scaled = resid / self.sparse_scalar

        reconstructed = resid_scaled

        # MSE reconstruction loss
        mse_loss = (reconstructed.float() - x.float()).pow(2).mean()
        act_mag = feature_acts.pow(2).mean()
        act_mag_loss = act_mag * self.act_decay

        loss = mse_loss + act_mag_loss

        return loss, act_mag, mse_loss, feature_acts, reconstructed