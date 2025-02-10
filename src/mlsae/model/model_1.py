from mlsae.model.model import DeepSAE, TopKActivation
import torch
import torch.nn as nn

class ScalarSAE(DeepSAE):
    def _init_encoder_params(self):
        self.dense_encoder_blocks = torch.nn.ModuleList()
        in_dim = self.act_size

        for dim in self.encoder_dims:
            linear_layer = self._create_linear_layer(
                in_dim, dim
            )
            self.dense_encoder_blocks.append(
                torch.nn.Sequential(linear_layer, nn.ReLU())
            )
            in_dim = dim
        
        self.sparse_scalar = nn.Parameter(torch.ones(1))

        self.sparse_encoder_block = torch.nn.Sequential(
            self._create_linear_layer(
                in_dim, self.sparse_dim
            ),
            nn.ReLU(),
            TopKActivation(self.topk),
        )

    def _forward(self, x):
        # Encode
        if self.encoder_dims:
            resid = x
            for block in self.dense_encoder_blocks:
                resid = block(resid)
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
    
class DeepSAE1(ScalarSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[1],
            sparse_dim_mult=8,
            decoder_dim_mults=[1],
            name="1-1-1",
            enc_dtype="fp32",
            device=device,
            topk=4,
            act_decay=0,
            lr=2e-4,
        )
    