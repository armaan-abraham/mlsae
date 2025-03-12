from mlsae.model.model import ExperimentSAEBase, TopKActivation
import torch
import math

class ResSAE(ExperimentSAEBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.act_squeeze == 0
        assert all(dim == self.act_size for dim in self.encoder_dims)
        assert all(dim == self.act_size for dim in self.decoder_dims)

    def _forward(self, x, iteration=None):
        # Encode
        resid = x

        if self.encoder_dims:

            for block in self.dense_encoder_blocks:
                resid = block(resid) + resid

            # Dead neuron counts are very sensitive to initial scaling. I have
            # found that dividing by the L2 norm of the input activations helps
            # for shallow SAEs. This division makes it so that we initially
            # divide by the L2 norm in combination with the layernorm (as std
            # and L2 norm are proportional).
            resid = resid / torch.sqrt(torch.tensor(resid.shape[-1]))
        
        # Continue with topk activation and the rest of the network
        feature_acts = self.sparse_encoder_block(resid)
        resid = feature_acts
        
        assert (
            (feature_acts == 0).float().sum(dim=-1) >= (self.sparse_dim - self.topk)
        ).all()

        resid = self.decoder_blocks[0](resid)

        for block in self.decoder_blocks[1:-1]:
            resid = block(resid) + resid
        
        resid = self.decoder_blocks[-1](resid)

        reconstructed = resid

        # MSE reconstruction loss
        mse_loss = (reconstructed.float() - x.float()).pow(2).mean()
        
        loss = mse_loss

        return {
            "loss": loss,
            "mse_loss": mse_loss,
            "feature_acts": feature_acts,
            "reconstructed": reconstructed,
        }

class ExperimentSAERes1x1(ResSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[1],
            sparse_dim_mult=32,
            decoder_dim_mults=[1],
            device=device,
            topk_init=16,
            topk_final=16,
            topk_decay_iter=2000,
            act_squeeze=0,
            optimizer_type="sparse_adam",
            optimizer_config={
                "lr": 4e-4,
            }
        )

class ExperimentSAERes1x1x1x1(ResSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[1, 1],
            sparse_dim_mult=32,
            decoder_dim_mults=[1, 1],
            device=device,
            topk_init=16,
            topk_final=16,
            topk_decay_iter=2000,
            act_squeeze=0,
            optimizer_type="sparse_adam",
            optimizer_config={
                "lr": 4e-4,
            }
        )