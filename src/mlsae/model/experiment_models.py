from mlsae.model.model import ExperimentSAEBase, TopKActivation
import torch
import torch.nn as nn


class ActSqueezeSAE(ExperimentSAEBase):
    def __init__(self, act_size: int, *args, act_squeeze: float = 1, **kwargs):
        assert kwargs.get("act_decay", 0) == 0, "ActInflationSAE does not support act_decay"
        super().__init__(
            act_size,
            *args,
            **kwargs
        )
        self.act_squeeze = act_squeeze

    def _forward(self, x, iteration=None):
        # Encode
        resid = x
        if self.encoder_dims:
            for block in self.dense_encoder_blocks:
                resid = block(resid)
        
        # Access pre-topk activations from sparse_encoder_block
        # sparse_encoder_block is Sequential(Linear, ReLU, TopKActivation)
        linear_out = self.sparse_encoder_block[0](resid)  # Linear output
        relu_out = self.sparse_encoder_block[1](linear_out)  # After ReLU, before TopK
        
        # Get pre-topk activations for inflation loss
        pre_topk_acts = relu_out
        
        # Mean activation across batch for each feature
        mean_acts_per_feature = pre_topk_acts.mean(dim=0)
        
        mean_acts_std = mean_acts_per_feature.std()
        
        act_squeeze_loss = mean_acts_std * self.act_squeeze
        
        # Continue with topk activation and the rest of the network
        feature_acts = self.sparse_encoder_block[2](relu_out)  # Apply TopK
        resid = feature_acts
        
        assert (
            (feature_acts == 0).float().sum(dim=-1) >= (self.sparse_dim - self.topk)
        ).all()

        for block in self.decoder_blocks:
            resid = block(resid)

        reconstructed = resid

        # MSE reconstruction loss
        mse_loss = (reconstructed.float() - x.float()).pow(2).mean()
        
        # No act_mag_loss since act_decay=0
        loss = mse_loss + act_squeeze_loss

        return {
            "loss": loss,
            "mse_loss": mse_loss,
            "feature_acts": feature_acts,
            "reconstructed": reconstructed,
            "act_squeeze_loss": act_squeeze_loss,
        }

class ExperimentSAE2x2Layernorm(ExperimentSAEBase):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[2],
            sparse_dim_mult=16,
            decoder_dim_mults=[2],
            device=device,
            lr=2e-4,
            topk=64,
            eps=1e-2,
            act_decay=0,
        )
    
    def _init_encoder_params(self):
        self.dense_encoder_blocks = torch.nn.ModuleList()
        in_dim = self.act_size

        for dim in self.encoder_dims:
            linear_layer = self._create_linear_layer(in_dim, dim)
            self.dense_encoder_blocks.append(
                torch.nn.Sequential(linear_layer, nn.ReLU(), nn.LayerNorm(dim))
            )
            in_dim = dim

        self.sparse_encoder_block = torch.nn.Sequential(
            self._create_linear_layer(in_dim, self.sparse_dim),
            nn.ReLU(),
            TopKActivation(self.topk),
        )

class ExperimentSAE2x2Ctrl(ExperimentSAEBase):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[2],
            sparse_dim_mult=16,
            decoder_dim_mults=[2],
            device=device,
            lr=2e-4,
            topk=64,
            eps=1e-2,
            act_decay=0,
        )

class ExperimentSAE2x2Squeeze(ActSqueezeSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[2],
            sparse_dim_mult=16,
            decoder_dim_mults=[2],
            device=device,
            lr=2e-4,
            topk=64,
            act_squeeze=1e-3,
            act_decay=0,
        )
