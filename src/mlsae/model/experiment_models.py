from mlsae.model.rl_sae import RLSAE
from mlsae.model.model import ExperimentSAEBase
import torch
import torch.nn as nn



class ExperimentSAETop47(ExperimentSAEBase):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[],
            sparse_dim_mult=8,
            decoder_dim_mults=[],
            device=device,
            lr=2e-4,
            topk=47,
            act_decay=0,
        )

class ExperimentSAETop42(ExperimentSAEBase):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[],
            sparse_dim_mult=8,
            decoder_dim_mults=[],
            device=device,
            lr=2e-4,
            topk=42,
            act_decay=0,
        )

class ExperimentSAERL0(RLSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[],
            sparse_dim_mult=8,
            decoder_dim_mults=[],
            device=device,
            lr=1e-3,
            temperature_initial=1,
            temperature_final=1,
            num_samples=10,
            L0_penalty=1e-5,
            rl_loss_weight=1e-3,
            prob_bias=-4,
            prob_deadness_penalty=2,
        )

class ExperimentSAERL2(RLSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[],
            sparse_dim_mult=8,
            decoder_dim_mults=[],
            device=device,
            lr=5e-4,
            temperature_initial=1,
            temperature_final=1,
            num_samples=10,
            L0_penalty=1e-5,
            rl_loss_weight=1e-3,
            prob_bias=-4,
            prob_deadness_penalty=2,
        )

class ExperimentSAERL3(RLSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[],
            sparse_dim_mult=8,
            decoder_dim_mults=[],
            device=device,
            lr=5e-4,
            temperature_initial=1,
            temperature_final=1,
            num_samples=10,
            L0_penalty=2e-5,
            rl_loss_weight=1e-3,
            prob_bias=-4,
            prob_deadness_penalty=2,
        )

class ExperimentSAERL4(RLSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            device=device,
            encoder_dim_mults=[],
            sparse_dim_mult=8,
            decoder_dim_mults=[],
            lr=5e-4,
            temperature_initial=1,
            temperature_final=1,
            num_samples=10,
            L0_penalty=4e-5,
            rl_loss_weight=1e-3,
            prob_bias=-4,
            prob_deadness_penalty=2,
        )

class ActInflationSAE(ExperimentSAEBase):
    def __init__(self, act_size: int, *args, act_inflate: float = 1, **kwargs):
        assert kwargs.get("act_decay", 0) == 0, "ActInflationSAE does not support act_decay"
        super().__init__(
            act_size,
            *args,
            **kwargs
        )
        self.act_inflate = act_inflate
    
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
        
        # Avoid log(0) by adding small epsilon
        eps = 1e-8
        log_mean_acts = torch.log(mean_acts_per_feature + eps)
        
        # Negative mean of log activations (encourage higher activations)
        act_inflation_loss = -log_mean_acts.mean() * self.act_inflate
        
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
        loss = mse_loss + act_inflation_loss

        return {
            "loss": loss,
            "mse_loss": mse_loss,
            "feature_acts": feature_acts,
            "reconstructed": reconstructed,
            "act_inflation_loss": act_inflation_loss,
        }
        
class ExperimentSAETop16Control(ActInflationSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[2],
            sparse_dim_mult=32,
            decoder_dim_mults=[2],
            device=device,
            lr=2e-4,
            topk=8,
            act_inflate=0,
            act_decay=0,
        )

class ExperimentSAETop16InflateNeg5(ActInflationSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[2],
            sparse_dim_mult=16,
            decoder_dim_mults=[2],
            device=device,
            lr=2e-4,
            topk=8,
            act_inflate=1e-5,
            act_decay=0,
        )

class ExperimentSAETop16InflateNeg4(ActInflationSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[2],
            sparse_dim_mult=16,
            decoder_dim_mults=[2],
            device=device,
            lr=2e-4,
            topk=8,
            act_inflate=1e-4,
            act_decay=0,
        )

class ExperimentSAETop16InflateNeg3(ActInflationSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[2],
            sparse_dim_mult=16,
            decoder_dim_mults=[2],
            device=device,
            lr=2e-4,
            topk=8,
            act_inflate=1e-3,
            act_decay=0,
        )

class ExperimentSAETop16InflateNeg2(ActInflationSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[2],
            sparse_dim_mult=16,
            decoder_dim_mults=[2],
            device=device,
            lr=2e-4,
            topk=8,
            act_inflate=1e-2,
            act_decay=0,
        ) 

class ExperimentSAETop16InflateNeg1(ActInflationSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[2],
            sparse_dim_mult=16,
            decoder_dim_mults=[2],
            device=device,
            lr=2e-4,
            topk=8,
            act_inflate=1e-1,
            act_decay=0,
        ) 

class ExperimentSAETop16Inflate0(ActInflationSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[2],
            sparse_dim_mult=16,
            decoder_dim_mults=[2],
            device=device,
            lr=2e-4,
            topk=8,
            act_inflate=1e0,
            act_decay=0,
        )

class ExperimentSAETop16Inflate1(ActInflationSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[2],
            sparse_dim_mult=16,
            decoder_dim_mults=[2],
            device=device,
            lr=2e-4,
            topk=8,
            act_inflate=1e1,
            act_decay=0,
        )
