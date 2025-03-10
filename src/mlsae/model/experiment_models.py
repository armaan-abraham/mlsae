from mlsae.model.model import ExperimentSAEBase, TopKActivation
import torch
import math

class ActSqueezeDecaySAE(ExperimentSAEBase):
    """
    Sparse autoencoder with a decaying act_squeeze parameter.
    
    The squeeze value decays exponentially from act_squeeze_max to act_squeeze_min
    over time using formula: act_squeeze = min + (max - min) * (0.1 ^ (iter / tau)).
    The parameter tau represents the number of iterations required for the decay factor
    to reach 0.1.
    """
    def __init__(
        self,
        act_size: int,
        encoder_dim_mults: list[float],
        sparse_dim_mult: float,
        decoder_dim_mults: list[float],
        device: str = "cpu",
        topk: int = 16,
        act_squeeze_max: float = 1e-3,
        act_squeeze_min: float = 1e-6,
        act_squeeze_tau: int = 1000,
        optimizer_type: str = "sparse_adam",
        optimizer_config: dict = None,
    ):
        self.act_squeeze_max = act_squeeze_max
        self.act_squeeze_min = act_squeeze_min
        self.act_squeeze_tau = act_squeeze_tau
        
        # Initialize with max squeeze value
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=encoder_dim_mults,
            sparse_dim_mult=sparse_dim_mult,
            decoder_dim_mults=decoder_dim_mults,
            device=device,
            topk=topk,
            act_squeeze=act_squeeze_max,  # Start with max value
            optimizer_type=optimizer_type,
            optimizer_config=optimizer_config or {},
        )
    
    def _forward(self, x, iteration=None):
        assert iteration is not None, "iteration must be provided"
        # Update act_squeeze based on iteration number if provided
        if iteration is not None:
            exp_factor = 0.1 ** (iteration / self.act_squeeze_tau)
            self.act_squeeze = self.act_squeeze_min + (self.act_squeeze_max - self.act_squeeze_min) * exp_factor
            
        # Call parent class forward method
        return super()._forward(x, iteration)
    
    def get_config_dict(self):
        # Get base configuration
        config = super().get_config_dict()
        
        # Add specific configuration for ActSqueezeDecaySAE
        config.update({
            "act_squeeze_max": self.act_squeeze_max,
            "act_squeeze_min": self.act_squeeze_min,
            "act_squeeze_tau": self.act_squeeze_tau,
        })
        
        return config

class ExperimentSAE2x2ActSqueezeDecay0(ActSqueezeDecaySAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[2],
            sparse_dim_mult=32,
            decoder_dim_mults=[2],
            device=device,
            topk=128,
            act_squeeze_max=1e-3,
            act_squeeze_min=1e-6,
            act_squeeze_tau=2000,
            optimizer_type="sparse_adam",
            optimizer_config={
                "lr": 4e-4,
            }
        )

class ExperimentSAE2x2ActSqueezeDecay1(ActSqueezeDecaySAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[2],
            sparse_dim_mult=32,
            decoder_dim_mults=[2],
            device=device,
            topk=128,
            act_squeeze_max=1e-3,
            act_squeeze_min=1e-7,
            act_squeeze_tau=2000,
            optimizer_type="sparse_adam",
            optimizer_config={
                "lr": 4e-4,
            }
        )

class ExperimentSAE2x2ActSqueezeDecay2(ActSqueezeDecaySAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[2],
            sparse_dim_mult=32,
            decoder_dim_mults=[2],
            device=device,
            topk=128,
            act_squeeze_max=1e-3,
            act_squeeze_min=1e-12,
            act_squeeze_tau=2000,
            optimizer_type="sparse_adam",
            optimizer_config={
                "lr": 4e-4,
            }
        )
