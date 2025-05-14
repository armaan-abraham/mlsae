from mlsae.model.model import ExperimentSAEBase
import torch
import math

class ExperimentSAEAccum1(ExperimentSAEBase):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[1, 1],
            sparse_dim_mult=32,
            decoder_dim_mults=[1, 1],
            device=device,
            l1_reg=5e-1,
            num_grad_accum_steps=1,
            optimizer_type="sparse_adam",
            optimizer_config={
                "lr": 4e-4,
            }
        )

class ExperimentSAEAccum4(ExperimentSAEBase):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[1, 1],
            sparse_dim_mult=32,
            decoder_dim_mults=[1, 1],
            device=device,
            l1_reg=5e-1,
            num_grad_accum_steps=2,
            optimizer_type="sparse_adam",
            optimizer_config={
                "lr": 4e-4,
            }
        )
