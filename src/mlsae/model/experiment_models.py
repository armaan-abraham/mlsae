from mlsae.model.rl_sae import RLSAE
from mlsae.model.model import ExperimentSAEBase
import torch
import torch.nn as nn


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
            prob_deadness_penalty=1,
        )

class ExperimentSAERL1(RLSAE):
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
            rl_loss_weight=2e-3,
            prob_bias=-4,
            prob_deadness_penalty=1,
        )

class ExperimentSAERL2(RLSAE):
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
            rl_loss_weight=4e-3,
            prob_bias=-4,
            prob_deadness_penalty=5e-1,
        )

class ExperimentSAERL3(RLSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            device=device,
            encoder_dim_mults=[],
            sparse_dim_mult=8,
            decoder_dim_mults=[],
            lr=1e-3,
            temperature_initial=1,
            temperature_final=1,
            num_samples=10,
            L0_penalty=1e-5,
            rl_loss_weight=8e-3,
            prob_bias=-4,
            prob_deadness_penalty=2.5e-1,
        )