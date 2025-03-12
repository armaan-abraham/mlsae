from mlsae.model.model import ExperimentSAEBase, TopKActivation
import torch
import math
import torch.nn as nn

from mlsae.model.rl_sae import RLSAE

class ExperimentSAERL(RLSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[1],
            sparse_dim_mult=16,
            decoder_dim_mults=[1],
            device=device,
            num_samples=10,
            L0_penalty=5e-6,
            rl_loss_weight=1e-3,
            prob_bias=-4,
            prob_deadness_penalty=1,
            optimizer_type="sparse_adam",
            optimizer_config={
                "lr": 5e-4,
            }
        )

class ExperimentSAETopk(ExperimentSAEBase):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[1],
            sparse_dim_mult=16,
            decoder_dim_mults=[1],
            device=device,
            topk_init=32,
            topk_final=32,
            topk_decay_iter=2000,
            act_squeeze=0,
            weight_decay=0,
            optimizer_type="sparse_adam",
            optimizer_config={
                "lr": 5e-4,
            }
        )
