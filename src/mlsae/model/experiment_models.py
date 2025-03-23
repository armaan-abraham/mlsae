from mlsae.model.model import ExperimentSAEBase, TopKActivation
import torch
import math
import torch.nn as nn

from mlsae.model.rl_sae import RLSAE

class ExperimentSAERL(RLSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[],
            sparse_dim_mult=8,
            decoder_dim_mults=[],
            device=device,
            num_samples=5,
            L0_penalty=2e-5,
            rl_loss_weight=0.2,
            optimizer_type="sparse_adam",
            optimizer_config={
                "lr": 2e-3,
            },
            optimize_steps=1,

            base_L0=2,
            action_collapse_penalty_lambda=2e-4,
        )

# class ExperimentSAETopk(ExperimentSAEBase):
#     def __init__(self, act_size: int, device: str = "cpu"):
#         super().__init__(
#             act_size=act_size,
#             device=device,
#             encoder_dim_mults=[],
#             sparse_dim_mult=8,
#             decoder_dim_mults=[],
#             topk_init=2,
#             topk_final=2,
#             topk_decay_iter=1000,
#             act_squeeze=0,
#             weight_decay=0,
#             optimize_steps=1,
#             optimizer_type="sparse_adam",
#             optimizer_config={
#                 "lr": 2e-3,
#             }
#         )
