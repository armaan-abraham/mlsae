from mlsae.model.model import ExperimentSAEBase, TopKActivation
import torch
import math
import torch.nn as nn

from mlsae.model.rl_sae import RLSAE

class ExperimentSAERL1_0(RLSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[1],
            sparse_dim_mult=16,
            decoder_dim_mults=[1],
            device=device,
            num_samples=10,
            L0_penalty=1e-5,
            rl_loss_weight=0.4,
            prob_bias=-4,
            prob_deadness_penalty=2e-6,
            optimizer_type="sparse_adam",
            optimizer_config={
                "lr": 5e-4,
            },
            optimize_steps=3,
            ppo_clip=0.05,
        )


# class ExperimentSAETopk(ExperimentSAEBase):
#     def __init__(self, act_size: int, device: str = "cpu"):
#         super().__init__(
#             act_size=act_size,
#             encoder_dim_mults=[1],
#             sparse_dim_mult=16,
#             decoder_dim_mults=[1],
#             device=device,
#             topk_init=35,
#             topk_final=35,
#             topk_decay_iter=2000,
#             act_squeeze=0,
#             weight_decay=0,
#             optimizer_type="sparse_adam",
#             optimizer_config={
#                 "lr": 5e-4,
#             },
#             optimize_steps=3,
#         )

