from mlsae.model.model import ExperimentSAEBase, TopKActivation
import torch
import math
import torch.nn as nn

from mlsae.model.rl_sae import RLSAE

# class ExperimentSAERL1_0(RLSAE):
#     def __init__(self, act_size: int, device: str = "cpu"):
#         super().__init__(
#             act_size=act_size,
#             encoder_dim_mults=[1],
#             sparse_dim_mult=8,
#             decoder_dim_mults=[1],
#             device=device,
#             num_samples=10,
#             L0_penalty=2e-5,
#             rl_loss_weight=1,
#             optimizer_type="sparse_adam",
#             optimizer_config={
#                 "lr": 1e-4,
#             },
#             optimize_steps=1,

#             base_L0=2,
#             action_collapse_penalty_lambda=2e-4,
#         )

class ExperimentSAETopk(ExperimentSAEBase):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            device=device,
            encoder_dim_mults=[1],
            sparse_dim_mult=8,
            decoder_dim_mults=[1],
            topk_init=2,
            topk_final=2,
            topk_decay_iter=1000,
            act_squeeze=0,
            weight_decay=0,
            optimize_steps=1,
            optimizer_type="mixed_muon",
            optimizer_config={
                "lr": 1e-4,
            }
        )
