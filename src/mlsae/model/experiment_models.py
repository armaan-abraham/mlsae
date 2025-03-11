from mlsae.model.model import ExperimentSAEBase, TopKActivation
import torch
import math

class ExperimentSAE2x4x4x2ActSqueeze0(ExperimentSAEBase):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[2, 4],
            sparse_dim_mult=32,
            decoder_dim_mults=[4, 2],
            device=device,
            topk=128,
            act_squeeze=0,
            optimizer_type="sparse_adam",
            optimizer_config={
                "lr": 4e-4,
            }
        )
