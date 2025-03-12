from mlsae.model.model import ExperimentSAEBase, TopKActivation
import torch
import math
import torch.nn as nn

class ResSAE(ExperimentSAEBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.act_squeeze == 0
        assert len(set(self.encoder_dims)) == 1
        assert len(set(self.decoder_dims)) == 1


class ExperimentSAERes1(ResSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[4],
            sparse_dim_mult=16,
            decoder_dim_mults=[4],
            device=device,
            topk_init=32,
            topk_final=32,
            topk_decay_iter=2000,
            act_squeeze=0,
            weight_decay=1e-2,
            optimizer_type="sparse_adam",
            optimizer_config={
                "lr": 5e-4,
            }
        )

class ExperimentSAERes2(ResSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[4, 4],
            sparse_dim_mult=16,
            decoder_dim_mults=[4, 4],
            device=device,
            topk_init=32,
            topk_final=32,
            topk_decay_iter=2000,
            act_squeeze=0,
            weight_decay=1e-2,
            optimizer_type="sparse_adam",
            optimizer_config={
                "lr": 5e-4,
            }
        )

class ExperimentSAERes3(ResSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[4, 4, 4],
            sparse_dim_mult=16,
            decoder_dim_mults=[4, 4, 4],
            device=device,
            topk_init=32,
            topk_final=32,
            topk_decay_iter=2000,
            act_squeeze=0,
            weight_decay=1e-2,
            optimizer_type="sparse_adam",
            optimizer_config={
                "lr": 5e-4,
            }
        )