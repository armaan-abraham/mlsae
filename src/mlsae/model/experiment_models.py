from mlsae.model.model import ExperimentSAEBase

class ExperimentSAELayernormSqueeze5eNeg5lr4eNeg4SparseAdam(ExperimentSAEBase):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[],
            sparse_dim_mult=32,
            decoder_dim_mults=[],
            device=device,
            lr=4e-4,
            topk=128,
            act_squeeze=5e-5,
            act_decay=0,
            optimizer_type="sparse_adam",
            optimizer_config={
                "betas": (0.9, 0.999),
                "eps": 1e-8
            }
        )

class ExperimentSAELayernormSqueeze5eNeg5lr4eNeg4MixedMuon(ExperimentSAEBase):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[],
            sparse_dim_mult=32,
            decoder_dim_mults=[],
            device=device,
            lr=4e-4,
            topk=128,
            act_squeeze=5e-5,
            act_decay=0,
            optimizer_type="mixed_muon",
            optimizer_config={
                "momentum": 0.95,
                "nesterov": True,
                "ns_steps": 5
            }
        )
