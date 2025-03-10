from mlsae.model.model import ExperimentSAEBase

class ExperimentSAESparseAdam(ExperimentSAEBase):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[],
            sparse_dim_mult=8,
            decoder_dim_mults=[],
            device=device,
            topk=4,
            act_squeeze=0,
            optimizer_type="sparse_adam",
            optimizer_config={
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "lr": 1e-3,
            }
        )

class ExperimentSAEMixedMuon0(ExperimentSAEBase):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[],
            sparse_dim_mult=8,
            decoder_dim_mults=[],
            device=device,
            topk=4,
            act_squeeze=0,
            optimizer_type="mixed_muon",
            optimizer_config={
                "momentum": 0.95,
                "nesterov": True,
                "ns_steps": 5,
                "lr": {
                    "muon": 2e-2,
                    "adam": 1e-3,
                }
            }
        )

class ExperimentSAEMixedMuon1(ExperimentSAEBase):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[],
            sparse_dim_mult=8,
            decoder_dim_mults=[],
            device=device,
            topk=4,
            act_squeeze=0,
            optimizer_type="mixed_muon",
            optimizer_config={
                "momentum": 0.95,
                "nesterov": True,
                "ns_steps": 5,
                "lr": {
                    "muon": 1e-2,
                    "adam": 1e-3,
                }
            }
        )

class ExperimentSAEMixedMuon2(ExperimentSAEBase):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[],
            sparse_dim_mult=8,
            decoder_dim_mults=[],
            device=device,
            topk=4,
            act_squeeze=0,
            optimizer_type="mixed_muon",
            optimizer_config={
                "momentum": 0.95,
                "nesterov": True,
                "ns_steps": 5,
                "lr": {
                    "muon": 5e-3,
                    "adam": 1e-3,
                }
            }
        )

class ExperimentSAEMixedMuon3(ExperimentSAEBase):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[],
            sparse_dim_mult=8,
            decoder_dim_mults=[],
            device=device,
            topk=4,
            act_squeeze=0,
            optimizer_type="mixed_muon",
            optimizer_config={
                "momentum": 0.95,
                "nesterov": True,
                "ns_steps": 5,
                "lr": {
                    "muon": 2.5e-3,
                    "adam": 1e-3,
                }
            }
        )