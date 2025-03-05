from mlsae.model.rl_sae import RLSAE


class DeepSAE0(RLSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[],
            sparse_dim_mult=8,
            decoder_dim_mults=[],
            name="RL0",
            device=device,
            lr=5e-4,
            temperature_initial=1,
            temperature_final=1,
            num_samples=10,
            L0_penalty=1e-5,
            rl_loss_weight=1e-3,
            prob_bias=-4,
            prob_deadness_penalty=2,
        )
