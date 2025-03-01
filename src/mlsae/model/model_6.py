from mlsae.model.rl_sae import RLSAE


class DeepSAE6(RLSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[],
            sparse_dim_mult=8,
            decoder_dim_mults=[],
            name="RL2",
            device=device,
            lr=5e-4,
            temperature_initial=1.1,
            temperature_final=1,
            temperature_decay_half_life=500,
            num_samples=10,
            L0_penalty=1e-5,
            rl_loss_weight=0.2,
            prob_bias=-4,
        )
