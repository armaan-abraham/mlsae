from mlsae.model.rl_sae import RLSAE


class DeepSAE5(RLSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[],
            sparse_dim_mult=8,
            decoder_dim_mults=[],
            name="RL2",
            device=device,
            lr=5e-4,
            temperature_initial=2.5,
            temperature_final=1.5,
            temperature_decay_half_life=3000,
            num_samples=5,
            L0_penalty=2.5e-6,
            rl_loss_weight=2,
            prob_bias=-2,
        )
