from mlsae.model.rl_sae import RLSAE


class DeepSAE3(RLSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[],
            sparse_dim_mult=8,
            decoder_dim_mults=[],
            name="RL2",
            device=device,
            lr=2e-4,
            rl_temperature=2,
            num_samples=3,
            L0_penalty=1,
            rl_loss_weight=2,
        )
