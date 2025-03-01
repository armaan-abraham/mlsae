import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from mlsae.model.model import DeepSAE
from mlsae.config import DTYPES
import logging


DEFAULT_TEMPERATURE_INITIAL = 5
DEFAULT_TEMPERATURE_FINAL = 1.1
DEFAULT_TEMPERATURE_DECAY_HALF_LIFE = 500

class RLFeatureSelector(nn.Module):
    """
    A RL-based feature selector that samples which features to activate
    based on probabilities from a sigmoid activation.
    """

    def __init__(self, sparse_dim, temperature_initial=DEFAULT_TEMPERATURE_INITIAL, temperature_final=DEFAULT_TEMPERATURE_FINAL, L0_penalty=1e-2, num_samples=10, decay_half_life=DEFAULT_TEMPERATURE_DECAY_HALF_LIFE, prob_bias=-4):
        super().__init__()
        self.sparse_dim = sparse_dim
        self.temperature_initial = temperature_initial
        self.temperature_final = temperature_final
        self.temperature = temperature_initial  # Current temperature
        self.L0_penalty = L0_penalty
        self.num_samples = num_samples
        self.decay_half_life = decay_half_life
        self.prob_bias = prob_bias

        # Separate biases for magnitude and selection paths
        self.magnitude_bias = nn.Parameter(torch.zeros(sparse_dim))
        self.selection_bias = nn.Parameter(torch.zeros(sparse_dim))
        self.feature_scales = nn.Parameter(torch.ones(sparse_dim))

    def get_current_temperature(self, iteration):
        """Calculate the current temperature based on training iteration"""
        # Standard annealing formula: final + (initial - final) * decay_factor
        decay_factor = 0.5 ** (iteration / self.decay_half_life)
        return self.temperature_final + (self.temperature_initial - self.temperature_final) * decay_factor
        
    def update_temperature(self, iteration):
        """Update the temperature based on the current training iteration"""
        self.temperature = self.get_current_temperature(iteration)
        return self.temperature

    def get_probs(self, x):
        """Get activation probabilities from raw encoder outputs"""
        assert self.temperature > 0, f"Temperature must be greater than 0, got {self.temperature}"
        # Add selection bias to inputs for the probability calculation
        return torch.sigmoid((x + self.selection_bias) / self.temperature + self.prob_bias)

    def sample_mask(self, probs):
        """Sample a binary mask from probabilities"""
        return torch.bernoulli(probs)

    def forward(self, x):
        """During inference, just use deterministic threshold"""
        probs = self.get_probs(x)
        mask = (probs > 0.5).float()
        # Apply magnitude bias for the feature values
        magnitudes = F.relu(x + self.magnitude_bias)
        output = magnitudes * mask * self.feature_scales.unsqueeze(0)
        return output

    def sample_masks(self, x, num_samples=None):
        """Generate multiple mask samples for evaluation"""
        if num_samples is None:
            num_samples = self.num_samples

        probs = self.get_probs(x)

        # Store for learning
        self.saved_probs = probs

        # Sample multiple masks: shape [num_samples, batch_size, sparse_dim]
        masks = torch.stack([self.sample_mask(probs) for _ in range(num_samples)])
        return masks

    def apply_mask(self, x, mask):
        """Apply a specific mask to the input"""
        # Apply magnitude bias for the feature values
        magnitudes = F.relu(x + self.magnitude_bias)
        output = magnitudes * mask * self.feature_scales.unsqueeze(0)
        return output

    def update_selector(self, best_masks, normalized_rewards=None):
        """
        Update the selector to increase probability of selecting the best mask.
        best_masks: [batch_size, sparse_dim]
        normalized_rewards: [batch_size] - optional reward scaling factors
        """
        if not hasattr(self, "saved_probs"):
            return 0.0

        # Calculate log-probabilities for each sample's best mask
        # shape [batch_size, sparse_dim]
        log_probs = (
            torch.log(self.saved_probs + 1e-8) * best_masks
            + torch.log(1 - self.saved_probs + 1e-8) * (1 - best_masks)
        )

        # If normalized rewards are provided, use them to scale the log probabilities
        if normalized_rewards is not None:
            # Reshape for broadcasting: [batch_size, 1]
            scale_factors = normalized_rewards.unsqueeze(1)
            log_probs = log_probs * scale_factors

        # Mean over batch and features
        selector_loss = -log_probs.mean()

        return selector_loss


class RLSAE(DeepSAE):
    """
    A Sparse Autoencoder that uses a sample-and-select-best approach
    for feature selection.
    """

    def __init__(
        self,
        act_size: int,
        encoder_dim_mults: list[float],
        sparse_dim_mult: float,
        decoder_dim_mults: list[float],
        name: str = None,
        enc_dtype: str = "fp32",
        device: str = "cpu",
        lr: float = 1e-4,
        num_samples: int = 3,
        L0_penalty: float = 1e-2,
        rl_loss_weight: float = 1.0,
        prob_bias: float = -4,

        temperature_initial: float = DEFAULT_TEMPERATURE_INITIAL,
        temperature_final: float = DEFAULT_TEMPERATURE_FINAL,
        temperature_decay_half_life: int = DEFAULT_TEMPERATURE_DECAY_HALF_LIFE,
    ):
        self.L0_penalty = L0_penalty

        self.temperature_initial = temperature_initial
        self.temperature_final = temperature_final
        self.temperature_decay_half_life = temperature_decay_half_life
        self.prob_bias = prob_bias

        self.num_samples = num_samples
        self.rl_loss_weight = rl_loss_weight
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=encoder_dim_mults,
            sparse_dim_mult=sparse_dim_mult,
            decoder_dim_mults=decoder_dim_mults,
            name=name,
            enc_dtype=enc_dtype,
            device=device,
            topk=-1,
            act_decay=0,
            lr=lr,
        )

    def _init_encoder_params(self):
        self.dense_encoder_blocks = torch.nn.ModuleList()
        in_dim = self.act_size
        for dim in self.encoder_dims:
            linear_layer = self._create_linear_layer(in_dim, dim)
            self.dense_encoder_blocks.append(
                torch.nn.Sequential(linear_layer, nn.ReLU())
            )
            in_dim = dim

        # Create linear layer without bias
        self.sparse_encoder_block = torch.nn.Sequential(
            self._create_linear_layer_no_bias(in_dim, self.sparse_dim),
        )

        self.rl_selector = RLFeatureSelector(
            sparse_dim=self.sparse_dim,
            temperature_initial=self.temperature_initial,
            temperature_final=self.temperature_final,
            L0_penalty=self.L0_penalty,
            num_samples=self.num_samples,
            decay_half_life=self.temperature_decay_half_life,
            prob_bias=self.prob_bias,
        )

    def _create_linear_layer_no_bias(self, in_dim, out_dim):
        """Create a linear layer without bias"""
        layer = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.kaiming_normal_(layer.weight)
        return layer

    def _encode_up_to_selection(self, x):
        resid = x
        for block in self.dense_encoder_blocks:
            resid = block(resid)
        sparse_features = self.sparse_encoder_block(resid)
        return sparse_features

    def _decode(self, feature_acts):
        resid = feature_acts
        for block in self.decoder_blocks:
            resid = block(resid)
        return resid

    @torch.no_grad()
    def _evaluate_masks(self, x, sparse_features, masks):
        """
        Evaluate per-sample rewards for each mask.
        masks: [num_samples, batch_size, sparse_dim]
        Returns rewards: tensor of shape [num_samples, batch_size] 
        with the reward (negative loss) per sample for each mask.
        """
        batch_size = x.shape[0]

        # We'll accumulate rewards for each sample, for each mask
        all_rewards = []

        for i in range(masks.shape[0]):
            mask_i = masks[i]  # shape [batch_size, sparse_dim]
            feature_acts = self.rl_selector.apply_mask(sparse_features, mask_i)
            reconstructed = self._decode(feature_acts)

            # MSE per sample: shape [batch_size]
            mse_loss_per_sample = (reconstructed - x).pow(2).mean(dim=1)

            # Sparsity penalty per sample: shape [batch_size]
            # sum of active features per sample, times L0 penalty
            sparsity_penalty_per_sample = mask_i.sum(dim=1) * self.L0_penalty

            # Combine losses per sample and convert to reward (negative loss)
            total_reward_per_sample = -(mse_loss_per_sample + sparsity_penalty_per_sample)
            all_rewards.append(total_reward_per_sample)

        # Stack into shape [num_samples, batch_size]
        return torch.stack(all_rewards, dim=0)

    def _forward(self, x, iteration=None):
        sparse_features = self._encode_up_to_selection(x)

        # Update temperature if iteration is provided
        if iteration is not None and self.training:
            self.rl_selector.update_temperature(iteration)

        if self.training:
            masks = self.rl_selector.sample_masks(sparse_features)

            # Evaluate each mask's per-sample reward
            per_sample_rewards = self._evaluate_masks(x, sparse_features, masks)
            # per_sample_rewards shape: [num_samples, batch_size]

            # For each sample, pick the mask that yields maximum reward
            best_indices = per_sample_rewards.argmax(dim=0)  # [batch_size]

            # Gather the best mask for each sample
            batch_size = x.shape[0]
            best_masks = masks[best_indices, torch.arange(batch_size)]  # [batch_size, sparse_dim]

            # Apply best masks and get final reconstruction
            best_feature_acts = self.rl_selector.apply_mask(sparse_features, best_masks)
            best_reconstructed = self._decode(best_feature_acts)

            # Compute the final batch-averaged MSE loss
            mse_loss = (best_reconstructed - x).pow(2).mean()
            
            # Calculate normalized rewards for each input
            # Extract best rewards for each input
            best_rewards = per_sample_rewards[best_indices, torch.arange(batch_size)]  # [batch_size]
            
            # Compute mean and std of rewards across masks for each input
            reward_means = per_sample_rewards.mean(dim=0)  # [batch_size]
            reward_stds = per_sample_rewards.std(dim=0)  # [batch_size]
            
            # Add epsilon to avoid division by zero
            eps = 1e-8
            normalized_rewards = (best_rewards - reward_means) / (reward_stds + eps)

            # RL update: encourage selector to pick the best masks with normalized rewards
            selector_loss = self.rl_selector.update_selector(best_masks, normalized_rewards)
            weighted_selector_loss = selector_loss * self.rl_loss_weight
            final_loss = mse_loss + weighted_selector_loss

            return {
                "loss": final_loss,
                "mse_loss": mse_loss,
                "feature_acts": best_feature_acts,
                "reconstructed": best_reconstructed,
                "selector_loss": selector_loss,
                "temperature": self.rl_selector.temperature,
            }

        else:
            # Inference: deterministic threshold
            feature_acts = self.rl_selector(sparse_features)
            reconstructed = self._decode(feature_acts)
            mse_loss = (reconstructed - x).pow(2).mean()
            loss = mse_loss  # plus any optional penalty

            return {
                "loss": loss,
                "mse_loss": mse_loss,
                "feature_acts": feature_acts,
                "reconstructed": reconstructed,
                "selector_loss": torch.tensor(0.0),
                "temperature": self.rl_selector.temperature,
            }

    def forward(self, x, iteration=None):
        result = self._forward(x, iteration)
        
        if self.track_acts_stats:
            fa_float = result["feature_acts"].float()
            self.acts_sum += fa_float.sum().item()
            self.acts_sq_sum += (fa_float**2).sum().item()
            self.acts_elem_count += fa_float.numel()

            self.mse_sum += result["mse_loss"].item()
            self.mse_count += 1

        return result

    @torch.no_grad()
    def resample_sparse_features(self, idx):
        logging.info(f"Resampling sparse features {idx.sum().item()}")
        new_W_enc = torch.zeros_like(self.sparse_encoder_block[0].weight)
        new_W_dec = torch.zeros_like(self.decoder_blocks[0].weight)
        nn.init.kaiming_normal_(new_W_enc)
        nn.init.kaiming_normal_(new_W_dec)

        # Reset magnitude and selection biases
        new_magnitude_bias = torch.zeros_like(self.rl_selector.magnitude_bias)
        new_selection_bias = torch.zeros_like(self.rl_selector.selection_bias)

        self.sparse_encoder_block[0].weight.data[idx] = new_W_enc[idx]
        self.rl_selector.magnitude_bias.data[idx] = new_magnitude_bias[idx]
        self.rl_selector.selection_bias.data[idx] = new_selection_bias[idx]
        self.decoder_blocks[0].weight.data[:, idx] = new_W_dec[:, idx]

        self.rl_selector.feature_scales.data[idx] = 1.0

    def get_config_dict(self):
        config = super().get_config_dict()
        config.update(
            {
                "temperature_initial": self.temperature_initial,
                "temperature_final": self.temperature_final,
                "temperature_decay_half_life": self.temperature_decay_half_life,
                "L0_penalty": self.rl_selector.L0_penalty,
                "num_samples": self.rl_selector.num_samples,
                "rl_loss_weight": self.rl_loss_weight,
                "prob_bias": self.rl_selector.prob_bias,
            }
        )
        return config

    def should_resample_sparse_features(self, idx):
        return False
