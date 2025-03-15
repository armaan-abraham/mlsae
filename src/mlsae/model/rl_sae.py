from networkx import radius
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from mlsae.model.model import ExperimentSAEBase


# TODO: check that biases are handled correctly
# TODO: implement config dict

class RLFeatureSelector(nn.Module):
    """
    A RL-based feature selector that samples which features to activate
    based on probabilities from a sigmoid activation.
    """

    def __init__(self, sparse_dim, L0_penalty=1e-2, num_samples=10, base_L0=30, action_collapse_penalty_lambda=1e-1):
        super().__init__()
        self.sparse_dim = sparse_dim
        self.L0_penalty = L0_penalty
        self.num_samples = num_samples
        self.base_L0 = base_L0
        self.action_collapse_penalty_lambda = action_collapse_penalty_lambda

        # Compute the base bias from the base L0
        self.base_prob = base_L0 / sparse_dim
        self.base_bias = np.log(self.base_prob / (1 - self.base_prob))

        # Separate bias and scalar for magnitudes
        self.magnitude_bias = nn.Parameter(torch.zeros(sparse_dim))
        self.magnitude_scalar = nn.Parameter(torch.ones(sparse_dim))

    def get_probs(self, x):
        """Get activation probabilities from raw encoder outputs"""
        # Add selection bias to inputs for the probability calculation
        return torch.sigmoid(x + self.base_bias)

    def sample_mask(self, probs):
        """Sample a binary mask from probabilities"""
        return torch.bernoulli(probs)

    def forward(self, x):
        """During inference, just use deterministic threshold"""
        probs = self.get_probs(x)
        mask = (probs > 0.5).float()
        magnitudes = self.get_feature_mags(x, mask)
        return magnitudes

    def sample_masks(self, x, num_samples=None):
        """Generate multiple mask samples for evaluation"""
        if num_samples is None:
            num_samples = self.num_samples
            
        # Store for learning
        self.saved_prebias_logits = x

        probs = self.get_probs(x)

        # Sample multiple masks: shape [num_samples, batch_size, sparse_dim]
        masks = torch.stack([self.sample_mask(probs) for _ in range(num_samples)])
        return masks

    def get_feature_mags(self, x, mask):
        magnitudes = F.relu(x * self.magnitude_scalar + self.magnitude_bias) * mask
        return magnitudes

    def update_selector(self, masks, rewards):
        """
        Update the selector using all masks, weighted by their rewards, with PPO clipping.
        masks: [num_samples, batch_size, sparse_dim] - all sampled masks
        rewards: [num_samples, batch_size] - rewards for each mask
        """
        assert hasattr(self, "saved_prebias_logits")

        num_samples, batch_size, _ = masks.shape

        eps = 1e-6
        
        # Normalize rewards across samples for each batch item
        reward_means = rewards.mean(dim=0, keepdim=True)  # [1, batch_size]
        reward_stds = rewards.std(dim=0, keepdim=True) + eps # [1, batch_size]
        advantages = (rewards - reward_means) / reward_stds  # [num_samples, batch_size]
        
        # Expand current probs and old probs for all samples
        # [batch_size, sparse_dim] -> [num_samples, batch_size, sparse_dim]
        saved_probs = self.get_probs(self.saved_prebias_logits)
        expanded_probs = saved_probs.unsqueeze(0).expand(num_samples, -1, -1)

        if torch.any(expanded_probs > 1) or torch.any(expanded_probs < 0):
            # Print problematic probability values for debugging
            invalid_probs = expanded_probs[expanded_probs > 1]
            if len(invalid_probs) > 0:
                raise ValueError(f"ERROR: Found {len(invalid_probs)} probabilities > 1. Max: {invalid_probs.max().item()}")
            
            invalid_probs = expanded_probs[expanded_probs < 0]
            if len(invalid_probs) > 0:
                raise ValueError(f"ERROR: Found {len(invalid_probs)} probabilities < 0. Min: {invalid_probs.min().item()}")
        
        log_mask_probs = torch.log(expanded_probs + eps) * masks + torch.log(1 - expanded_probs + eps) * (1 - masks)

        selector_loss = -torch.mean(log_mask_probs * advantages[:, :, None])

        # Compute penalty for logit deviation from base prob
        logit_deviation = (self.saved_prebias_logits).abs().mean()
        action_collapse_penalty = self.action_collapse_penalty_lambda * logit_deviation
        
        # Add to selector loss
        selector_loss = selector_loss + action_collapse_penalty
        
        return selector_loss


class RLSAE(ExperimentSAEBase):
    def __init__(
        self,
        act_size: int,
        encoder_dim_mults: list[float],
        sparse_dim_mult: float,
        decoder_dim_mults: list[float],
        enc_dtype: str = "fp32",
        device: str = "cpu",
        num_samples: int = 3,
        L0_penalty: float = 1e-2,
        rl_loss_weight: float = 1.0,
        optimizer_type: str = "sparse_adam",
        optimizer_config: dict = None,
        optimize_steps: int = 1,
        weight_decay: float = 0,

        base_L0: float = 30,
        action_collapse_penalty_lambda: float = 1e-1,
    ):
        self.L0_penalty = L0_penalty
        self.num_samples = num_samples
        self.rl_loss_weight = rl_loss_weight

        self.base_L0 = base_L0
        self.action_collapse_penalty_lambda = action_collapse_penalty_lambda


        super().__init__(
            act_size=act_size,
            encoder_dim_mults=encoder_dim_mults,
            sparse_dim_mult=sparse_dim_mult,
            decoder_dim_mults=decoder_dim_mults,
            enc_dtype=enc_dtype,
            device=device,
            topk=-1,
            weight_decay=weight_decay,
            act_squeeze=0,
            optimizer_type=optimizer_type,
            optimizer_config=optimizer_config,
            optimize_steps=optimize_steps,
        )

    def _init_params(self):
        super()._init_params()

        self.rl_selector = RLFeatureSelector(
            sparse_dim=self.sparse_dim,
            L0_penalty=self.L0_penalty,
            num_samples=self.num_samples,
            base_L0=self.base_L0,
            action_collapse_penalty_lambda=self.action_collapse_penalty_lambda,
        )

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
            feature_acts = self.rl_selector.get_feature_mags(sparse_features, mask_i)
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
        preacts = self._get_preacts(x)

        if self.training:
            masks = self.rl_selector.sample_masks(preacts)

            # Evaluate each mask's per-sample reward
            per_sample_rewards = self._evaluate_masks(x, preacts, masks)
            # per_sample_rewards shape: [num_samples, batch_size]

            # For each sample, pick the mask that yields maximum reward
            best_indices = per_sample_rewards.argmax(dim=0)  # [batch_size]

            # Gather the best mask for each sample
            batch_size = x.shape[0]
            best_masks = masks[best_indices, torch.arange(batch_size)]  # [batch_size, sparse_dim]

            # Apply best masks and get final reconstruction
            best_feature_acts = self.rl_selector.get_feature_mags(preacts, best_masks)
            best_reconstructed = self._decode(best_feature_acts)

            # Compute the final batch-averaged MSE loss
            mse_loss = (best_reconstructed - x).pow(2).mean()
            
            # RL update: use all masks weighted by their rewards
            selector_loss = self.rl_selector.update_selector(masks, per_sample_rewards)
            weighted_selector_loss = selector_loss * self.rl_loss_weight
            final_loss = mse_loss + weighted_selector_loss

            return {
                "loss": final_loss,
                "mse_loss": mse_loss,
                "feature_acts": best_feature_acts,
                "reconstructed": best_reconstructed,
                "selector_loss": selector_loss,
            }

        else:
            # Inference: deterministic threshold
            feature_acts = self.rl_selector(preacts)
            reconstructed = self._decode(feature_acts)
            mse_loss = (reconstructed - x).pow(2).mean()
            loss = mse_loss  # plus any optional penalty

            return {
                "loss": loss,
                "mse_loss": mse_loss,
                "feature_acts": feature_acts,
                "reconstructed": reconstructed,
            }

    def optimize(self, x, optimizer, iteration=None):
        with torch.no_grad():
            preacts = self._get_preacts(x)
            probs = self.rl_selector.get_probs(preacts)
            
            sorted_probs = torch.sort(probs.flatten())[0]
            n = sorted_probs.size(0)
            index = torch.arange(1, n + 1, device=preacts.device)
            gini = (2 * (index * sorted_probs).sum() / (n * sorted_probs.sum())) - (n + 1) / n

            avg_prob = probs.mean()
            deviation = (probs - self.rl_selector.base_prob).pow(2).mean()
            

        result = super().optimize(x, optimizer, iteration)
        result["gini"] = gini.item()
        result["prob_penalty"] = deviation.item()
        result["prob_mean"] = avg_prob.item()
        result["prob_min"] = probs.min().item()
        result["prob_max"] = probs.max().item()
        result["prob_std"] = probs.std().item()

        
        return result

    def get_config_dict(self):
        # Get base configuration from parent class
        config = super().get_config_dict()
        
        # Add RL-specific parameters
        config.update({
            "num_samples": self.num_samples,
            "L0_penalty": self.L0_penalty,
            "rl_loss_weight": self.rl_loss_weight,
            "base_L0": self.base_L0,
            "action_collapse_penalty_lambda": self.action_collapse_penalty_lambda,
        })
        
        return config