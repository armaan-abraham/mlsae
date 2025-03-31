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

    def __init__(
        self, sparse_dim, L0_penalty=1e-2, num_samples=10, base_L0=30, temperature=1.0
    ):
        super().__init__()
        self.sparse_dim = sparse_dim
        self.L0_penalty = L0_penalty
        self.num_samples = num_samples
        self.base_L0 = base_L0
        self.temperature = temperature

        # Compute the base bias from the base L0
        self.base_prob = base_L0 / sparse_dim
        self.base_bias = np.log(self.base_prob / (1 - self.base_prob))

        # Separate bias and scalar for magnitudes
        self.magnitude_bias = nn.Parameter(torch.zeros(sparse_dim))
        self.magnitude_scalar = nn.Parameter(torch.ones(sparse_dim))

        self.prob_bias = nn.Parameter(torch.zeros(sparse_dim))
        self.prob_scalar = nn.Parameter(torch.ones(sparse_dim))

    def get_probs(self, x, temperature=1.0):
        """Get activation probabilities from raw encoder outputs"""
        prob_acts = x * self.prob_scalar + self.prob_bias
        # Add selection bias to inputs for the probability calculation
        probs = torch.sigmoid(
            (prob_acts / temperature + self.base_bias).clamp(min=-5, max=5)
        )
        if torch.any(probs > 1) or torch.any(probs < 0):
            # Print out the problematic values
            invalid_probs = torch.logical_or(probs > 1, probs < 0)
            if torch.any(invalid_probs):
                print(f"WARNING: Invalid probabilities detected!")
                print(f"Number of invalid values: {invalid_probs.sum().item()}")
                invalid_indices = torch.nonzero(invalid_probs, as_tuple=True)
                invalid_values = probs[invalid_indices]
                print(f"Invalid probability values: {invalid_values.tolist()}")
                print(f"Corresponding input x values: {x[invalid_indices].tolist()}")
                probs = torch.clamp(probs, 0, 1)
        assert torch.all(probs >= 0) and torch.all(
            probs <= 1
        ), "Probabilities must be between 0 and 1"
        return probs

    def sample_mask(self, probs):
        """Sample a binary mask from probabilities"""
        mask = torch.bernoulli(probs)
        return mask

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

        probs = self.get_probs(x, temperature=self.temperature)

        # Sample multiple masks: shape [num_samples, batch_size, sparse_dim]
        masks = torch.stack([self.sample_mask(probs) for _ in range(num_samples)])
        return masks

    def get_feature_mags(self, x, mask):
        magnitudes = (
            torch.nn.functional.softplus(
                x * self.magnitude_scalar + self.magnitude_bias, beta=0.5
            )
            * mask
        )
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
        reward_stds = rewards.std(dim=0, keepdim=True) + eps  # [1, batch_size]
        advantages = (rewards - reward_means) / reward_stds  # [num_samples, batch_size]

        # Expand current probs and old probs for all samples
        # [batch_size, sparse_dim] -> [num_samples, batch_size, sparse_dim]
        saved_probs = self.get_probs(self.saved_prebias_logits)
        expanded_probs = saved_probs.unsqueeze(0).expand(num_samples, -1, -1)

        if torch.any(expanded_probs > 1) or torch.any(expanded_probs < 0):
            # Print problematic probability values for debugging
            invalid_probs = expanded_probs[expanded_probs > 1]
            if len(invalid_probs) > 0:
                raise ValueError(
                    f"ERROR: Found {len(invalid_probs)} probabilities > 1. Max: {invalid_probs.max().item()}"
                )

            invalid_probs = expanded_probs[expanded_probs < 0]
            if len(invalid_probs) > 0:
                raise ValueError(
                    f"ERROR: Found {len(invalid_probs)} probabilities < 0. Min: {invalid_probs.min().item()}"
                )

        log_mask_probs = torch.log(expanded_probs + eps) * masks + torch.log(
            1 - expanded_probs + eps
        ) * (1 - masks)

        selector_loss = -torch.mean(log_mask_probs * advantages[:, :, None])

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
        initial_temperature: float = 1.0,
        min_temperature: float = 0.1,
        temperature_tau: float = 0.999,
        loss_stats_momentum: float = 0.9,
        eval_batch_size: int = 10,
    ):
        self.L0_penalty = L0_penalty
        self.num_samples = num_samples
        self.rl_loss_weight = rl_loss_weight
        self.base_L0 = base_L0
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.temperature_tau = temperature_tau
        self.loss_stats_momentum = loss_stats_momentum
        self.eval_batch_size = eval_batch_size

        # Initialize tracking flag - tensors will be created in _init_params
        self.loss_stats_initialized = False

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
            temperature=self.initial_temperature,
        )

        # Initialize running statistics as torch tensors on the correct device
        self.mse_loss_mean = torch.tensor(0.0, device=self.device)
        self.mse_loss_sq_mean = torch.tensor(0.0, device=self.device)
        self.selector_loss_mean = torch.tensor(0.0, device=self.device)
        self.selector_loss_sq_mean = torch.tensor(0.0, device=self.device)

    @torch.no_grad()
    def _evaluate_masks(self, x, sparse_features, masks):
        """
        Evaluate per-sample rewards for each mask using batched processing.
        masks: [num_samples, batch_size, sparse_dim]
        Returns rewards: tensor of shape [num_samples, batch_size]
        """
        num_samples, batch_size, _ = masks.shape
        all_rewards = []

        # Process masks in batches to reduce memory usage
        eval_batch_size = self.eval_batch_size
        for start_idx in range(0, num_samples, eval_batch_size):
            end_idx = min(start_idx + eval_batch_size, num_samples)
            batch_masks = masks[start_idx:end_idx]

            # Process current batch of masks
            batch_feature_acts = self.rl_selector.get_feature_mags(
                sparse_features.unsqueeze(0).expand(len(batch_masks), -1, -1),
                batch_masks,
            )

            # Decode all samples in current batch
            reconstructed_flat = self._decode(
                batch_feature_acts.view(-1, self.sparse_dim)
            )
            reconstructed = reconstructed_flat.view(len(batch_masks), batch_size, -1)

            # Calculate rewards for this batch
            batch_mse = (reconstructed - x.unsqueeze(0)).pow(2).mean(dim=2)
            batch_sparsity = batch_masks.sum(dim=2) * self.L0_penalty
            batch_rewards = -(batch_mse + batch_sparsity)

            all_rewards.append(batch_rewards)

        return torch.cat(all_rewards, dim=0)

    def _update_loss_stats(self, mse_loss, selector_loss):
        """Update running statistics for loss normalization"""
        # Detach to avoid computation graph issues
        mse_val = mse_loss.detach()
        selector_val = selector_loss.detach()

        if not self.loss_stats_initialized:
            # Initialize with first values
            self.mse_loss_mean = mse_val.clone()
            self.mse_loss_sq_mean = mse_val.pow(2)
            self.selector_loss_mean = selector_val.clone()
            self.selector_loss_sq_mean = selector_val.pow(2)
            self.loss_stats_initialized = True
        else:
            # Update running statistics with momentum
            alpha = 1 - self.loss_stats_momentum

            # Update means
            self.mse_loss_mean = (
                self.loss_stats_momentum * self.mse_loss_mean + alpha * mse_val
            )
            self.mse_loss_sq_mean = (
                self.loss_stats_momentum * self.mse_loss_sq_mean
                + alpha * mse_val.pow(2)
            )

            self.selector_loss_mean = (
                self.loss_stats_momentum * self.selector_loss_mean
                + alpha * selector_val
            )
            self.selector_loss_sq_mean = (
                self.loss_stats_momentum * self.selector_loss_sq_mean
                + alpha * selector_val.pow(2)
            )

    @torch.no_grad()
    def _get_loss_stds(self):
        """Calculate standard deviations from running statistics"""
        mse_var = self.mse_loss_sq_mean - self.mse_loss_mean.pow(2)
        selector_var = self.selector_loss_sq_mean - self.selector_loss_mean.pow(2)

        return torch.clamp(torch.sqrt(mse_var), min=1e-8), torch.clamp(
            torch.sqrt(selector_var), min=1e-8
        )

    def current_temperature(self, iteration):
        return self.min_temperature + (
            self.initial_temperature - self.min_temperature
        ) * (0.5 ** (iteration / self.temperature_tau))

    def _forward(self, x, iteration=None):
        assert "cuda" in str(self.device) and "cuda" in str(
            next(self.rl_selector.parameters()).device
        ), "Both SAE and RL selector must be on GPU"

        # Update temperature based on iteration if provided
        if iteration is not None and self.training:
            self.rl_selector.temperature = self.current_temperature(iteration)

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
            best_masks = masks[
                best_indices, torch.arange(batch_size)
            ]  # [batch_size, sparse_dim]

            # Apply best masks and get final reconstruction
            best_feature_acts = self.rl_selector.get_feature_mags(preacts, best_masks)
            best_reconstructed = self._decode(best_feature_acts)

            # Compute the final batch-averaged MSE loss
            mse_loss = (best_reconstructed - x).pow(2).mean()

            # RL update: use all masks weighted by their rewards
            selector_loss = self.rl_selector.update_selector(masks, per_sample_rewards)

            # Update running statistics and normalize losses
            self._update_loss_stats(mse_loss, selector_loss)

            # Calculate normalized losses
            mse_std, selector_std = self._get_loss_stds()
            mse_norm = (mse_loss - self.mse_loss_mean) / mse_std
            selector_norm = (selector_loss - self.selector_loss_mean) / selector_std

            # Combine normalized losses with weighting
            final_loss = mse_loss + self.rl_loss_weight * selector_loss

            result = {
                "loss": final_loss,
                "mse_loss": mse_loss,
                "feature_acts": best_feature_acts,
                "reconstructed": best_reconstructed,
                "selector_loss": selector_loss,
                "mse_norm": mse_norm.item(),
                "selector_norm": selector_norm.item(),
                "mse_mean": self.mse_loss_mean.item(),
                "mse_std": mse_std.item(),
                "selector_mean": self.selector_loss_mean.item(),
                "selector_std": selector_std.item(),
                "temperature": self.current_temperature(iteration),
                "magnitude_max": best_feature_acts.max().item(),
                "magnitude_min": best_feature_acts.min().item(),
                "magnitude_mean": best_feature_acts.mean().item(),
                "magnitude_std": best_feature_acts.std().item(),
            }

        else:
            # Inference: deterministic threshold
            feature_acts = self.rl_selector(preacts)
            reconstructed = self._decode(feature_acts)
            mse_loss = (reconstructed - x).pow(2).mean()
            loss = mse_loss  # plus any optional penalty

            result = {
                "loss": loss,
                "mse_loss": mse_loss,
                "feature_acts": feature_acts,
                "reconstructed": reconstructed,
            }

        result["preacts_mean"] = preacts.mean()
        result["preacts_std"] = preacts.std()
        result["preacts_min"] = preacts.min()
        result["preacts_max"] = preacts.max()

        # Add magnitude parameter statistics
        result["magnitude_scalar_min"] = self.rl_selector.magnitude_scalar.min().item()
        result["magnitude_scalar_max"] = self.rl_selector.magnitude_scalar.max().item()
        result["magnitude_bias_min"] = self.rl_selector.magnitude_bias.min().item()
        result["magnitude_bias_max"] = self.rl_selector.magnitude_bias.max().item()

        return result

    def optimize(self, x, optimizer, iteration=None):
        with torch.no_grad():
            preacts = self._get_preacts(x)
            probs = self.rl_selector.get_probs(
                preacts, temperature=self.current_temperature(iteration)
            )

            sorted_probs = torch.sort(probs.flatten())[0]
            n = sorted_probs.size(0)
            index = torch.arange(1, n + 1, device=preacts.device)
            gini = (2 * (index * sorted_probs).sum() / (n * sorted_probs.sum())) - (
                n + 1
            ) / n

            avg_prob = probs.mean()
            deviation = (probs - self.rl_selector.base_prob).pow(2).mean()

        # Store the first result for returning
        first_result = None

        # Run multiple optimization steps on the same batch
        for step in range(self.optimize_steps):
            # Get forward results
            result = self.forward(x, iteration=iteration)
            loss = result["loss"]

            # Save the first result to return
            if step == 0:
                first_result = result

            # Backpropagate
            loss.backward()

            # Process gradients (gradient clipping, etc.)
            self.process_gradients()

            # Update parameters
            optimizer.step()
            optimizer.zero_grad()

        first_result["optimizer_step"] = optimizer.get_step()
        optimizer_exp_avg = optimizer.state[list(optimizer.state.keys())[0]]["exp_avg"]
        first_result["optimizer_exp_avg_mean"] = optimizer_exp_avg.mean()
        optimizer_exp_avg_sq = optimizer.state[list(optimizer.state.keys())[0]][
            "exp_avg_sq"
        ]
        first_result["optimizer_exp_avg_sq_mean"] = optimizer_exp_avg_sq.mean()

        result = first_result

        result["gini"] = gini.item()
        result["prob_penalty"] = deviation.item()
        result["prob_mean"] = avg_prob.item()
        result["prob_min"] = probs.min().item()
        result["prob_max"] = probs.max().item()
        result["prob_std"] = probs.std().item()

        return result

    def copy_tensors_(self, other):
        """
        Override to properly copy RL selector parameters and running statistics
        """
        # Copy standard parameters from the parent's method
        super().copy_tensors_(other)

        # Handle RLFeatureSelector parameters if both have it
        if hasattr(self, "rl_selector") and hasattr(other, "rl_selector"):
            # Manually copy parameters from other.rl_selector to self.rl_selector
            for self_param, other_param in zip(
                self.rl_selector.parameters(), other.rl_selector.parameters()
            ):
                self_param.data.copy_(other_param.data)

        # Copy running statistics
        self.mse_loss_mean = other.mse_loss_mean.clone()
        self.mse_loss_sq_mean = other.mse_loss_sq_mean.clone()
        self.selector_loss_mean = other.selector_loss_mean.clone()
        self.selector_loss_sq_mean = other.selector_loss_sq_mean.clone()
        self.loss_stats_initialized = other.loss_stats_initialized

    def share_memory(self):
        """
        Share memory for parallel processing, including the running statistics
        """
        super().share_memory()

        # Share memory for statistics tensors
        self.mse_loss_mean = self.mse_loss_mean.share_memory_()
        self.mse_loss_sq_mean = self.mse_loss_sq_mean.share_memory_()
        self.selector_loss_mean = self.selector_loss_mean.share_memory_()
        self.selector_loss_sq_mean = self.selector_loss_sq_mean.share_memory_()

        return self

    def clone(self):
        """
        Creates a new RLSAE instance with the same configuration and copies parameters.
        """
        # Create a new RLSAE with identical hyperparameters
        new_encoder_dim_mults = [dim / self.act_size for dim in self.encoder_dims]
        new_decoder_dim_mults = [dim / self.act_size for dim in self.decoder_dims]
        new_sparse_dim_mult = self.sparse_dim / self.act_size

        new_sae = RLSAE(
            act_size=self.act_size,
            encoder_dim_mults=new_encoder_dim_mults,
            sparse_dim_mult=new_sparse_dim_mult,
            decoder_dim_mults=new_decoder_dim_mults,
            enc_dtype=self.enc_dtype,
            device=self.device,
            num_samples=self.num_samples,
            L0_penalty=self.L0_penalty,
            rl_loss_weight=self.rl_loss_weight,
            optimizer_type=self.optimizer_type
            if hasattr(self, "optimizer_type")
            else "sparse_adam",
            optimizer_config=self.optimizer_config
            if hasattr(self, "optimizer_config")
            else None,
            optimize_steps=self.optimize_steps,
            weight_decay=self.weight_decay,
            base_L0=self.base_L0,
            initial_temperature=self.initial_temperature,
            min_temperature=self.min_temperature,
            temperature_tau=self.temperature_tau,
            loss_stats_momentum=self.loss_stats_momentum,
            eval_batch_size=self.eval_batch_size,
        )

        # Copy parameter data from the current model (includes running statistics)
        new_sae.copy_tensors_(self)

        return new_sae

    def to(self, *args, **kwargs):
        """
        Override to move running statistics to the target device/dtype
        """
        result = super().to(*args, **kwargs)

        # Move running statistics to the same device as the model
        if hasattr(self, "mse_loss_mean"):
            self.mse_loss_mean = self.mse_loss_mean.to(self.device)
            self.mse_loss_sq_mean = self.mse_loss_sq_mean.to(self.device)
            self.selector_loss_mean = self.selector_loss_mean.to(self.device)
            self.selector_loss_sq_mean = self.selector_loss_sq_mean.to(self.device)

        return result

    def get_config_dict(self):
        # Get base configuration from parent class
        config = super().get_config_dict()

        # Add RL-specific parameters
        config.update(
            {
                "num_samples": self.num_samples,
                "L0_penalty": self.L0_penalty,
                "rl_loss_weight": self.rl_loss_weight,
                "base_L0": self.base_L0,
                "initial_temperature": self.initial_temperature,
                "min_temperature": self.min_temperature,
                "temperature_tau": self.temperature_tau,
                "loss_stats_momentum": self.loss_stats_momentum,
                "eval_batch_size": self.eval_batch_size,
            }
        )

        return config
