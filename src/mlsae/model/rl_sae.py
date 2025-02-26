import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from mlsae.model.model import DeepSAE
from mlsae.config import DTYPES


class RLFeatureSelector(nn.Module):
    """
    A RL-based feature selector that samples which features to activate
    based on probabilities from a sigmoid activation.
    """

    def __init__(self, sparse_dim, temperature=1.0, L0_penalty=1e-2, num_samples=10):
        super().__init__()
        self.sparse_dim = sparse_dim
        self.temperature = temperature
        self.L0_penalty = L0_penalty
        self.feature_scales = nn.Parameter(torch.ones(sparse_dim))
        self.num_samples = num_samples

    def get_probs(self, x):
        """Get activation probabilities from raw encoder outputs"""
        return torch.sigmoid(x / self.temperature)

    def sample_mask(self, probs):
        """Sample a binary mask from probabilities"""
        return torch.bernoulli(probs)

    def forward(self, x):
        """During inference, just use deterministic threshold"""
        probs = self.get_probs(x)
        mask = (probs > 0.5).float()
        magnitudes = F.relu(x)
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
        magnitudes = F.relu(x)
        output = magnitudes * mask * self.feature_scales.unsqueeze(0)
        return output

    def update_selector(self, best_masks):
        """
        Update the selector to increase probability of selecting the best mask.
        best_masks: [batch_size, sparse_dim]
        """
        if not hasattr(self, "saved_probs"):
            return 0.0

        # self.saved_probs: [batch_size, sparse_dim]
        # best_masks: [batch_size, sparse_dim]

        # Calculate log-probabilities for each sample's best mask
        # shape [batch_size, sparse_dim]
        log_probs = (
            torch.log(self.saved_probs + 1e-8) * best_masks
            + torch.log(1 - self.saved_probs + 1e-8) * (1 - best_masks)
        )

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
        rl_temperature: float = 1.0,
        num_samples: int = 3,
        L0_penalty: float = 1e-2,
        rl_loss_weight: float = 1.0,
    ):
        self.L0_penalty = L0_penalty
        self.rl_temperature = rl_temperature
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

        self.sparse_encoder_block = torch.nn.Sequential(
            self._create_linear_layer(in_dim, self.sparse_dim),
        )

        self.rl_selector = RLFeatureSelector(
            sparse_dim=self.sparse_dim,
            temperature=self.rl_temperature,
            L0_penalty=self.L0_penalty,
            num_samples=self.num_samples,
        )

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

    def _evaluate_masks(self, x, sparse_features, masks):
        """
        Evaluate per-sample losses for each mask.
        masks: [num_samples, batch_size, sparse_dim]
        Returns a tensor of shape [num_samples, batch_size]
        with the total loss per sample for each mask.
        """
        batch_size = x.shape[0]

        # We'll accumulate a loss for each sample, for each mask
        all_losses = []

        for i in range(masks.shape[0]):
            mask_i = masks[i]  # shape [batch_size, sparse_dim]
            feature_acts = self.rl_selector.apply_mask(sparse_features, mask_i)
            reconstructed = self._decode(feature_acts)

            # MSE per sample: shape [batch_size]
            mse_loss_per_sample = (reconstructed - x).pow(2).mean(dim=1)

            # Sparsity penalty per sample: shape [batch_size]
            # sum of active features per sample, times L0 penalty
            sparsity_penalty_per_sample = mask_i.sum(dim=1) * self.L0_penalty

            # No additional activation magnitude penalty in this snippet,
            # but if you have self.act_decay, it can be added similarly per sample.
            # For example:
            # act_mag_per_sample = feature_acts.pow(2).mean(dim=1)
            # act_mag_loss_per_sample = act_mag_per_sample * self.act_decay

            # Combine losses per sample
            total_loss_per_sample = mse_loss_per_sample + sparsity_penalty_per_sample
            all_losses.append(total_loss_per_sample)

        # Stack into shape [num_samples, batch_size]
        return torch.stack(all_losses, dim=0)

    def _forward(self, x):
        sparse_features = self._encode_up_to_selection(x)

        if self.training:
            masks = self.rl_selector.sample_masks(sparse_features)
            # Evaluate each mask's per-sample loss
            per_sample_losses = self._evaluate_masks(x, sparse_features, masks)
            # per_sample_losses shape: [num_samples, batch_size]

            # For each sample, pick the mask that yields minimal loss
            best_indices = per_sample_losses.argmin(dim=0)  # [batch_size]

            # Gather the best mask for each sample
            batch_size = x.shape[0]
            best_masks = []
            for b in range(batch_size):
                best_masks.append(masks[best_indices[b], b])  # [sparse_dim]

            best_masks = torch.stack(best_masks, dim=0)  # [batch_size, sparse_dim]

            # Apply best masks and get final reconstruction
            best_feature_acts = self.rl_selector.apply_mask(sparse_features, best_masks)
            best_reconstructed = self._decode(best_feature_acts)

            # Compute the final batch-averaged MSE loss
            mse_loss = (best_reconstructed - x).pow(2).mean()

            # RL update: encourage selector to pick the best masks
            selector_loss = self.rl_selector.update_selector(best_masks)
            weighted_selector_loss = selector_loss * self.rl_loss_weight
            final_loss = mse_loss + weighted_selector_loss

            return (
                final_loss,
                torch.tensor(0.0),  # act_mag placeholder
                mse_loss,
                best_feature_acts,
                best_reconstructed,
                selector_loss,
            )

        else:
            # Inference: deterministic threshold
            feature_acts = self.rl_selector(sparse_features)
            reconstructed = self._decode(feature_acts)
            mse_loss = (reconstructed - x).pow(2).mean()
            loss = mse_loss  # plus any optional penalty

            return (
                loss,
                torch.tensor(0.0),  # act_mag placeholder
                mse_loss,
                feature_acts,
                reconstructed,
                torch.tensor(0.0),
            )

    def forward(self, x):
        loss, act_mag, mse_loss, feature_acts, reconstructed, selector_loss = (
            self._forward(x)
        )

        if self.track_acts_stats:
            fa_float = feature_acts.float()
            self.acts_sum += fa_float.sum().item()
            self.acts_sq_sum += (fa_float**2).sum().item()
            self.acts_elem_count += fa_float.numel()

            self.mse_sum += mse_loss.item()
            self.mse_count += 1

        return loss, act_mag, mse_loss, feature_acts, reconstructed

    @torch.no_grad()
    def resample_sparse_features(self, idx):
        logging.info(f"Resampling sparse features {idx.sum().item()}")
        new_W_enc = torch.zeros_like(self.sparse_encoder_block[0].weight)
        new_W_dec = torch.zeros_like(self.decoder_blocks[0].weight)
        nn.init.kaiming_normal_(new_W_enc)
        nn.init.kaiming_normal_(new_W_dec)

        new_b_enc = torch.zeros_like(self.sparse_encoder_block[0].bias)

        self.sparse_encoder_block[0].weight.data[idx] = new_W_enc[idx]
        self.sparse_encoder_block[0].bias.data[idx] = new_b_enc[idx]
        self.decoder_blocks[0].weight.data[:, idx] = new_W_dec[:, idx]

        self.rl_selector.feature_scales.data[idx] = 1.0

    def get_config_dict(self):
        config = super().get_config_dict()
        config.update(
            {
                "rl_temperature": self.rl_selector.temperature,
                "L0_penalty": self.rl_selector.L0_penalty,
                "num_samples": self.rl_selector.num_samples,
                "rl_loss_weight": self.rl_loss_weight,
            }
        )
        return config

    def should_resample_sparse_features(self, idx):
        return False
