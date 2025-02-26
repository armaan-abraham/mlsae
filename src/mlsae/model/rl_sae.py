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

        # Sample multiple masks - shape: [num_samples, batch_size, sparse_dim]
        masks = torch.stack([self.sample_mask(probs) for _ in range(num_samples)])

        return masks

    def apply_mask(self, x, mask):
        """Apply a specific mask to the input"""
        magnitudes = F.relu(x)
        output = magnitudes * mask * self.feature_scales.unsqueeze(0)
        return output

    def update_selector(self, best_mask):
        """
        Update the selector to increase probability of selecting the best mask
        """
        if not hasattr(self, "saved_probs"):
            return 0.0

        # Calculate log-probabilities of the best mask
        log_probs = torch.log(self.saved_probs + 1e-8) * best_mask + torch.log(
            1 - self.saved_probs + 1e-8
        ) * (1 - best_mask)

        # Simple loss to maximize probability of the best mask
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
        # Initialize with parent class but don't add TopKActivation
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
        """
        Overriding to remove TopKActivation - we'll add RL feature selector separately
        """
        self.dense_encoder_blocks = torch.nn.ModuleList()
        in_dim = self.act_size

        for dim in self.encoder_dims:
            linear_layer = self._create_linear_layer(in_dim, dim)
            self.dense_encoder_blocks.append(
                torch.nn.Sequential(linear_layer, nn.ReLU())
            )
            in_dim = dim

        # Create the sparse encoder without activation (we'll apply RL selection separately)
        self.sparse_encoder_block = torch.nn.Sequential(
            self._create_linear_layer(in_dim, self.sparse_dim),
        )

        # Feature selector
        self.rl_selector = RLFeatureSelector(
            sparse_dim=self.sparse_dim,
            temperature=self.rl_temperature,
            L0_penalty=self.L0_penalty,
            num_samples=self.num_samples,
        )

    def _encode_up_to_selection(self, x):
        """Encode the input up to the point of feature selection"""
        resid = x
        # Pass through dense encoder blocks
        if self.encoder_dims:
            for block in self.dense_encoder_blocks:
                resid = block(resid)

        # Get raw sparse encoder outputs
        sparse_features = self.sparse_encoder_block(resid)
        return sparse_features

    def _decode(self, feature_acts):
        """Decode from the feature activations"""
        resid = feature_acts
        for block in self.decoder_blocks:
            resid = block(resid)
        return resid

    def _evaluate_sample(self, x, sparse_features, mask):
        """Evaluate a single feature selection sample"""
        # Apply mask to get feature activations
        feature_acts = self.rl_selector.apply_mask(sparse_features, mask)

        # Decode and compute losses
        reconstructed = self._decode(feature_acts)

        # MSE reconstruction loss
        mse_loss = (reconstructed.float() - x.float()).pow(2).mean()

        assert torch.all((mask == 0) | (mask == 1)), "Mask must be binary"
        # Sparsity penalty - encourage using fewer features
        sparsity_penalty = torch.mean(mask.sum(dim=1)) * self.L0_penalty

        # Regularization loss for feature magnitudes
        act_mag = feature_acts.pow(2).mean()
        act_mag_loss = act_mag * self.act_decay

        # Total loss/reward
        loss = mse_loss + sparsity_penalty + act_mag_loss

        return {
            "loss": loss,
            "mse_loss": mse_loss,
            "feature_acts": feature_acts,
            "reconstructed": reconstructed,
            "act_mag": act_mag,
            "sparsity_penalty": sparsity_penalty,
            "mask": mask,
        }

    def _forward(self, x):
        # Encode up to the point of selection
        sparse_features = self._encode_up_to_selection(x)

        if self.training:
            # Sample multiple feature selections
            masks = self.rl_selector.sample_masks(sparse_features)

            # Evaluate each sample
            results = []
            for i in range(masks.shape[0]):
                mask = masks[i]
                result = self._evaluate_sample(x, sparse_features, mask)
                results.append(result)

            # Find the best sample (lowest loss)
            best_idx = min(range(len(results)), key=lambda i: results[i]["loss"])
            best_result = results[best_idx]

            # Compute selector loss to increase probability of best mask
            selector_loss = self.rl_selector.update_selector(best_result["mask"])

            # Add weighted selector loss to the total
            weighted_selector_loss = selector_loss * self.rl_loss_weight
            best_result["loss"] = best_result["loss"] + weighted_selector_loss
            best_result["selector_loss"] = selector_loss

            return (
                best_result["loss"],
                best_result["act_mag"],
                best_result["mse_loss"],
                best_result["feature_acts"],
                best_result["reconstructed"],
                best_result["selector_loss"],
            )
        else:
            # In inference mode, just use the deterministic forward
            feature_acts = self.rl_selector(sparse_features)
            reconstructed = self._decode(feature_acts)

            # Calculate losses for metrics
            mse_loss = (reconstructed.float() - x.float()).pow(2).mean()
            act_mag = feature_acts.pow(2).mean()
            act_mag_loss = act_mag * self.act_decay

            loss = mse_loss + act_mag_loss

            return (
                loss,
                act_mag,
                mse_loss,
                feature_acts,
                reconstructed,
                torch.tensor(0.0),
            )

    def forward(self, x):
        loss, act_mag, mse_loss, feature_acts, reconstructed, selector_loss = (
            self._forward(x)
        )

        # Tracking logic
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
        # Handle the encoder weights
        new_W_enc = torch.zeros_like(self.sparse_encoder_block[0].weight)
        new_W_dec = torch.zeros_like(self.decoder_blocks[0].weight)
        nn.init.kaiming_normal_(new_W_enc)
        nn.init.kaiming_normal_(new_W_dec)

        new_b_enc = torch.zeros_like(self.sparse_encoder_block[0].bias)

        self.sparse_encoder_block[0].weight.data[idx] = new_W_enc[idx]
        self.sparse_encoder_block[0].bias.data[idx] = new_b_enc[idx]
        self.decoder_blocks[0].weight.data[:, idx] = new_W_dec[:, idx]

        # Also reset the feature scales for resampled features
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
