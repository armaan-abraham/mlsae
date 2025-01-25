import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlsae.data import DTYPES

model_dir = Path(__file__).parent / "checkpoints"

ZERO_ACT_THRESHOLD = 0


class DeepSAE(nn.Module):
    """
    Multi-layer sparse autoencoder with a single sparse representation layer,
    where we replace L1 regularization with a top-k activation mask.

    Statistics for the feature activations can be tracked only after calling
    the start_act_stat_tracking() method. If that method is called, we accumulate
    the sum of feature activations, the sum of their squares, and the total element
    count across all forward() calls. We also track the running sum of MSE losses
    over calls to forward(). These can then be used to compute the overall mean,
    std, and mean MSE via get_activation_stats().
    """

    def __init__(
        self,
        encoder_dim_mults: list[int],
        sparse_dim_mult: int,
        decoder_dim_mults: list[int],
        act_size: int,
        name: str = None,
        enc_dtype: str = "fp32",
        device: str = "cuda:0",
        l1_lambda: float = 0.1,
    ):
        super().__init__()

        assert all(mult == 1 for mult in encoder_dim_mults)
        self.name = name
        self.encoder_dims = [dim * act_size for dim in encoder_dim_mults]
        self.decoder_dims = [dim * act_size for dim in decoder_dim_mults]
        self.sparse_dim = sparse_dim_mult * act_size
        self.act_size = act_size
        self.enc_dtype = enc_dtype
        self.dtype = DTYPES[enc_dtype]
        self.device = str(device)
        self.l1_lambda = l1_lambda

        # Indicates whether we're tracking global stats of feature activations
        self.track_acts_stats = False

        # For tracking global stats of feature activations if enabled
        self.acts_sum = 0.0
        self.acts_sq_sum = 0.0
        self.acts_elem_count = 0

        logging.info(f"Encoder dims: {self.encoder_dims}")
        logging.info(f"Decoder dims: {self.decoder_dims}")
        logging.info(f"Sparse dim: {self.sparse_dim}")
        logging.info(f"L1 lambda: {self.l1_lambda}")
        logging.info(f"Device: {self.device}")
        logging.info(f"Dtype: {self.dtype}")

        # Parameter groups for L2: stored in class-level lists
        self.params_with_decay = []
        self.params_no_decay = []

        # --------------------------------------------------------
        # Build encoder
        # --------------------------------------------------------
        encoder_layers = []
        in_dim = self.act_size

        # Bulid the repeated (Linear -> ReLU -> LayerNorm) blocks
        for dim in self.encoder_dims:
            linear_layer = self._create_linear_layer(
                in_dim, dim, apply_weight_decay=True
            )
            encoder_layers.append(linear_layer)
            encoder_layers.append(nn.Tanh())
            encoder_layers.append(nn.LayerNorm(dim))
            in_dim = dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Final encoder layer creates sparse representation
        self.sparse_layer = self._create_linear_layer(
            in_dim, self.sparse_dim, apply_weight_decay=False
        )

        # --------------------------------------------------------
        # Build decoder
        # --------------------------------------------------------
        decoder_layers = []
        out_dim = self.sparse_dim

        for dim in self.decoder_dims:
            linear_layer = self._create_linear_layer(
                out_dim, dim, apply_weight_decay=True
            )
            decoder_layers.append(linear_layer)
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.LayerNorm(dim))
            out_dim = dim

        # Final decoder layer -> output
        linear_out = self._create_linear_layer(
            out_dim, self.act_size, apply_weight_decay=False
        )
        decoder_layers.append(linear_out)

        self.decoder = nn.Sequential(*decoder_layers)

        # Initialize weights, move to device/dtype
        self.init_weights()
        self.to(self.device, self.dtype)

    def start_act_stat_tracking(self):
        """
        Turns on tracking for feature activations and MSE. If invoked multiple times,
        previously accumulated statistics are reset.
        """
        self.track_acts_stats = True
        self.acts_sum = 0.0
        self.acts_sq_sum = 0.0
        self.acts_elem_count = 0
        self.mse_sum = 0.0
        self.mse_count = 0

    def _create_linear_layer(self, in_dim, out_dim, apply_weight_decay: bool):
        """
        Creates a Linear(in_dim, out_dim) and assigns the weight to
        params_with_decay or params_no_decay accordingly. The bias
        is always placed in params_no_decay (commonly done in PyTorch).
        """
        layer = nn.Linear(in_dim, out_dim)

        if apply_weight_decay:
            self.params_with_decay.append(layer.weight)
        else:
            self.params_no_decay.append(layer.weight)

        if layer.bias is not None:
            self.params_no_decay.append(layer.bias)

        return layer

    def get_param_groups(self, weight_decay=1e-4):
        """
        Return parameter groups for the optimizer:
          - One group with weight_decay
          - One group without weight_decay
        """
        return [
            {"params": self.params_with_decay, "weight_decay": weight_decay},
            {"params": self.params_no_decay, "weight_decay": 0.0},
        ]

    def init_weights(self):
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        # Encode
        if self.encoder_dims:
            resid = self.encoder(x)
            resid += x
        else:
            resid = x

        feature_acts = F.relu(self.sparse_layer(resid))

        # Decode
        reconstructed = self.decoder(feature_acts)

        # MSE reconstruction loss
        mse_loss = (reconstructed.float() - x.float()).pow(2).mean()

        # L1 penalty on feature activations
        l1_loss = self.l1_lambda * feature_acts.abs().mean()
        loss = mse_loss + l1_loss

        # Number of nonzero activations (for monitoring)
        nonzero_acts = (feature_acts > ZERO_ACT_THRESHOLD).float().sum(dim=1).mean()

        # Optionally track activation stats and MSE
        if self.track_acts_stats:
            fa_float = feature_acts.float()
            self.acts_sum += fa_float.sum().item()
            self.acts_sq_sum += (fa_float**2).sum().item()
            self.acts_elem_count += fa_float.numel()

            self.mse_sum += mse_loss.item()
            self.mse_count += 1

        return loss, mse_loss, l1_loss, nonzero_acts, feature_acts, reconstructed

    def get_activation_stats(self):
        """
        Returns the overall mean and std of the feature activations
        and the average MSE across all forward() calls, if tracking is active.
        Returns None if no data has been accumulated or tracking is off.
        """
        if (
            not self.track_acts_stats
            or self.acts_elem_count == 0
            or self.mse_count == 0
        ):
            return None

        mean = self.acts_sum / self.acts_elem_count
        var = (self.acts_sq_sum / self.acts_elem_count) - (mean**2)
        std = var**0.5 if var > 0 else 0.0
        avg_mse = self.mse_sum / self.mse_count
        return mean, std, avg_mse

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        # Keep final decoder layer weights unit norm in each row
        if hasattr(self.decoder[-1], "weight"):
            w = self.decoder[-1].weight
            w_normed = w / w.norm(dim=-1, keepdim=True)
            if w.grad is not None:
                w_dec_grad_proj = (w.grad * w_normed).sum(-1, keepdim=True) * w_normed
                w.grad -= w_dec_grad_proj
            w.data = w_normed

    def save(self, architecture_name, model_id=None, save_to_s3=False):
        """
        Wrapper that delegates to the save_model function in save.py
        """
        from mlsae.save import save_model

        save_model(
            model=self,
            architecture_name=architecture_name,
            model_id=model_id,
            save_to_s3=save_to_s3,
        )

    @classmethod
    def load(cls, architecture_name, model_id=None, load_from_s3=False):
        """
        Wrapper that delegates to the load_model function in save.py
        """
        from mlsae.save import load_model

        return load_model(
            cls=cls,
            architecture_name=architecture_name,
            model_id=model_id,
            load_from_s3=load_from_s3,
        )

    @torch.no_grad()
    def resample_sparse_features(self, idx):
        logging.info("Resampling sparse features...")
        new_W_enc = torch.zeros_like(self.encoder[-1].weight)
        new_W_dec = torch.zeros_like(self.decoder[0].weight)
        nn.init.kaiming_normal_(new_W_enc)
        nn.init.kaiming_normal_(new_W_dec)

        new_b_enc = torch.zeros_like(self.encoder[-1].bias)

        self.encoder[-1].weight.data[idx] = new_W_enc[idx]
        self.encoder[-1].bias.data[idx] = new_b_enc[idx]
        self.decoder[0].weight.data[:, idx] = new_W_dec[:, idx]

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = self.sparse_layer.weight.device
        self.dtype = self.sparse_layer.weight.dtype
