import json
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlsae.data import DTYPES

model_dir = Path(__file__).parent / "checkpoints"


class TopKActivation(nn.Module):
    """
    A custom activation that keeps only the top k values along dim=1
    (for each row in the batch), zeroing out the rest.
    """

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If k >= number of features, do nothing
        if self.k >= x.size(1):
            return x
        # Otherwise, keep only top k
        topk_vals, topk_idx = torch.topk(x, self.k, dim=1)
        mask = torch.zeros_like(x)
        mask.scatter_(1, topk_idx, 1.0)
        return x * mask


class DeepSAE(nn.Module):
    def __init__(
        self,
        encoder_dim_mults: list[int],
        sparse_dim_mult: int,
        decoder_dim_mults: list[int],
        act_size: int,
        name: str = None,
        enc_dtype: str = "fp32",
        device: str = "cuda:0",
        topk: int = 16,
    ):
        super().__init__()

        assert not encoder_dim_mults or encoder_dim_mults[-1] == 1
        assert not decoder_dim_mults or (decoder_dim_mults[0] == decoder_dim_mults[-1])
        self.name = name
        self.encoder_dims = [int(dim * act_size) for dim in encoder_dim_mults]
        self.decoder_dims = [int(dim * act_size) for dim in decoder_dim_mults]
        self.sparse_dim = int(sparse_dim_mult * act_size)
        self.act_size = act_size
        self.enc_dtype = enc_dtype
        self.dtype = DTYPES[enc_dtype]
        self.device = str(device)
        self.topk = topk  # save topk
        self.track_acts_stats = False

        # Tracking stats
        self.acts_sum = 0.0
        self.acts_sq_sum = 0.0
        self.acts_elem_count = 0

        logging.info(f"Encoder dims: {self.encoder_dims}")
        logging.info(f"Decoder dims: {self.decoder_dims}")
        logging.info(f"Sparse dim: {self.sparse_dim}")
        logging.info(f"TopK: {self.topk}")
        logging.info(f"Device: {self.device}")
        logging.info(f"Dtype: {self.dtype}")

        # Parameter groups for L2
        self.params_with_decay = []
        self.params_no_decay = []

        # --------------------------------------------------------
        # Build encoder
        # --------------------------------------------------------
        self.dense_encoder_blocks = torch.nn.ModuleList()
        in_dim = self.act_size

        for dim in self.encoder_dims:
            linear_layer = self._create_linear_layer(
                in_dim, dim, apply_weight_decay=True
            )
            self.dense_encoder_blocks.append(
                torch.nn.Sequential(linear_layer, nn.Tanh(), nn.LayerNorm(dim))
            )
            in_dim = dim

        self.sparse_encoder_block = torch.nn.Sequential(
            self._create_linear_layer(
                in_dim, self.sparse_dim, apply_weight_decay=False
            ),
            nn.ReLU(),
            TopKActivation(self.topk),
        )

        # --------------------------------------------------------
        # Build decoder
        # --------------------------------------------------------
        decoder_blocks = []
        in_dim = self.sparse_dim

        for dim in self.decoder_dims:
            linear_layer = self._create_linear_layer(
                in_dim, dim, apply_weight_decay=True
            )
            decoder_blocks.append(linear_layer)
            decoder_blocks.append(nn.ReLU())
            in_dim = dim

        decoder_blocks.append(
            self._create_linear_layer(in_dim, self.act_size, apply_weight_decay=False)
        )

        self.decoder = nn.Sequential(*decoder_blocks)

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
        nn.init.kaiming_normal_(layer.weight)
        nn.init.zeros_(layer.bias)

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

    def forward(self, x):
        # Encode
        if self.encoder_dims:
            resid = x
            for block in self.dense_encoder_blocks:
                resid = block(resid)
            resid += x
        else:
            resid = x

        feature_acts = self.sparse_encoder_block(resid)

        reconstructed = self.decoder(feature_acts)

        # MSE reconstruction loss
        mse_loss = (reconstructed.float() - x.float()).pow(2).mean()

        loss = mse_loss

        # Optionally track activation stats and MSE
        if self.track_acts_stats:
            fa_float = feature_acts.float()
            self.acts_sum += fa_float.sum().item()
            self.acts_sq_sum += (fa_float**2).sum().item()
            self.acts_elem_count += fa_float.numel()

            self.mse_sum += mse_loss.item()
            self.mse_count += 1

        return loss, feature_acts, reconstructed

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
        logging.info(f"Resampling sparse features {idx.sum().item()}")
        enc_layer = self.sparse_encoder_block[0]
        dec_layer = self.decoder[0]
        new_W_enc = torch.zeros_like(enc_layer.weight)
        new_W_dec = torch.zeros_like(dec_layer.weight)
        nn.init.kaiming_normal_(new_W_enc)
        nn.init.kaiming_normal_(new_W_dec)

        new_b_enc = torch.zeros_like(enc_layer.bias)

        enc_layer.weight.data[idx] = new_W_enc[idx]
        enc_layer.bias.data[idx] = new_b_enc[idx]
        dec_layer.weight.data[:, idx] = new_W_dec[:, idx]

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = self.sparse_encoder_block[0].weight.device
        self.dtype = self.sparse_encoder_block[0].weight.dtype
