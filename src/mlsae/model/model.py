import json
import logging
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlsae.config import DTYPES

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


# TODO check imports of DeepSAE
class DeepSAE(nn.Module):
    def __init__(
        self,
        act_size: int,
        encoder_dim_mults: list[float],
        sparse_dim_mult: float,
        decoder_dim_mults: list[float],
        name: str = None,
        enc_dtype: str = "fp32",
        device: str = "cpu",
        topk: int = 16,
        act_l2_coeff: float = 0.0,
        weight_decay: float = 1e-4,
        lr: float = 1e-4,
    ):
        super().__init__()

        assert not encoder_dim_mults or encoder_dim_mults[-1] == 1
        self.name = name
        self.encoder_dims = [int(dim * act_size) for dim in encoder_dim_mults]
        self.decoder_dims = [int(dim * act_size) for dim in decoder_dim_mults]
        self.sparse_dim = int(sparse_dim_mult * act_size)
        self.act_size = act_size
        self.enc_dtype = enc_dtype
        self.dtype = DTYPES[enc_dtype]
        self.device = str(device)
        self.topk = topk
        assert self.topk < self.sparse_dim, f"TopK must be less than sparse dim"
        self.act_l2_coeff = act_l2_coeff
        self.weight_decay = weight_decay
        self.lr = lr

        self.track_acts_stats = False
        # Tracking stats
        self.acts_sum = 0.0
        self.acts_sq_sum = 0.0
        self.acts_elem_count = 0

        self._init_params()

        self.to(self.device, self.dtype)

    def _init_encoder_params(self):
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
                in_dim, self.sparse_dim, apply_weight_decay=True
            ),
            nn.ReLU(),
            TopKActivation(self.topk),
        )

    def _init_decoder_params(self):
        self.decoder_blocks = nn.ModuleList()
        in_dim = self.sparse_dim

        for dim in self.decoder_dims:
            linear_layer = self._create_linear_layer(
                in_dim, dim, apply_weight_decay=True
            )
            self.decoder_blocks.append(linear_layer)
            self.decoder_blocks.append(nn.ReLU())
            in_dim = dim

        self.decoder_blocks.append(
            self._create_linear_layer(in_dim, self.act_size, apply_weight_decay=False)
        )

    def _init_params(self):
        self.params_with_decay = []
        self.params_no_decay = []

        self._init_encoder_params()
        self._init_decoder_params()

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

    def get_param_groups(self):
        """
        Return parameter groups for the optimizer:
          - One group with weight_decay
          - One group without weight_decay
        """
        return [
            {
                "params": self.params_with_decay,
                "weight_decay": self.weight_decay,
                "lr": self.lr,
            },
            {"params": self.params_no_decay, "weight_decay": 0.0, "lr": self.lr},
        ]

    def _forward(self, x):
        # Encode
        if self.encoder_dims:
            resid = x
            for block in self.dense_encoder_blocks:
                resid = block(resid)
            resid += x
        else:
            resid = x

        resid = feature_acts = self.sparse_encoder_block(resid)

        for block in self.decoder_blocks:
            resid = block(resid)

        reconstructed = resid

        # MSE reconstruction loss
        mse_loss = (reconstructed.float() - x.float()).pow(2).mean()
        l2_loss = (feature_acts**2).mean() * (self.act_l2_coeff / self.topk)

        loss = mse_loss + l2_loss

        return loss, l2_loss, mse_loss, feature_acts, reconstructed

    def forward(self, x):
        loss, l2_loss, mse_loss, feature_acts, reconstructed = self._forward(x)

        # Optionally track activation stats and MSE
        if self.track_acts_stats:
            fa_float = feature_acts.float()
            self.acts_sum += fa_float.sum().item()
            self.acts_sq_sum += (fa_float**2).sum().item()
            self.acts_elem_count += fa_float.numel()

            self.mse_sum += mse_loss.item()
            self.mse_count += 1

        return loss, l2_loss, feature_acts, reconstructed

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
    def process_gradients(self):
        self.make_decoder_weights_and_grad_unit_norm()

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        w = self.decoder_blocks[-1].weight
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

    def should_resample_sparse_features(self, idx):
        return False

    @torch.no_grad()
    def resample_sparse_features(self, idx):
        logging.info(f"Resampling sparse features {idx.sum().item()}")
        enc_layer = self.sparse_encoder_block[0]
        dec_layer = self.decoder_blocks[0]
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

    def copy_tensors_(self, other):
        """
        Only copies parameters
        """
        # Copy each parameter's data
        for self_param, other_param in zip(self.parameters(), other.parameters()):
            self_param.data.copy_(other_param.data)

    def clone(self):
        """
        Creates a new DeepSAE instance with the same architecture/configuration and
        copies over the current parameters from self.
        """
        if self.__class__ is DeepSAE:
            # Reconstruct original dimension multipliers by dividing stored absolute dims by act_size
            new_encoder_dim_mults = [dim / self.act_size for dim in self.encoder_dims]
            new_decoder_dim_mults = [dim / self.act_size for dim in self.decoder_dims]
            new_sparse_dim_mult = self.sparse_dim / self.act_size

            # Instantiate a new model with identical hyperparameters
            new_sae = DeepSAE(
                encoder_dim_mults=new_encoder_dim_mults,
                sparse_dim_mult=new_sparse_dim_mult,
                decoder_dim_mults=new_decoder_dim_mults,
                act_size=self.act_size,
                name=self.name,
                enc_dtype=self.enc_dtype,
                device=self.device,
                topk=self.topk,
                act_l2_coeff=self.act_l2_coeff,
                weight_decay=self.weight_decay,
                lr=self.lr,
            )

        else:
            logging.info(f"Cloning {self.__class__.__name__}")
            new_sae = self.__class__(
                act_size=self.act_size,
                device=self.device,
            )

        # Copy parameter data from the current model
        new_sae.copy_tensors_(self)

        return new_sae


class SparseAdam(torch.optim.Optimizer):
    """
    This optimizer performs Adam-style updates but only on gradient elements
    that are nonzero. It does not rely on torch sparse tensors.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, maximize=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, maximize=maximize)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    state["step"] = torch.zeros(
                        1, dtype=torch.long, device=p.data.device
                    )
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

    def get_step(self):
        """Returns the step of the first parameter encountered"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    return self.state[p]["step"]
        return 0

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    # This version is intended for dense gradients.
                    # Skip or raise an error as needed.
                    raise RuntimeError(
                        "DenseMaskAdam does not handle sparse gradients."
                    )

                # Identify nonzero locations in the gradient
                mask = grad != 0

                # State initialization
                state = self.state[p]

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"].add_(1)
                step = state["step"]

                # Update moments for masked entries only
                exp_avg[mask] = (
                    exp_avg[mask].mul_(beta1).add_(grad[mask], alpha=1 - beta1)
                )
                exp_avg_sq[mask] = (
                    exp_avg_sq[mask]
                    .mul_(beta2)
                    .addcmul_(grad[mask], grad[mask], value=1 - beta2)
                )

                # Bias correction
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                # Compute step size
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1

                # Update parameters at masked locations
                if not maximize:
                    p.data[mask] -= step_size * (exp_avg[mask] / denom[mask])
                else:
                    p.data[mask] += step_size * (exp_avg[mask] / denom[mask])

        return loss

    def share_memory_(self):
        """
        Moves state tensors to shared memory, allowing
        multiprocessing sharing of the optimizer state.
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p, None)
                if not state:
                    continue

                state["step"].share_memory_()
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()

        # Returning self allows for optional chaining.
        return self

    def to(self, *args, **kwargs):
        """
        Moves the optimizer state tensors to a specified device/dtype.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(*args, **kwargs)
        return self

    def copy_tensors_(self, other):
        """
        Copies the parameter groups and state from another SparseAdam instance.
        """
        if len(self.param_groups) != len(other.param_groups):
            raise ValueError(
                "Cannot copy_ from an optimizer with different param group sizes."
            )

        assert len(self.param_groups) == len(other.param_groups)

        for group_self, group_other in zip(self.param_groups, other.param_groups):
            for p_self, p_other in zip(group_self["params"], group_other["params"]):
                if p_self.requires_grad:
                    if p_self.data.size() != p_other.data.size():
                        raise ValueError("Param size mismatch in SparseAdam copy_.")
                    state = self.state[p_self]
                    state["step"].copy_(other.state[p_other]["step"])
                    state["exp_avg"].copy_(other.state[p_other]["exp_avg"])
                    state["exp_avg_sq"].copy_(other.state[p_other]["exp_avg_sq"])
