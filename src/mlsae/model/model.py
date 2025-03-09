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
        if self.k >= x.size(-1):
            return x
        # Otherwise, keep only top k
        topk_vals, topk_idx = torch.topk(x, self.k, dim=-1)
        mask = torch.zeros_like(x)
        mask.scatter_(-1, topk_idx, 1.0)
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
        act_squeeze: float = 0,
        lr: float = 1e-4,
    ):
        super().__init__()

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
        self.act_squeeze = act_squeeze
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
            linear_layer = self._create_linear_layer(in_dim, dim)
            self.dense_encoder_blocks.append(
                torch.nn.Sequential(linear_layer, nn.ReLU(), nn.LayerNorm(dim))
            )
            in_dim = dim

        self.sparse_encoder_block = torch.nn.Sequential(
            self._create_linear_layer(in_dim, self.sparse_dim),
            nn.ReLU(),
            TopKActivation(self.topk),
        )

    def _init_decoder_params(self):
        self.decoder_blocks = nn.ModuleList()
        in_dim = self.sparse_dim

        for dim in self.decoder_dims:
            linear_layer = self._create_linear_layer(in_dim, dim, normalize=True)
            self.decoder_blocks.append(linear_layer)
            self.decoder_blocks.append(nn.ReLU())
            in_dim = dim

        self.decoder_blocks.append(
            self._create_linear_layer(in_dim, self.act_size, normalize=True)
        )

    def _init_params(self):
        self._init_encoder_params()
        self._init_decoder_params()
        self._tie_weights()

    def _tie_weights(self):
        # Get the encoder linear layers:
        # For each dense encoder block, the linear layer is the first module in the Sequential.
        encoder_linears = [block[0] for block in self.dense_encoder_blocks]
        # Append the linear layer from the sparse encoder block (its first module)
        encoder_linears.append(self.sparse_encoder_block[0])

        # Get the decoder linear layers.
        # Note that in _init_decoder_params, we add linear layers interleaved with ReLU.
        # Extract only the Linear modules.
        decoder_linears = [
            module for module in self.decoder_blocks if isinstance(module, nn.Linear)
        ]

        # Pair layers until either encoder or decoder runs out
        # Last encoder layer pairs with first decoder layer, etc.
        num_pairs = min(len(encoder_linears), len(decoder_linears))
        for i in range(num_pairs):
            enc_layer = encoder_linears[-(i + 1)]  # Start from last encoder
            dec_layer = decoder_linears[i]  # Start from first decoder
            dec_layer.weight.data.copy_(enc_layer.weight.data.t())
            dec_layer.weight.data = dec_layer.weight.data / dec_layer.weight.data.norm(
                dim=-1, keepdim=True
            )

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

    def _create_linear_layer(self, in_dim, out_dim, normalize=False):
        layer = nn.Linear(in_dim, out_dim)
        nn.init.kaiming_normal_(layer.weight)
        if normalize:
            layer.weight.data = layer.weight.data / layer.weight.data.norm(
                dim=-1, keepdim=True
            )
        nn.init.zeros_(layer.bias)
        return layer

    def get_param_groups(self):
        return [
            {
                "params": self.parameters(),
                "lr": self.lr,
            },
        ]

    def _forward(self, x, iteration=None):
        # Encode
        resid = x
        if self.encoder_dims:
            for block in self.dense_encoder_blocks:
                resid = block(resid)
            resid = resid / torch.sqrt(torch.tensor(resid.shape[-1]))
        
        # Access pre-topk activations from sparse_encoder_block
        # sparse_encoder_block is Sequential(Linear, ReLU, TopKActivation)
        linear_out = self.sparse_encoder_block[0](resid)  # Linear output
        relu_out = self.sparse_encoder_block[1](linear_out)  # After ReLU, before TopK
        
        # Get pre-topk activations for inflation loss
        pre_topk_acts = relu_out
        
        # Mean activation across batch for each feature
        mean_acts_per_feature = pre_topk_acts.mean(dim=0)
        
        mean_acts_std = mean_acts_per_feature.std()
        
        act_squeeze_loss = mean_acts_std * self.act_squeeze
        
        # Continue with topk activation and the rest of the network
        feature_acts = self.sparse_encoder_block[2](relu_out)  # Apply TopK
        resid = feature_acts
        
        assert (
            (feature_acts == 0).float().sum(dim=-1) >= (self.sparse_dim - self.topk)
        ).all()

        for block in self.decoder_blocks:
            resid = block(resid)

        reconstructed = resid

        # MSE reconstruction loss
        mse_loss = (reconstructed.float() - x.float()).pow(2).mean()
        
        loss = mse_loss + act_squeeze_loss

        return {
            "loss": loss,
            "mse_loss": mse_loss,
            "feature_acts": feature_acts,
            "reconstructed": reconstructed,
            "act_squeeze_loss": act_squeeze_loss,
        }

    def forward(self, x, iteration=None):
        result = self._forward(x, iteration)

        # Optionally track activation stats and MSE
        if self.track_acts_stats:
            fa_float = result["feature_acts"].float()
            self.acts_sum += fa_float.sum().item()
            self.acts_sq_sum += (fa_float**2).sum().item()
            self.acts_elem_count += fa_float.numel()

            self.mse_sum += result["mse_loss"].item()
            self.mse_count += 1

        return result

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
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

    def _make_decoder_weights_and_grad_unit_norm(self, weight):
        w_normed = weight / weight.norm(dim=-1, keepdim=True)
        if weight.grad is not None:
            w_dec_grad_proj = (weight.grad * w_normed).sum(-1, keepdim=True) * w_normed
            weight.grad -= w_dec_grad_proj
        weight.data = w_normed

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        for block in self.decoder_blocks:
            if isinstance(block, nn.Linear):
                self._make_decoder_weights_and_grad_unit_norm(block.weight)

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

    def get_config_dict(self):
        return {
            "encoder_dims": self.encoder_dims,
            "decoder_dims": self.decoder_dims,
            "sparse_dim": self.sparse_dim,
            "act_size": self.act_size,
            "enc_dtype": self.enc_dtype,
            "topk": self.topk,
            "act_squeeze": self.act_squeeze,
            "name": self.name,
            "lr": self.lr,
        }

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
                act_decay=self.act_decay,
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


class ExperimentSAEBase(DeepSAE):
    """Base class for all experimental SAE models.
    Automatically extracts the name from the class name (text after 'ExperimentSAE').
    """
    def __init__(self, *args, **kwargs):
        print(args, kwargs)
        # Extract model name from class name (e.g., ExperimentSAE0 -> "0")
        class_name = self.__class__.__name__
        model_name = class_name[len("ExperimentSAE"):]
        
        # Pass this to the parent class
        super().__init__(
            *args,
            name=model_name,
            **kwargs
        )
