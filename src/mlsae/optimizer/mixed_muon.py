import math
import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple, Union, Iterable


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2  # batched Muon implementation
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # Try to use bfloat16 if available, otherwise use float32
    try:
        X = G.bfloat16()
    except RuntimeError:
        X = G.float()  # Fallback to float32
        
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic computation strategy
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class MixedMuon(torch.optim.Optimizer):
    """
    MixedMuon - A mixed optimizer that uses Muon for 2D+ parameters and AdamW for <2D parameters.
    
    Muon internally runs standard SGD-momentum and performs an orthogonalization post-processing 
    step. For parameters with less than 2 dimensions, AdamW is used instead.
    
    Args:
        params: Model parameters
        lr: Learning rate (default: 0.02 for Muon, 0.001 for AdamW)
        weight_decay: Weight decay (default: 0.01)
        momentum: Momentum factor for Muon (default: 0.95)
        nesterov: Whether to use Nesterov momentum for Muon (default: True)
        ns_steps: Number of Newton-Schulz iteration steps (default: 5)
        adam_betas: Betas for AdamW (default: (0.9, 0.999))
        adam_eps: Epsilon for AdamW (default: 1e-8)
        rank: Current process rank for distributed training (default: 0)
        world_size: Total world size for distributed training (default: 1)
    """
    
    def __init__(
        self, 
        params: Iterable[torch.Tensor],
        lr: Union[float, Dict[str, float]] = {"muon": 0.02, "adam": 0.001},
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
        rank: int = 0,
        world_size: int = 1
    ):
        self.rank = rank
        self.world_size = world_size
        
        # Convert lr to dict if it's a float
        if isinstance(lr, float):
            lr = {"muon": lr, "adam": lr}
            
        # Set defaults
        default_lr = {"muon": 0.02, "adam": 0.001}
        for key in default_lr:
            if key not in lr:
                lr[key] = default_lr[key]
                
        # Separate parameters into 2D+ (for Muon) and <2D (for AdamW)
        muon_params = []
        adam_params = []
        
        params = list(params)  # Convert to list to allow multiple iterations
        
        for p in params:
            if p.ndim >= 2:
                muon_params.append(p)
            else:
                adam_params.append(p)
                
        # Store param classification for later use
        self.param_type = {}
        for p in muon_params:
            self.param_type[p] = "muon"
        for p in adam_params:
            self.param_type[p] = "adam"
            
        # Initialize the internal optimizers
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "adam_betas": adam_betas,
            "adam_eps": adam_eps
        }
        
        # Initialize parameter groups for the parent optimizer
        param_groups = []
        
        # Add Muon param groups
        if muon_params:
            muon_sizes = {p.numel() for p in muon_params}
            for size in muon_sizes:
                # Use parameter device for testing, in production would use CUDA
                device = next(p.device for p in muon_params if p.numel() == size)
                # Use bfloat16 if available, otherwise fallback to float32
                try:
                    b = torch.empty(world_size, size, dtype=torch.bfloat16, device=device)
                except RuntimeError:
                    b = torch.empty(world_size, size, dtype=torch.float32, device=device)
                    
                group = {
                    "params": [p for p in muon_params if p.numel() == size],
                    "lr": lr["muon"],
                    "weight_decay": weight_decay,
                    "momentum": momentum,
                    "nesterov": nesterov,
                    "ns_steps": ns_steps,
                    "optimizer_type": "muon",
                    "update_buffer": b,
                    "update_buffer_views": [b[i] for i in range(world_size)]
                }
                param_groups.append(group)
        
        # Add AdamW param groups
        if adam_params:
            group = {
                "params": adam_params,
                "lr": lr["adam"],
                "weight_decay": weight_decay,
                "betas": adam_betas,
                "eps": adam_eps,
                "optimizer_type": "adam"
            }
            param_groups.append(group)
        
        super().__init__(param_groups, defaults)
        
        # Initialize state for AdamW parameters
        for group in self.param_groups:
            if group["optimizer_type"] == "adam":
                for p in group["params"]:
                    state = self.state[p]
                    state["step"] = torch.zeros(1, dtype=torch.long, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
    
    def get_step(self):
        """Returns the step of the first parameter encountered"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    if "step" in self.state[p]:
                        return self.state[p]["step"]
        return torch.tensor([0], dtype=torch.long)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        # Step for Muon parameter groups
        for group in self.param_groups:
            if group["optimizer_type"] == "muon":
                update_buffer = group["update_buffer"]
                update_buffer_views = group["update_buffer_views"]
                
                # Generate weight updates in distributed fashion
                params = group["params"]
                handle = None
                params_world = None
                
                def update_prev():
                    """Update previous parameters with gathered gradients"""
                    handle.wait()
                    for p_world, g_world in zip(params_world, update_buffer_views):
                        p_world.mul_(1 - group["lr"] * group["weight_decay"])
                        p_world.add_(
                            g_world.view_as(p_world),
                            alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5
                        )
                
                # Process parameters in chunks of world_size
                for base_i in range(len(params))[::self.world_size]:
                    if base_i + self.rank < len(params):
                        p = params[base_i + self.rank]
                        g = p.grad
                        assert g is not None
                        
                        state = self.state[p]
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(g)
                            
                        buf = state["momentum_buffer"]
                        buf.lerp_(g, 1 - group["momentum"])
                        g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                        
                        if g.ndim == 4:  # For conv filters
                            g = g.view(len(g), -1)
                            
                        g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                    else:
                        g = update_buffer_views[self.rank]
                        
                    if base_i > 0:
                        update_prev()
                        
                    # Handle distributed case for GPU; for CPU testing just copy
                    if dist.is_available() and dist.is_initialized():
                        handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                    else:
                        # In CPU testing mode, just copy the tensor directly
                        update_buffer_views[0].copy_(g)
                        class MockHandle:
                            def wait(self):
                                pass
                        handle = MockHandle()
                    params_world = params[base_i:base_i + self.world_size]
                    
                if base_i > 0 or len(params) > 0:
                    update_prev()
        
        # Step for AdamW parameter groups
        for group in self.param_groups:
            if group["optimizer_type"] == "adam":
                for p in group["params"]:
                    if p.grad is None:
                        continue
                        
                    grad = p.grad
                    state = self.state[p]
                    
                    # Get optimizer state
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    step = state["step"]
                    
                    # Increment step
                    step += 1
                    
                    # AdamW update
                    beta1, beta2 = group["betas"]
                    
                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                    
                    # Bias corrections
                    bias_correction1 = 1 - beta1 ** step.item()
                    bias_correction2 = 1 - beta2 ** step.item()
                    
                    # Apply weight decay before the optimizer step
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])
                    
                    # Compute step size
                    step_size = group["lr"] / bias_correction1
                    
                    # Update parameters
                    p.data.addcdiv_(
                        exp_avg,
                        exp_avg_sq.sqrt().add_(group["eps"]) / math.sqrt(bias_correction2),
                        value=-step_size
                    )
                    
        return loss
                
    def share_memory_(self):
        """
        Moves all state tensors to shared memory for multiprocessing.
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p, None)
                if not state:
                    continue
                    
                # Share common state tensors
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key].share_memory_()
                
            # Share update buffer for Muon if present
            if group.get("optimizer_type") == "muon" and "update_buffer" in group:
                group["update_buffer"].share_memory_()
                
        return self
        
    def to(self, *args, **kwargs):
        """
        Moves all optimizer state tensors to the specified device/dtype.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(*args, **kwargs)
                            
            # Move update buffer for Muon if present
            if group.get("optimizer_type") == "muon" and "update_buffer" in group:
                group["update_buffer"] = group["update_buffer"].to(*args, **kwargs)
                group["update_buffer_views"] = [
                    group["update_buffer"][i] for i in range(self.world_size)
                ]
                
        return self
        
    def copy_state_from(self, source_optimizer):
        """
        Copy optimizer state from source optimizer.
        
        Args:
            source_optimizer: The optimizer to copy state from
            
        Returns:
            self with updated state
            
        Raises:
            ValueError: If optimizers have different param group sizes or parameter shapes
        """
        if len(source_optimizer.param_groups) != len(self.param_groups):
            raise ValueError(
                "Cannot copy between optimizers with different param group sizes."
            )
        
        for group_source, group_target in zip(source_optimizer.param_groups, self.param_groups):
            # Copy parameter-specific state
            for p_source, p_target in zip(group_source["params"], group_target["params"]):
                if p_target.requires_grad:
                    if p_target.data.size() != p_source.data.size():
                        raise ValueError("Parameter size mismatch between optimizers.")
                    
                    # Get state dictionaries for both parameters
                    state_source = source_optimizer.state[p_source]
                    state_target = self.state[p_target]
                    
                    # Copy each state tensor
                    for key in state_source:
                        if torch.is_tensor(state_source[key]):
                            if key not in state_target:
                                # Initialize if not present in target (should not happen normally)
                                state_target[key] = torch.zeros_like(state_source[key])
                            # Copy tensor data
                            state_target[key].copy_(state_source[key])
                        else:
                            # Copy non-tensor state (e.g., scalars)
                            state_target[key] = state_source[key]
            
            # Copy group-specific buffers (for MixedMuon's update buffer)
            if "optimizer_type" in group_source and group_source["optimizer_type"] == "muon":
                if "update_buffer" in group_source and "update_buffer" in group_target:
                    # Copy update buffer data
                    group_target["update_buffer"].copy_(group_source["update_buffer"])
                    
                    # Re-create views if needed
                    if "update_buffer_views" in group_target:
                        group_target["update_buffer_views"] = [
                            group_target["update_buffer"][i] 
                            for i in range(len(group_target["update_buffer_views"]))
                        ]
        
        return self