import math
import torch


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

        for group_source, group_target in zip(
            source_optimizer.param_groups, self.param_groups
        ):
            for p_source, p_target in zip(
                group_source["params"], group_target["params"]
            ):
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

        return self
