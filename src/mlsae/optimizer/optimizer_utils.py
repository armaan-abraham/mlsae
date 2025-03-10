import torch


def copy_optimizer_state(source_optimizer, target_optimizer):
    """
    Copy optimizer state from source to target.
    
    Args:
        source_optimizer: The optimizer to copy state from
        target_optimizer: The optimizer to copy state to
        
    Returns:
        The target optimizer with updated state
        
    Raises:
        ValueError: If optimizers have different param group sizes or parameter shapes
    """
    if len(source_optimizer.param_groups) != len(target_optimizer.param_groups):
        raise ValueError(
            "Cannot copy between optimizers with different param group sizes."
        )
    
    for group_source, group_target in zip(source_optimizer.param_groups, target_optimizer.param_groups):
        # Copy parameter-specific state
        for p_source, p_target in zip(group_source["params"], group_target["params"]):
            if p_target.requires_grad:
                if p_target.data.size() != p_source.data.size():
                    raise ValueError("Parameter size mismatch between optimizers.")
                
                # Get state dictionaries for both parameters
                state_source = source_optimizer.state[p_source]
                state_target = target_optimizer.state[p_target]
                
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
    
    return target_optimizer