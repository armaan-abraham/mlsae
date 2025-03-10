import pytest
import torch
import torch.nn as nn

from mlsae.optimizer import MixedMuon, copy_optimizer_state


class MixedModel(nn.Module):
    """
    A model with parameters of different dimensions to test the MixedMuon optimizer.
    """
    def __init__(self, input_dim=10, hidden_dim=5, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 2D parameters
        self.bias = nn.Parameter(torch.zeros(1))     # 1D parameter
        self.scalar = nn.Parameter(torch.tensor(0.5)) # 0D parameter
        
    def forward(self, x):
        x = torch.relu(self.fc1(x) + self.bias)
        x = x * self.scalar
        return x


def test_mixed_muon_basic():
    """Test that MixedMuon properly separates parameters by dimension."""
    model = MixedModel()
    
    # Create the optimizer
    opt = MixedMuon(model.parameters(), rank=0, world_size=1)
    
    # Check parameter separation
    has_muon_group = False
    has_adam_group = False
    
    for group in opt.param_groups:
        if group["optimizer_type"] == "muon":
            has_muon_group = True
            # Check that all parameters in the muon group are 2D+
            for p in group["params"]:
                assert p.ndim >= 2, f"Parameter with {p.ndim} dimensions in muon group"
        
        if group["optimizer_type"] == "adam":
            has_adam_group = True
            # Check that all parameters in the adam group are <2D
            for p in group["params"]:
                assert p.ndim < 2, f"Parameter with {p.ndim} dimensions in adam group"
    
    assert has_muon_group, "No muon parameter group found"
    assert has_adam_group, "No adam parameter group found"


def test_mixed_muon_optimizer_state_copying():
    """Test copying optimizer state between two MixedMuon optimizers."""
    # Create two identical models
    model_source = MixedModel()
    model_target = MixedModel()
    
    # Copy parameters to ensure they're the same
    model_target.load_state_dict(model_source.state_dict())
    
    # Create optimizers
    opt_source = MixedMuon(model_source.parameters(), rank=0, world_size=1)
    opt_target = MixedMuon(model_target.parameters(), rank=0, world_size=1)
    
    # Generate some fake data and perform updates on source model
    x = torch.randn(5, 10)
    y = torch.randn(5, 5)  # Match output dimensions
    
    # Do a few optimization steps on the source
    for _ in range(3):
        opt_source.zero_grad()
        output = model_source(x)
        loss = ((output - y) ** 2).mean()
        loss.backward()
        opt_source.step()
    
    # Modify the update buffer to a known value to test copying
    for group in opt_source.param_groups:
        if "optimizer_type" in group and group["optimizer_type"] == "muon":
            if "update_buffer" in group:
                # Set to recognizable pattern for testing
                group["update_buffer"].fill_(0.42)
    
    # Copy optimizer state
    copy_optimizer_state(opt_source, opt_target)
    
    # Verify states match for each parameter
    for (p_source, p_target) in zip(model_source.parameters(), model_target.parameters()):
        if p_source.requires_grad:
            source_state = opt_source.state[p_source]
            target_state = opt_target.state[p_target]
            
            # Check that all state tensors match
            for key in source_state:
                if torch.is_tensor(source_state[key]):
                    assert torch.allclose(source_state[key], target_state[key]), \
                        f"State mismatch for key {key} in {p_source.shape} parameter"
                else:
                    assert source_state[key] == target_state[key], \
                        f"Non-tensor state mismatch for key {key}"
    
    # Verify update buffer copying
    for group_source, group_target in zip(opt_source.param_groups, opt_target.param_groups):
        if "optimizer_type" in group_source and group_source["optimizer_type"] == "muon":
            if "update_buffer" in group_source and "update_buffer" in group_target:
                # Check that update buffer was copied correctly
                assert torch.allclose(group_source["update_buffer"], group_target["update_buffer"]), \
                    "Update buffer not copied correctly"
                
                # Check that views are correctly pointing to the buffer
                for i, view in enumerate(group_target["update_buffer_views"]):
                    assert torch.allclose(view, group_target["update_buffer"][i]), \
                        f"Update buffer view {i} not correctly linked to update buffer"


def test_mixed_muon_different_param_sizes():
    """Test that copying state raises an error with different parameter sizes."""
    # Create two models with different parameter sizes
    model_source = MixedModel(input_dim=10, hidden_dim=5)
    model_target = MixedModel(input_dim=12, hidden_dim=6)  # Different dimensions
    
    # Create optimizers
    opt_source = MixedMuon(model_source.parameters(), rank=0, world_size=1)
    opt_target = MixedMuon(model_target.parameters(), rank=0, world_size=1)
    
    # Update source optimizer to ensure state is initialized
    x = torch.randn(5, 10)
    y = torch.randn(5, 5)  # Match output dimensions
    opt_source.zero_grad()
    output = model_source(x)
    loss = ((output - y) ** 2).mean()
    loss.backward()
    opt_source.step()
    
    # Verify that copying state raises an error due to mismatched parameter sizes
    with pytest.raises(ValueError) as excinfo:
        copy_optimizer_state(opt_source, opt_target)
    
    assert "Parameter size mismatch" in str(excinfo.value)


def test_mixed_muon_optimization():
    """Test that MixedMuon actually optimizes parameters of all dimensions."""
    model = MixedModel()
    
    # Create the optimizer
    opt = MixedMuon(model.parameters(), rank=0, world_size=1)
    
    # Generate some fake data
    x = torch.randn(20, 10)
    y = torch.randn(20, 5)
    
    # Record initial parameter values
    initial_params = {}
    for name, p in model.named_parameters():
        initial_params[name] = p.clone().detach()
    
    # Train for a few steps
    for _ in range(5):
        opt.zero_grad()
        output = model(x)
        loss = ((output - y) ** 2).mean()
        loss.backward()
        opt.step()
    
    # Verify that all parameters have changed
    for name, p in model.named_parameters():
        assert not torch.allclose(p, initial_params[name]), \
            f"Parameter {name} with shape {p.shape} and dim {p.ndim} did not change during optimization"