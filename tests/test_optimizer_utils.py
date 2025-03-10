import pytest
import torch
import torch.nn as nn

from mlsae.optimizer import SparseAdam, copy_optimizer_state


class SimpleModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=5, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def test_copy_optimizer_state_sparse_adam():
    # Create two identical models
    model_source = SimpleModel()
    model_target = SimpleModel()
    
    # Copy parameters to ensure they're the same
    model_target.load_state_dict(model_source.state_dict())
    
    # Create optimizers
    opt_source = SparseAdam(model_source.parameters(), lr=0.01)
    opt_target = SparseAdam(model_target.parameters(), lr=0.01)
    
    # Generate some fake data and perform updates on source model
    x = torch.randn(5, 10)
    y = torch.randn(5, 2)
    
    # Do a few optimization steps on the source
    for _ in range(3):
        opt_source.zero_grad()
        output = model_source(x)
        loss = ((output - y) ** 2).mean()
        loss.backward()
        opt_source.step()
    
    # Copy optimizer state
    copy_optimizer_state(opt_source, opt_target)
    
    # Verify states match
    for (p_source, p_target) in zip(model_source.parameters(), model_target.parameters()):
        if p_source.requires_grad:
            source_state = opt_source.state[p_source]
            target_state = opt_target.state[p_target]
            
            # Check step count
            assert torch.equal(source_state["step"], target_state["step"]), "Step count mismatch"
            
            # Check exp_avg
            assert torch.allclose(source_state["exp_avg"], target_state["exp_avg"]), \
                "First moment (exp_avg) mismatch"
            
            # Check exp_avg_sq
            assert torch.allclose(source_state["exp_avg_sq"], target_state["exp_avg_sq"]), \
                "Second moment (exp_avg_sq) mismatch"


def test_copy_optimizer_state_multiple_param_groups():
    # Create two identical models
    model_source = SimpleModel()
    model_target = SimpleModel()
    
    # Copy parameters to ensure they're the same
    model_target.load_state_dict(model_source.state_dict())
    
    # Create optimizers with multiple param groups
    opt_source = SparseAdam([
        {'params': model_source.fc1.parameters(), 'lr': 0.01},
        {'params': model_source.fc2.parameters(), 'lr': 0.001}
    ])
    
    opt_target = SparseAdam([
        {'params': model_target.fc1.parameters(), 'lr': 0.01},
        {'params': model_target.fc2.parameters(), 'lr': 0.001}
    ])
    
    # Generate some fake data and perform updates on source model
    x = torch.randn(5, 10)
    y = torch.randn(5, 2)
    
    # Do a few optimization steps on the source
    for _ in range(3):
        opt_source.zero_grad()
        output = model_source(x)
        loss = ((output - y) ** 2).mean()
        loss.backward()
        opt_source.step()
    
    # Copy optimizer state
    copy_optimizer_state(opt_source, opt_target)
    
    # Verify states match for each parameter group
    for g_source, g_target in zip(opt_source.param_groups, opt_target.param_groups):
        for p_source, p_target in zip(g_source['params'], g_target['params']):
            if p_source.requires_grad:
                source_state = opt_source.state[p_source]
                target_state = opt_target.state[p_target]
                
                # Check step count
                assert torch.equal(source_state["step"], target_state["step"]), "Step count mismatch"
                
                # Check exp_avg
                assert torch.allclose(source_state["exp_avg"], target_state["exp_avg"]), \
                    "First moment (exp_avg) mismatch"
                
                # Check exp_avg_sq
                assert torch.allclose(source_state["exp_avg_sq"], target_state["exp_avg_sq"]), \
                    "Second moment (exp_avg_sq) mismatch"


def test_copy_optimizer_state_mismatched_param_groups():
    # Create two models with different parameter group counts
    model_source = SimpleModel()
    model_target = SimpleModel()
    
    # Create optimizers with different parameter group counts
    opt_source = SparseAdam([
        {'params': model_source.fc1.parameters(), 'lr': 0.01},
        {'params': model_source.fc2.parameters(), 'lr': 0.001}
    ])
    
    opt_target = SparseAdam(model_target.parameters(), lr=0.01)
    
    # Verify that copying state raises an error due to mismatched param groups
    with pytest.raises(ValueError) as excinfo:
        copy_optimizer_state(opt_source, opt_target)
    
    assert "Cannot copy between optimizers with different param group sizes" in str(excinfo.value)


def test_copy_optimizer_state_different_param_sizes():
    # Create two models with different parameter sizes
    model_source = SimpleModel(input_dim=10, hidden_dim=5)
    model_target = SimpleModel(input_dim=12, hidden_dim=6)  # Different dimensions
    
    # Create optimizers
    opt_source = SparseAdam(model_source.parameters(), lr=0.01)
    opt_target = SparseAdam(model_target.parameters(), lr=0.01)
    
    # Update source optimizer to ensure state is initialized
    x = torch.randn(5, 10)
    y = torch.randn(5, 2)
    opt_source.zero_grad()
    output = model_source(x)
    loss = ((output - y) ** 2).mean()
    loss.backward()
    opt_source.step()
    
    # Verify that copying state raises an error due to mismatched parameter sizes
    with pytest.raises(ValueError) as excinfo:
        copy_optimizer_state(opt_source, opt_target)
    
    assert "Parameter size mismatch" in str(excinfo.value)