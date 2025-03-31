import pytest
import torch
import torch.nn as nn

from mlsae.optimizer import MixedMuon


class MixedModel(nn.Module):
    """
    A model with parameters of different dimensions to test the MixedMuon optimizer.
    """

    def __init__(self, input_dim=10, hidden_dim=5, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 2D parameters
        self.bias = nn.Parameter(torch.zeros(1))  # 1D parameter
        self.scalar = nn.Parameter(torch.tensor(0.5))  # 0D parameter

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

    # Create optimizers with different learning rates to ensure hyperparameters aren't copied
    opt_source = MixedMuon(
        model_source.parameters(),
        lr={"muon": 0.03, "adam": 0.002},
        weight_decay=0.02,
        momentum=0.94,
        rank=0,
        world_size=1,
    )
    opt_target = MixedMuon(
        model_target.parameters(),
        lr={"muon": 0.01, "adam": 0.001},
        weight_decay=0.01,
        momentum=0.95,
        rank=0,
        world_size=1,
    )

    # Generate some fake data and perform updates on source model
    x = torch.randn(5, 10)
    y = torch.randn(5, 5)  # Match output dimensions

    # Do a few optimization steps on the source - use different loss scales
    # to create distinct momentum values for thorough testing
    for i in range(3):
        opt_source.zero_grad()
        output = model_source(x)
        # Vary the loss scale to create diverse state values
        loss = ((output - y) ** 2).mean() * (i + 1)
        loss.backward()
        opt_source.step()

    # Modify the update buffer to a known value to test copying
    distinct_values = {}
    for group_idx, group in enumerate(opt_source.param_groups):
        if "optimizer_type" in group and group["optimizer_type"] == "muon":
            if "update_buffer" in group:
                # Set to recognizable pattern with distinct values per group
                test_value = 0.42 + (group_idx * 0.1)
                group["update_buffer"].fill_(test_value)
                distinct_values[group_idx] = test_value

                # Verify buffer views are correctly linked before copying
                for i, view in enumerate(group["update_buffer_views"]):
                    assert torch.allclose(
                        view, group["update_buffer"][i]
                    ), f"Pre-copy: Source buffer view {i} not correctly linked to update buffer"

    # Make a copy of source optimizer state for later comparison
    source_state_copy = {}
    for group_idx, group in enumerate(opt_source.param_groups):
        for p_idx, p in enumerate(group["params"]):
            if p.requires_grad:
                param_id = (group_idx, p_idx)
                source_state_copy[param_id] = {
                    key: value.clone() if torch.is_tensor(value) else value
                    for key, value in opt_source.state[p].items()
                }

    # Copy optimizer state
    opt_target.copy_state_from(opt_source)

    # Verify parameter-level states match exactly
    for group_idx, (group_source, group_target) in enumerate(
        zip(opt_source.param_groups, opt_target.param_groups)
    ):
        for p_idx, (p_source, p_target) in enumerate(
            zip(group_source["params"], group_target["params"])
        ):
            if p_source.requires_grad:
                source_state = opt_source.state[p_source]
                target_state = opt_target.state[p_target]
                param_id = (group_idx, p_idx)

                # Check that all state tensors were copied correctly
                for key in source_state:
                    if torch.is_tensor(source_state[key]):
                        assert torch.allclose(
                            source_state[key], target_state[key]
                        ), f"State mismatch for key {key} in parameter at index ({group_idx}, {p_idx})"

                        # Verify against our manual copy to ensure no later modifications affected the test
                        assert torch.allclose(
                            source_state_copy[param_id][key], target_state[key]
                        ), f"State doesn't match original source for key {key} at index ({group_idx}, {p_idx})"
                    else:
                        assert (
                            source_state[key] == target_state[key]
                        ), f"Non-tensor state mismatch for key {key} at index ({group_idx}, {p_idx})"

    # Verify hyperparameters were NOT copied (should retain target values)
    for group_idx, (group_source, group_target) in enumerate(
        zip(opt_source.param_groups, opt_target.param_groups)
    ):
        if (
            "optimizer_type" in group_target
            and group_target["optimizer_type"] == "muon"
        ):
            assert (
                group_target["lr"] == 0.01
            ), f"Muon lr was incorrectly copied in group {group_idx}"
            assert (
                group_target["weight_decay"] == 0.01
            ), f"Weight decay was incorrectly copied in group {group_idx}"
            assert (
                group_target["momentum"] == 0.95
            ), f"Momentum was incorrectly copied in group {group_idx}"
        elif (
            "optimizer_type" in group_target
            and group_target["optimizer_type"] == "adam"
        ):
            assert (
                group_target["lr"] == 0.001
            ), f"Adam lr was incorrectly copied in group {group_idx}"
            assert (
                group_target["weight_decay"] == 0.01
            ), f"Weight decay was incorrectly copied in group {group_idx}"

    # Verify update buffer copying and view integrity
    for group_idx, (group_source, group_target) in enumerate(
        zip(opt_source.param_groups, opt_target.param_groups)
    ):
        if (
            "optimizer_type" in group_source
            and group_source["optimizer_type"] == "muon"
        ):
            if "update_buffer" in group_source and "update_buffer" in group_target:
                # Check that update buffer was copied correctly with exact values
                test_value = distinct_values.get(group_idx, 0.42)
                buffer_dtype = group_target["update_buffer"].dtype
                test_tensor = torch.tensor(
                    test_value,
                    device=group_target["update_buffer"].device,
                    dtype=buffer_dtype,
                )
                assert torch.allclose(
                    group_target["update_buffer"], test_tensor
                ), f"Update buffer not copied correctly for group {group_idx}, expected {test_value}"

                # Check that views are correctly pointing to the buffer
                for i, view in enumerate(group_target["update_buffer_views"]):
                    assert torch.allclose(
                        view, group_target["update_buffer"][i]
                    ), f"Update buffer view {i} not correctly linked in group {group_idx}"

                    # Modify view and verify changes reflect in buffer
                    test_value = 0.99
                    view.fill_(test_value)
                    test_tensor = torch.tensor(
                        test_value, device=view.device, dtype=view.dtype
                    )
                    assert torch.allclose(
                        group_target["update_buffer"][i], test_tensor
                    ), f"Modifying view {i} in group {group_idx} did not update buffer"


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
        opt_target.copy_state_from(opt_source)

    assert "Parameter size mismatch" in str(excinfo.value)


def test_mixed_muon_to_method():
    """Test that MixedMuon's to() method correctly moves all state to new device/dtype."""
    # Skip if CUDA not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping device transfer test")

    model = MixedModel()

    # Create the optimizer
    opt = MixedMuon(model.parameters(), rank=0, world_size=1)

    # Generate some fake data and perform a step to initialize state
    x = torch.randn(5, 10)
    y = torch.randn(5, 5)
    opt.zero_grad()
    output = model(x)
    loss = ((output - y) ** 2).mean()
    loss.backward()
    opt.step()

    # Set unique values in update buffers for tracking
    buffer_values = {}
    for group_idx, group in enumerate(opt.param_groups):
        if "optimizer_type" in group and group["optimizer_type"] == "muon":
            if "update_buffer" in group:
                test_value = 0.5 + (group_idx * 0.1)
                group["update_buffer"].fill_(test_value)
                buffer_values[group_idx] = test_value

    # Store original device and state references
    original_state_refs = {}
    original_buffer_refs = {}
    for group_idx, group in enumerate(opt.param_groups):
        for p_idx, p in enumerate(group["params"]):
            if p.requires_grad:
                original_state_refs[(group_idx, p_idx)] = {
                    key: id(value)
                    for key, value in opt.state[p].items()
                    if torch.is_tensor(value)
                }

        if "optimizer_type" in group and group["optimizer_type"] == "muon":
            if "update_buffer" in group:
                original_buffer_refs[group_idx] = id(group["update_buffer"])

    # Create full copies for comparison
    state_copies = {}
    for group_idx, group in enumerate(opt.param_groups):
        for p_idx, p in enumerate(group["params"]):
            if p.requires_grad:
                state_copies[(group_idx, p_idx)] = {
                    key: value.clone() if torch.is_tensor(value) else value
                    for key, value in opt.state[p].items()
                }

    # Move to CUDA
    device = torch.device("cuda:0")
    opt.to(device)

    # Verify all state tensors moved to the correct device
    for group_idx, group in enumerate(opt.param_groups):
        for p_idx, p in enumerate(group["params"]):
            if p.requires_grad:
                state = opt.state[p]
                param_id = (group_idx, p_idx)

                # Check state tensors moved to correct device
                for key, value in state.items():
                    if torch.is_tensor(value):
                        assert (
                            value.device == device
                        ), f"State tensor {key} for param ({group_idx}, {p_idx}) not moved to {device}"

                        # Verify values are preserved
                        orig_value = state_copies[param_id][key].to(device)
                        assert torch.allclose(
                            value, orig_value
                        ), f"Values not preserved for {key} in param ({group_idx}, {p_idx})"

                        # Verify new tensors were created (not just views)
                        assert (
                            id(value) != original_state_refs[param_id][key]
                        ), f"State tensor {key} for param ({group_idx}, {p_idx}) was not properly recreated"

        # Check update buffer for Muon groups
        if "optimizer_type" in group and group["optimizer_type"] == "muon":
            if "update_buffer" in group:
                # Verify buffer moved to device
                assert (
                    group["update_buffer"].device == device
                ), f"Update buffer for group {group_idx} not moved to {device}"

                # Verify buffer contains expected value
                test_value = buffer_values.get(group_idx, 0.5)
                buffer_dtype = group["update_buffer"].dtype
                test_tensor = torch.tensor(
                    test_value, device=device, dtype=buffer_dtype
                )
                assert torch.allclose(
                    group["update_buffer"], test_tensor
                ), f"Update buffer value not preserved for group {group_idx}"

                # Verify new buffer was created
                assert (
                    id(group["update_buffer"]) != original_buffer_refs[group_idx]
                ), f"Update buffer for group {group_idx} was not properly recreated"

                # Verify views point to the correct buffer
                for i, view in enumerate(group["update_buffer_views"]):
                    assert (
                        view.device == device
                    ), f"Update buffer view {i} for group {group_idx} not moved to {device}"
                    assert torch.allclose(
                        view, group["update_buffer"][i]
                    ), f"Update buffer view {i} not correctly linked in group {group_idx}"


def test_mixed_muon_share_memory():
    """Test that share_memory_() correctly moves state tensors to shared memory."""
    model = MixedModel()

    # Create the optimizer
    opt = MixedMuon(model.parameters(), rank=0, world_size=1)

    # Generate some fake data and perform a step to initialize state
    x = torch.randn(5, 10)
    y = torch.randn(5, 5)
    opt.zero_grad()
    output = model(x)
    loss = ((output - y) ** 2).mean()
    loss.backward()
    opt.step()

    # Set unique values in update buffers for tracking
    for group_idx, group in enumerate(opt.param_groups):
        if "optimizer_type" in group and group["optimizer_type"] == "muon":
            if "update_buffer" in group:
                test_value = 0.7 + (group_idx * 0.1)
                group["update_buffer"].fill_(test_value)

    # Check initial shared memory status
    initial_is_shared = {}
    for group_idx, group in enumerate(opt.param_groups):
        for p_idx, p in enumerate(group["params"]):
            if p.requires_grad:
                state = opt.state[p]
                for key, value in state.items():
                    if torch.is_tensor(value):
                        # Store initial shared memory status
                        initial_is_shared[(group_idx, p_idx, key)] = value.is_shared()

        # Check update buffer initial status
        if "optimizer_type" in group and group["optimizer_type"] == "muon":
            if "update_buffer" in group:
                initial_is_shared[(group_idx, "buffer")] = group[
                    "update_buffer"
                ].is_shared()

    # Call share_memory_()
    opt.share_memory_()

    # Verify all state tensors are now in shared memory
    for group_idx, group in enumerate(opt.param_groups):
        for p_idx, p in enumerate(group["params"]):
            if p.requires_grad:
                state = opt.state[p]
                for key, value in state.items():
                    if torch.is_tensor(value):
                        assert value.is_shared(), f"State tensor {key} for param ({group_idx}, {p_idx}) not moved to shared memory"

        # Check update buffer for Muon groups
        if "optimizer_type" in group and group["optimizer_type"] == "muon":
            if "update_buffer" in group:
                # Verify buffer is in shared memory
                assert (
                    group["update_buffer"].is_shared()
                ), f"Update buffer for group {group_idx} not moved to shared memory"

                # Verify views still correctly point to the buffer
                for i, view in enumerate(group["update_buffer_views"]):
                    assert torch.allclose(
                        view, group["update_buffer"][i]
                    ), f"Update buffer view {i} not correctly linked after share_memory in group {group_idx}"


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
        assert not torch.allclose(
            p, initial_params[name]
        ), f"Parameter {name} with shape {p.shape} and dim {p.ndim} did not change during optimization"
