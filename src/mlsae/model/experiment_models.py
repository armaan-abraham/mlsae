from mlsae.model.model import ExperimentSAEBase, TopKActivation
import torch
import math
import torch.nn as nn

from mlsae.model.rl_sae import RLSAE


def create_model_variants(base_class, param_grid):
    """
    Dynamically creates model classes with varying hyperparameters.
    
    Args:
        base_class: The base class to inherit from
        param_grid: Dictionary mapping param names to lists of values
        prefix: Prefix for the generated class names
    
    Returns:
        List of created model classes
    """
    created_classes = []
    
    # Generate all combinations of hyperparameters
    import itertools
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for i, values in enumerate(itertools.product(*param_values)):
        params = dict(zip(param_names, values))
        
        # Create a descriptive name based on the params
        param_str = "_".join(f"{k}_{v}".replace(".", "p") for k, v in params.items())
        class_name = f"{base_class.__name__}_{param_str}"
        
        # Create a new class with these parameters
        def create_init(fixed_params):
            def __init__(self, act_size: int, device: str = "cpu", **kwargs):
                # Merge the fixed params with any additional kwargs
                all_params = {**fixed_params, **kwargs}
                super(self.__class__, self).__init__(
                    act_size=act_size,
                    device=device,
                    **all_params
                )
            return __init__
        
        # Create the new class
        new_class = type(class_name, (base_class,), {
            "__init__": create_init(params),
            "__doc__": f"Auto-generated variant of {base_class.__name__} with {params}"
        })
        
        # Add the class to the module's global namespace
        globals()[class_name] = new_class
        created_classes.append(new_class)
    
    # Delete the base class from the global namespace
    base_class_name = base_class.__name__
    if base_class_name in globals():
        del globals()[base_class_name]
    
    return created_classes

class ExperimentSAERL(RLSAE):
    def __init__(self, act_size: int, device: str = "cpu", rl_loss_weight=0.2, optimizer_config=None, action_collapse_penalty_lambda=0, loss_stats_momentum=0.9):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[],
            sparse_dim_mult=8,
            decoder_dim_mults=[],
            device=device,
            num_samples=5,
            L0_penalty=4e-4,
            rl_loss_weight=rl_loss_weight,
            optimizer_type="sparse_adam",
            optimizer_config=optimizer_config,
            optimize_steps=1,
            loss_stats_momentum=loss_stats_momentum,

            base_L0=13,
            action_collapse_penalty_lambda=action_collapse_penalty_lambda,
        )

rl_experiment_variants = create_model_variants(
    ExperimentSAERL,
    {
        "loss_stats_momentum": [
            0.9,
            0.95,
            0.99,
        ],
        "rl_loss_weight": [
            2,
            5,
            10,
        ],
        "optimizer_config": [
            {
                "lr": 2e-3,
            },
            {
                "lr": 1e-3,
            },
        ],
    }
)

# class ExperimentSAETopK(ExperimentSAEBase):
#     def __init__(self, act_size: int, device: str = "cpu", optimizer_config=None):
#         super().__init__(
#             act_size=act_size,
#             encoder_dim_mults=[],
#             sparse_dim_mult=8,
#             decoder_dim_mults=[],
#             device=device,
#             topk_init=14,
#             topk_final=14,
#             topk_decay_iter=1000,
#             optimizer_type="sparse_adam",
#             optimizer_config=optimizer_config,
#             optimize_steps=1,
#             weight_decay=0,
#             act_squeeze=0,
#         )


# experiment_variants = create_model_variants(
#     ExperimentSAETopK,
#     {
#         "optimizer_config": [
#             {
#                 "lr": 1e-3,
#             },
#         ],
#     }
# )
