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
                    act_size=act_size, device=device, **all_params
                )

            return __init__

        # Create the new class
        new_class = type(
            class_name,
            (base_class,),
            {
                "__init__": create_init(params),
                "__doc__": f"Auto-generated variant of {base_class.__name__} with {params}",
            },
        )

        # Add the class to the module's global namespace
        globals()[class_name] = new_class
        created_classes.append(new_class)

    # Delete the base class from the global namespace
    base_class_name = base_class.__name__
    if base_class_name in globals():
        del globals()[base_class_name]

    return created_classes


# mult-1
class ExperimentSAERL(RLSAE):
    def __init__(
        self,
        act_size: int,
        device: str = "cpu",
        initial_temperature=6,
        temperature_tau=4000,
        rl_loss_weight=0.6,
        lr=2e-3,
    ):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[],
            sparse_dim_mult=32,
            decoder_dim_mults=[],
            device=device,
            num_samples=10,
            L0_penalty=5e-5,
            rl_loss_weight=rl_loss_weight,
            optimizer_type="sparse_adam",
            optimizer_config={"lr": lr},
            optimize_steps=1,
            base_L0=2048,
            initial_temperature=initial_temperature,
            min_temperature=1.0,
            temperature_tau=temperature_tau,
        )


# class ExperimentSAETopK(ExperimentSAEBase):
#     def __init__(self, act_size: int, device: str = "cpu", lr: float = 1e-3, topk: int = 9):
#         super().__init__(
#             act_size=act_size,
#             encoder_dim_mults=[],
#             sparse_dim_mult=8,
#             decoder_dim_mults=[],
#             device=device,
#             topk=topk,
#             optimizer_type="sparse_adam",
#             optimizer_config={"lr": lr},
#             optimize_steps=1,
#             weight_decay=0,
#             act_squeeze=0,
#         )


experiment_variants = create_model_variants(
    ExperimentSAERL,
    {
        "lr": [
            5e-4,
            1e-3,
            2e-3,
        ],
        "rl_loss_weight": [
            0.6,
            2.4,
        ],
        "initial_temperature": [
            24,
            48
        ],
        "temperature_tau": [
            4000,
            8000,
        ],
    }
)