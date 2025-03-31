import importlib
import inspect
from pathlib import Path

# Import base classes that should always be available
from mlsae.model.model import DeepSAE, TopKActivation
from mlsae.optimizer import SparseAdam

# Import experimental models
from mlsae.model.experiment_models import *

# Collect all experimental models (classes that start with ExperimentSAE)
models = []
for name, obj in list(locals().items()):
    if (
        name.startswith("ExperimentSAE")
        and inspect.isclass(obj)
        and issubclass(obj, DeepSAE)
        and obj is not DeepSAE
        and obj is not ExperimentSAEBase
    ):
        models.append(obj)

# Log the discovered experimental models
import logging

logging.info(f"Discovered {len(models)} experimental SAE models:")
for model_class in models:
    logging.info(f"  - {model_class.__name__}")
