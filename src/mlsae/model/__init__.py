import importlib
import pkgutil
import re
from pathlib import Path

# Import base classes that should always be available
from mlsae.model.model import DeepSAE, SparseAdam, TopKActivation

# Dynamically import all model classes
models = []
package_dir = Path(__file__).parent
for module_info in pkgutil.iter_modules([str(package_dir)]):
    if re.match(r"model_\d+$", module_info.name):
        module = importlib.import_module(f"mlsae.model.{module_info.name}")
        class_name = f"DeepSAE{len(models)}"
        if hasattr(module, class_name):
            models.append(getattr(module, class_name))
