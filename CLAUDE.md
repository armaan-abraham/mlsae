# MLSAE Project Guide

## Development Commands
- Install dependencies: `rye sync`
- Run tests: `pytest`
- Run single test: `pytest tests/test_file.py::test_function -v`
- Lint: `rye lint`
- Format: `rye fmt`

## Code Style Guidelines
- **Imports**: Standard library → third-party → project imports
- **Types**: Use type annotations for all function parameters and returns
- **Naming**: Classes: PascalCase, functions/variables: snake_case, constants: UPPER_SNAKE_CASE
- **Documentation**: Docstrings for classes and functions with parameter descriptions
- **Error handling**: Assertions for validation with explicit messages, ValueError with descriptions

## Project Overview
- Deep Sparse Autoencoder (DeepSAE) implementation with distributed training support
- Main components: models in `model/`, training in `train.py`, evaluation in `scripts/eval.py`
- Configuration in `config.py`, optimizer utilities in `optimizer/`
- Uses PyTorch and supports AWS S3 for model storage