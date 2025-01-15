# %%
import argparse
import csv
import importlib
import json

import pandas as pd
import torch
import tqdm

import mlsae.model
import mlsae.utils

importlib.reload(mlsae.model)
importlib.reload(mlsae.utils)
from pathlib import Path

from mlsae.model import DeepSAE
from mlsae.utils import Buffer, data_cfg

# Load gpt2 model

# Load dataset and use eval_data_seed to shuffle
# Decide on a total number of eval samples

# Run gpt2 on all samples, save activations

# Load SAEs, run on all activations, save results

# Plot results for each SAE

# Define models to evaluate
MODELS_TO_EVALUATE = [
    {"architecture_name": "0-0", "version": 23},
    {"architecture_name": "0-0", "version": 24},
    {"architecture_name": "0-0", "version": 25},
    {"architecture_name": "1-0.1", "version": 3},
    {"architecture_name": "1-0.1", "version": 4},
    {"architecture_name": "1-0.1", "version": 5},
    {"architecture_name": "1-0.2", "version": 18},
    {"architecture_name": "1-0.2", "version": 19},
    {"architecture_name": "1-0.2", "version": 20},
    {"architecture_name": "1-1", "version": 3},
    {"architecture_name": "1-1", "version": 4},
    {"architecture_name": "1-1", "version": 5},
]
output_dir = Path(__file__.parent / "results")
output_dir.mkdir(parents=True, exist_ok=True)
output_csv = output_dir / "evaluation_results.csv"


# %%
buffer = Buffer(eval=True, use_multiprocessing=False)

# %%
results = []
for mp in MODELS_TO_EVALUATE:
    arch_name = mp["architecture_name"]
    version = mp["version"]
    autoenc = DeepSAE.load(arch_name, version)
    buffer.pointer = 0
    autoenc.to("cuda")
    autoenc.eval()

    total_mse = 0.0
    total_count = 0

    eval_batches = data_cfg.eval_tokens // data_cfg.buffer_batch_size_tokens
    with torch.no_grad():
        for _ in tqdm.trange(eval_batches, desc=f"Evaluating {arch_name}, v{version}"):
            acts = buffer.next()
            acts = acts.to(autoenc.device, autoenc.dtype)
            loss, feature_acts = autoenc(acts)

            # MSE
            total_mse += loss.item()
            total_count += acts.shape[0]

    avg_mse = total_mse / total_count
    print(f"Evaluated {total_count} tokens")
    results.append(
        {
            "architecture_name": arch_name,
            "k": autoenc.k,
            "sparse_dim": autoenc.sparse_dim,
            "encoder_dim": autoenc.encoder_dims[0] if autoenc.encoder_dims else None,
            "decoder_dim": autoenc.decoder_dims[0] if autoenc.decoder_dims else None,
            "version": version,
            "avg_mse": avg_mse,
        }
    )

# Convert results to pandas DataFrame and save to CSV
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"Saved evaluation results to {output_csv}")

# %%
