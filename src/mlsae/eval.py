# %%
import argparse
import csv
import importlib
import json

import pandas as pd
import torch
import tqdm

import mlsae.model
import mlsae.data

importlib.reload(mlsae.model)
importlib.reload(mlsae.data)
from pathlib import Path

from mlsae.model import DeepSAE
from mlsae.data import Buffer, data_cfg

# Load gpt2 model

# Load dataset and use eval_data_seed to shuffle
# Decide on a total number of eval samples

# Run gpt2 on all samples, save activations

# Load SAEs, run on all activations, save results

# Plot results for each SAE

# Define models to evaluate
output_dir = Path(__file__).parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)
output_csv = output_dir / "evaluation_results.csv"


if __name__ == "__main__":
    buffer = Buffer(eval=True, use_multiprocessing=False)

    results = []
    arch_name = "9"
    autoenc = DeepSAE.load(arch_name, load_from_s3=True, model_id="safely-bright-kit").eval()
    autoenc.start_act_stat_tracking()
    autoenc.to("cuda")
    autoenc.eval()

    total_mse = 0.0
    total_count = 0
    nonzero_acts_sum = 0

    with torch.no_grad():
        for _ in tqdm.trange(data_cfg.eval_batches, desc=f"Evaluating {arch_name}"):
            acts = buffer.next()
            acts = acts.to(autoenc.device, autoenc.dtype)
            acts *= 10
            _, mse_loss, _, nonzero_acts, _, _ = autoenc(acts)

            # MSE
            total_mse += mse_loss.item()
            total_count += 1
            nonzero_acts_sum += nonzero_acts.item()
            print(f"Act stats: {autoenc.get_activation_stats()}")

    avg_mse = total_mse / total_count
    avg_nonzero_acts = nonzero_acts_sum / total_count
    print(f"Evaluated {total_count * data_cfg.buffer_batch_size_tokens} tokens")
    results.append(
        {
            "architecture_name": arch_name,
            "l1_lambda": autoenc.l1_lambda,
            "sparse_dim": autoenc.sparse_dim,
            "encoder_dim": autoenc.encoder_dims[0] if autoenc.encoder_dims else None,
            "decoder_dim": autoenc.decoder_dims[0] if autoenc.decoder_dims else None,
            "avg_mse": avg_mse,
            "avg_nonzero_acts": avg_nonzero_acts,
        }
    )
    buffer.chunk_index = 0
    buffer.pointer = 0

    # Convert results to pandas DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved evaluation results to {output_csv}")

