import argparse
import csv
import json

import torch
import tqdm
from utils import Buffer, data_cfg
import pandas as pd

from mlsae.model import DeepSAE

# Load gpt2 model

# Load dataset and use eval_data_seed to shuffle
# Decide on a total number of eval samples

# Run gpt2 on all samples, save activations

# Load SAEs, run on all activations, save results

# Plot results for each SAE

def evaluate_autoencoder(model_paths, output_csv):
    """
    model_paths is a list of dicts with { "architecture_name": str, "version": int }
    Evaluates each model on average L0 and average MSE.
    Saves results to output_csv.
    """
    buffer = Buffer(eval=True)

    results = []
    for mp in model_paths:
        arch_name = mp["architecture_name"]
        version = mp["version"]
        autoenc = DeepSAE.load(arch_name, version)
        autoenc.eval()

        total_mse = 0.0
        total_count = 0

        eval_batches = data_cfg.eval_tokens // data_cfg.buffer_batch_size_tokens
        with torch.no_grad():
            for _ in tqdm.trange(
                eval_batches, desc=f"Evaluating {arch_name}, v{version}"
            ):
                acts = buffer.next()
                loss, feature_acts = autoenc(acts)

                # MSE
                total_mse += loss.item()
                total_count += acts.shape[0]

        avg_mse = total_mse / total_count
        results.append(
            {
                "architecture_name": arch_name,
                "version": version,
                "avg_mse": avg_mse,
            }
        )

    # Convert results to pandas DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved evaluation results to {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_paths",
        type=str,
        required=True,
        help=(
            'JSON string with a list of {"architecture_name": str, "version": int} to evaluate. '
            'Example: \'[{"architecture_name":"arch1","version":0},{"architecture_name":"arch2","version":2}]\''
        ),
    )
    parser.add_argument("--output_csv", type=str, default="evaluation_results.csv")
    args = parser.parse_args()

    mp_list = json.loads(args.model_paths)
    evaluate_autoencoder(mp_list, cfg, args.output_csv)


if __name__ == "__main__":
    main()
