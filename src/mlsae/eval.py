import torch
import csv
from utils import Buffer, Config
from mlsae.model import MultiLayerSAE
import tqdm
import argparse
import json


def evaluate_autoencoder(model_paths, cfg: Config, output_csv):
    """
    model_paths is a list of dicts with { "architecture_name": str, "version": int }
    Evaluates each model on average L0 and average MSE.
    Saves results to output_csv.
    """
    buffer = Buffer()
    # Evaluate on a subset of the data
    eval_batches = 100
    batch_size = cfg.batch_size

    results = []
    for mp in model_paths:
        arch_name = mp["architecture_name"]
        version = mp["version"]
        autoenc = MultiLayerSAE.load(arch_name, version)
        autoenc.eval()

        total_mse = 0.0
        total_count = 0
        total_nonzero = 0.0
        total_acts = 0

        with torch.no_grad():
            for _ in tqdm.trange(
                eval_batches, desc=f"Evaluating {arch_name}, v{version}"
            ):
                acts = buffer.next()
                loss, feature_acts, l2_loss, l1_loss = autoenc(acts)
                reconstructed = autoenc.decoder(feature_acts)
                # MSE
                mse_val = (reconstructed - acts).pow(2).sum().item()
                total_mse += mse_val
                total_count += acts.shape[0] * acts.shape[1]  # batch * features

                # L0 is fraction of nonzero in the sparse representation
                nonzero_count = (feature_acts != 0).sum().item()
                total_nonzero += nonzero_count
                total_acts += feature_acts.numel()

        avg_mse = total_mse / total_count
        avg_l0 = total_nonzero / total_acts  # fraction of nonzero
        results.append(
            {
                "architecture_name": arch_name,
                "version": version,
                "avg_mse": avg_mse,
                "avg_l0": avg_l0,
            }
        )

    fieldnames = ["architecture_name", "version", "avg_mse", "avg_l0"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
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
    cfg = Config()
    evaluate_autoencoder(mp_list, cfg, args.output_csv)


if __name__ == "__main__":
    main()
