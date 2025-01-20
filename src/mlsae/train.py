import concurrent.futures
from dataclasses import dataclass, field

import torch
import tqdm
import wandb

from mlsae.model import DeepSAE, ZERO_ACT_THRESHOLD
from mlsae.data import Buffer, data_cfg


@dataclass
class TrainConfig:
    architectures: list = field(
        default_factory=lambda: [
            {
                "name": "0",
                "encoder_dim_mults": [],
                "sparse_dim_mult": 2,
                "decoder_dim_mults": [],
                "l1_lambda": 1,
                "weight_decay": 4e-4,
                "lr": 4e-3,
                "leaky_relu_slope": 0.05,
            },
            {
                "name": "1",
                "encoder_dim_mults": [1],
                "sparse_dim_mult": 2,
                "decoder_dim_mults": [],
                "l1_lambda": 1,
                "weight_decay": 5e-4,
                "lr": 4e-3,
                "leaky_relu_slope": 0.1,
            },
            {
                "name": "2",
                "encoder_dim_mults": [1],
                "sparse_dim_mult": 2,
                "decoder_dim_mults": [1],
                "l1_lambda": 1,
                "weight_decay": 1e-4,
                "lr": 4e-3,
                "leaky_relu_slope": 0.1,
            },
            {
                "name": "3",
                "encoder_dim_mults": [1],
                "sparse_dim_mult": 2,
                "decoder_dim_mults": [],
                "l1_lambda": 1,
                "weight_decay": 1e-4,
                "lr": 4e-3,
                "leaky_relu_slope": 0.1,
            },
        ]
    )

    num_tokens: int = int(4e8)
    beta1: float = 0.9
    beta2: float = 0.99
    wandb_project: str = "mlsae"
    wandb_entity: str = "armaanabraham-independent"
    n_epochs: int = 8

    resample_dead_every_n_batches: int = int(1e9)
    measure_freq_over_n_batches: int = 6

    log_every_n_batches: int = 10

    save_to_s3: bool = True


train_cfg = TrainConfig()


def model_step(entry, acts, is_train=True):
    """
    Step function to train or evaluate.
    Now uses L1 penalty for sparsity.
    """
    device = entry["device"]
    acts_local = acts.to(device, non_blocking=True) * 10
    autoenc = entry["model"]
    arch_name = entry["name"]

    if is_train:
        optimizer = entry["optimizer"]
        loss, mse_loss, l1_loss, nonzero_acts, feature_acts, reconstructed = autoenc(acts_local)
        loss.backward()
        autoenc.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        optimizer.zero_grad()
    else:
        with torch.no_grad():
            loss, mse_loss, l1_loss, nonzero_acts, feature_acts, reconstructed = autoenc(acts_local)


    return {
        "loss": loss.item(),
        "mse_loss": mse_loss.item(),
        "l1_loss": l1_loss.item(),
        "nonzero_acts": nonzero_acts.item(),
        "arch_name": arch_name,
        "feature_acts": feature_acts.detach(),
    }


def measure_and_resample_dead_features(
    autoencoders, buffer, measure_freq_over_n_batches, executor
):
    """
    Samples from the buffer for 'measure_freq_over_n_batches' batches, in parallel,
    and tracks the total number of activations of each feature. Any feature that never
    activates is considered 'dead' and is resampled via the model's resample_sparse_features method.
    """
    # We'll hold a separate activation count vector for each autoencoder
    activation_counts_list = []
    for entry in autoencoders:
        autoenc = entry["model"]
        activation_counts_list.append(
            torch.zeros(autoenc.sparse_dim, device=autoenc.device, dtype=torch.long)
        )

    # Collect features in parallel over multiple batches
    for _ in range(measure_freq_over_n_batches):
        acts = buffer.next()
        futures = []

        # Submit tasks in eval mode (is_train=False)
        for i, entry in enumerate(autoencoders):
            futures.append(executor.submit(model_step, entry, acts, False))

        # Aggregate activation counts
        for i, f in enumerate(futures):
            result = f.result()
            active_mask = result["feature_acts"] > ZERO_ACT_THRESHOLD
            activation_counts_list[i] += active_mask.sum(dim=0).long()

    # Resample any feature that never activated
    for i, entry in enumerate(autoencoders):
        autoenc = entry["model"]
        activation_counts = activation_counts_list[i]
        dead_features = (activation_counts == 0).nonzero().flatten()
        if len(dead_features) > 0:
            print(f"Resampling {len(dead_features)} dead features for {entry['name']}")
            autoenc.resample_sparse_features(dead_features)


def main():
    print("Starting training...")
    for k, v in data_cfg.__dict__.items():
        print(f"{k}: {v}")
    wandb.init(project=train_cfg.wandb_project, entity=train_cfg.wandb_entity)
    wandb.run.name = "multi_sae_single_buffer_topk"

    wandb.config.update(train_cfg)
    wandb.config.update(data_cfg)
    print(wandb.config)

    print("Building buffer...")
    buffer = Buffer()

    print("Building SAEs...")
    n_gpus = torch.cuda.device_count()
    autoencoders = []
    idx = 0

    for arch_dict in train_cfg.architectures:
        device_id = idx % n_gpus
        device_str = f"cuda:{device_id}"
        idx += 1

        autoenc = DeepSAE(
            encoder_dim_mults=arch_dict["encoder_dim_mults"],
            sparse_dim_mult=arch_dict["sparse_dim_mult"],
            decoder_dim_mults=arch_dict["decoder_dim_mults"],
            leaky_relu_slope=arch_dict["leaky_relu_slope"],
            act_size=data_cfg.act_size,
            enc_dtype=data_cfg.enc_dtype,
            device=device_str,
            l1_lambda=arch_dict["l1_lambda"],
            name=arch_dict["name"],
        )
        optimizer = torch.optim.SGD(
            autoenc.get_param_groups(weight_decay=arch_dict["weight_decay"]),
            lr=arch_dict["lr"],
        )

        # Initialize a local activation count tensor (one entry per feature in sparse_dim)
        local_activation_counts = torch.zeros(
            autoenc.sparse_dim, device=device_str, dtype=torch.long
        )

        autoencoders.append(
            {
                "model": autoenc,
                "optimizer": optimizer,
                "device": device_str,
                "name": arch_dict["name"],
                "local_activation_counts": local_activation_counts,
            }
        )

    total_steps = train_cfg.num_tokens // data_cfg.buffer_batch_size_tokens
    print("Training all SAEs...")

    # We'll use a ThreadPoolExecutor for both training and dead-feature measurement
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=torch.cuda.device_count())

    try:
        for epoch in range(train_cfg.n_epochs):
            for step_idx in tqdm.trange(total_steps, desc="Training SAEs"):
                acts = buffer.next()

                # Training step (in parallel for each autoencoder)
                futures = []
                for entry in autoencoders:
                    futures.append(executor.submit(model_step, entry, acts, True))

                for entry, f in zip(autoencoders, futures):
                    result = f.result()

                    # Accumulate nonzero activations for each feature
                    active_mask = result["feature_acts"] > ZERO_ACT_THRESHOLD
                    entry["local_activation_counts"] += active_mask.sum(dim=0).long()

                    # Log periodically
                    if (step_idx + 1) % train_cfg.log_every_n_batches == 0:
                        # Count how many features never activated in this log interval
                        dead_features_count = (entry["local_activation_counts"] == 0).sum().item()
                        wandb.log({
                            f"{entry['name']}_dead_features": dead_features_count,
                            f"{entry['name']}_loss": result["loss"],
                            f"{entry['name']}_loss_mse": result["mse_loss"],
                            f"{entry['name']}_loss_l1": result["l1_loss"],
                            f"{entry['name']}_nonzero_acts": result["nonzero_acts"],
                        })

                        # Reset counts for next interval
                        entry["local_activation_counts"].zero_()

                    # Also log any other metrics periodically if desired
                    # (Already logging loss, so you might combine or place above/below)

                # Periodic dead-feature resampling
                if (step_idx + 1) % train_cfg.resample_dead_every_n_batches == 0:
                    measure_and_resample_dead_features(
                        autoencoders,
                        buffer,
                        train_cfg.measure_freq_over_n_batches,
                        executor,
                    )
            buffer.chunk_index = 0

    finally:
        print("Saving all SAEs...")
        for entry in autoencoders:
            arch_name = entry["name"]
            entry["model"].save(arch_name, save_to_s3=train_cfg.save_to_s3)
        wandb.finish()


if __name__ == "__main__":
    main()
