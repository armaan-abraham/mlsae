from dataclasses import dataclass, field

import torch
import tqdm

import wandb
from mlsae.model import ZERO_ACT_THRESHOLD, DeepSAE
from mlsae.data import Buffer, data_cfg
import petname


@dataclass
class TrainConfig:
    architecture: dict = field(
        default_factory=lambda: {
            "name": "1-0",
            "encoder_dim_mults": [1],
            "sparse_dim_mult": 16,
            "decoder_dim_mults": [],
        }
    )
    l1_coeff: float = 12

    lr: float = 1e-4
    num_tokens: int = int(1e9)
    beta1: float = 0.9
    beta2: float = 0.99
    wandb_project: str = "mlsae"
    wandb_entity: str = "armaanabraham-independent"
    weight_decay: float = 1e-4

    resample_dead_every_n_batches: int = 3000
    measure_freq_over_n_batches: int = 12

    log_every_n_batches: int = 10


train_cfg = TrainConfig()


def measure_and_resample_dead_features(autoenc, buffer, measure_freq_over_n_batches):
    """
    Samples from the buffer for 'measure_freq_over_n_batches' batches (single-threaded),
    and tracks the total number of activations of each feature. Any feature that never
    activates is considered 'dead' and is resampled via the model's resample_sparse_features method.
    """
    activation_counts = torch.zeros(
        autoenc.sparse_dim, device=autoenc.device, dtype=torch.long
    )

    for _ in range(measure_freq_over_n_batches):
        acts = buffer.next().to(autoenc.device)
        with torch.no_grad():
            _, _, _, _, feature_acts = autoenc(acts)
        active_mask = feature_acts > ZERO_ACT_THRESHOLD
        activation_counts += active_mask.sum(dim=0).long()

    # Resample any feature that never activated
    dead_features = (activation_counts == 0).nonzero().flatten()
    if len(dead_features) > 0:
        print(f"Resampling {len(dead_features)} dead features.")
        autoenc.resample_sparse_features(dead_features)


def main():
    print("Starting training...")
    for k, v in data_cfg.__dict__.items():
        print(f"{k}: {v}")
    wandb.init(project=train_cfg.wandb_project, entity=train_cfg.wandb_entity)
    base_name = f"{train_cfg.architecture['name']}_l1={train_cfg.l1_coeff}"
    wandb.run.name = f"{base_name}-{petname.generate()}"

    wandb.config.update(train_cfg)
    wandb.config.update(data_cfg)
    print(wandb.config)

    print("Building buffer...")
    buffer = Buffer()

    print("Building SAE...")
    autoenc = DeepSAE(
        encoder_dim_mults=train_cfg.architecture["encoder_dim_mults"],
        sparse_dim_mult=train_cfg.architecture["sparse_dim_mult"],
        decoder_dim_mults=train_cfg.architecture["decoder_dim_mults"],
        act_size=data_cfg.act_size,
        enc_dtype=data_cfg.enc_dtype,
        device=data_cfg.device,
        l1_coeff=train_cfg.l1_coeff,
    )

    optimizer = torch.optim.Adam(
        autoenc.get_param_groups(weight_decay=train_cfg.weight_decay),
        lr=train_cfg.lr,
        betas=(train_cfg.beta1, train_cfg.beta2),
    )

    total_steps = train_cfg.num_tokens // data_cfg.buffer_batch_size_tokens
    print("Training SAE...")

    try:
        for step_idx in tqdm.trange(total_steps, desc="Training"):
            acts = buffer.next().to(data_cfg.device)
            loss, mse_loss, l1_loss, nonzero_acts, feature_acts = autoenc(acts)

            loss.backward()
            autoenc.make_decoder_weights_and_grad_unit_norm()
            optimizer.step()
            optimizer.zero_grad()

            # Log periodically
            if (step_idx + 1) % train_cfg.log_every_n_batches == 0:
                metrics = {
                    f"loss": loss.item(),
                    f"mse_loss": mse_loss.item(),
                    f"l1_loss": l1_loss.item(),
                    f"nonzero_acts": nonzero_acts.item(),
                }
                wandb.log(metrics)

            # Every resample_dead_every_n_batches, measure dead features (single-threaded)
            if (step_idx + 1) % train_cfg.resample_dead_every_n_batches == 0:
                measure_and_resample_dead_features(
                    autoenc, buffer, train_cfg.measure_freq_over_n_batches
                )

    finally:
        print("Saving the SAE...")
        autoenc.save(train_cfg.architecture["name"])
        wandb.finish()


if __name__ == "__main__":
    main()
