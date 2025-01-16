import concurrent.futures
from dataclasses import dataclass, field

import torch
import tqdm

import wandb
from mlsae.model import DeepSAE
from mlsae.utils import Buffer, data_cfg


@dataclass
class TrainConfig:
    architectures: list = field(
        default_factory=lambda: [
            {
                "name": "0-0",
                "encoder_dim_mults": [],
                "sparse_dim_mult": 16,
                "decoder_dim_mults": [],
            },
            {
                "name": "1-0",
                "encoder_dim_mults": [1],
                "sparse_dim_mult": 16,
                "decoder_dim_mults": [],
            },
            {
                "name": "1-1",
                "encoder_dim_mults": [1],
                "sparse_dim_mult": 16,
                "decoder_dim_mults": [1],
            },
        ]
    )
    l1_values: list = field(default_factory=lambda: [1, 24])

    lr: float = 1e-4
    num_tokens: int = int(1e9)
    beta1: float = 0.9
    beta2: float = 0.99
    wandb_project: str = "mlsae"
    wandb_entity: str = "armaanabraham-independent"
    weight_decay: float = 5e-3


train_cfg = TrainConfig()


def train_step(entry, acts):
    device = entry["device"]
    acts_local = acts.to(device, non_blocking=True)
    autoenc = entry["model"]
    optimizer = entry["optimizer"]
    arch_name = entry["name"]
    l1_val = entry["l1_coeff"]

    loss, mse_loss, l1_loss, nonzero_acts, feature_acts = autoenc(acts_local)

    loss.backward()
    autoenc.make_decoder_weights_and_grad_unit_norm()
    optimizer.step()
    optimizer.zero_grad()

    return (
        loss.item(),
        mse_loss.item(),
        l1_loss.item(),
        nonzero_acts.item(),
        arch_name,
        l1_val,
    )


def main():
    print("Starting training...")
    for k, v in data_cfg.__dict__.items():
        print(f"{k}: {v}")
    wandb.init(project=train_cfg.wandb_project, entity=train_cfg.wandb_entity)
    wandb.run.name = "multi_sae_single_buffer_l1"

    wandb.config.update(train_cfg)
    wandb.config.update(data_cfg)
    print(wandb.config)

    print("Building buffer...")
    buffer = Buffer()

    print("Building SAEs...")
    n_gpus = torch.cuda.device_count()
    autoencoders = []
    idx = 0

    # Loop over different architectures and L1 coefficients
    for arch_dict in train_cfg.architectures:
        for l1_val in train_cfg.l1_values:
            device_id = idx % n_gpus
            device_str = f"cuda:{device_id}"
            idx += 1

            autoenc = DeepSAE(
                encoder_dim_mults=arch_dict["encoder_dim_mults"],
                sparse_dim_mult=arch_dict["sparse_dim_mult"],
                decoder_dim_mults=arch_dict["decoder_dim_mults"],
                act_size=data_cfg.act_size,
                enc_dtype=data_cfg.enc_dtype,
                device=device_str,
                l1_coeff=l1_val,  # L1 sparsity penalty
            )
            optimizer = torch.optim.Adam(
                autoenc.get_param_groups(weight_decay=train_cfg.weight_decay),
                lr=train_cfg.lr,
                betas=(train_cfg.beta1, train_cfg.beta2),
            )
            autoencoders.append(
                {
                    "model": autoenc,
                    "optimizer": optimizer,
                    "device": device_str,
                    "name": arch_dict["name"],
                    "l1_coeff": l1_val,
                }
            )

    total_steps = train_cfg.num_tokens // data_cfg.buffer_batch_size_tokens
    print("Training all SAEs...")

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(autoencoders))

    try:
        for step_idx in tqdm.trange(total_steps, desc="Training SAEs"):
            acts = buffer.next()

            futures = []
            for entry in autoencoders:
                futures.append(executor.submit(train_step, entry, acts))

            for f in futures:
                loss_val, mse_val, l1_val_loss, nonzero_acts, arch_name, l1_val = (
                    f.result()
                )

                # Log periodically
                if (step_idx + 1) % 25 == 0:
                    metrics = {
                        f"{arch_name}_l1={l1_val}_loss": loss_val,
                        f"{arch_name}_l1={l1_val}_mse": mse_val,
                        f"{arch_name}_l1={l1_val}_l1_loss": l1_val_loss,
                        f"{arch_name}_l1={l1_val}_nonzero_acts": nonzero_acts,
                    }
                    wandb.log(metrics)
    finally:
        print("Saving all SAEs...")
        for entry in autoencoders:
            autoenc = entry["model"]
            arch_name = entry["name"]
            autoenc.save(arch_name)
        wandb.finish()


if __name__ == "__main__":
    main()
