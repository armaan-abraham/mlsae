import torch
import tqdm
import wandb
import concurrent.futures
from mlsae.model import MultiLayerSAE
from mlsae.utils import data_cfg, Buffer
from dataclasses import dataclass, field

@dataclass
class TrainConfig:
    architectures: list = field(
        default_factory=lambda: [
            {
                "name": "0-0",
                "encoder_dim_mults": [],
                "sparse_dim_mult": 32,
                "decoder_dim_mults": [],
            },
            {
                "name": "1-0.2",
                "encoder_dim_mults": [2],
                "sparse_dim_mult": 16,
                "decoder_dim_mults": [],
            },
            {
                "name": "1-0.1",
                "encoder_dim_mults": [1],
                "sparse_dim_mult": 16,
                "decoder_dim_mults": [],
            },
            {
                "name": "1-1",
                "encoder_dim_mults": [2],
                "sparse_dim_mult": 16,
                "decoder_dim_mults": [2],
            },
        ]
    )
    # Instead of multiple L1 coefficients, use multiple k values
    k_values: list = field(default_factory=lambda: [32, 128, 512])

    lr: float = 1e-4
    num_tokens: int = int(2e8)
    beta1: float = 0.9
    beta2: float = 0.99
    wandb_project: str = "mlsae"
    wandb_entity: str = "armaanabraham-independent"

train_cfg = TrainConfig()

def train_step(entry, acts):
    device = entry["device"]
    acts_local = acts.to(device, non_blocking=True)
    autoenc = entry["model"]
    optimizer = entry["optimizer"]
    arch_name = entry["name"]
    k_val = entry["k"]

    loss, feature_acts = autoenc(acts_local)

    loss.backward()
    autoenc.make_decoder_weights_and_grad_unit_norm()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item(), arch_name, k_val

def main():
    print("Starting training...")
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
        for k_val in train_cfg.k_values:
            device_id = idx % n_gpus
            device_str = f"cuda:{device_id}"
            idx += 1

            autoenc = MultiLayerSAE(
                encoder_dim_mults=arch_dict["encoder_dim_mults"],
                sparse_dim_mult=arch_dict["sparse_dim_mult"],
                decoder_dim_mults=arch_dict["decoder_dim_mults"],
                act_size=data_cfg.act_size,
                top_k=k_val,  # Pass the top-k value
                enc_dtype=data_cfg.enc_dtype,
                device=device_str,
            )
            optimizer = torch.optim.Adam(
                autoenc.parameters(),
                lr=train_cfg.lr,
                betas=(train_cfg.beta1, train_cfg.beta2),
            )
            autoencoders.append(
                {
                    "model": autoenc,
                    "optimizer": optimizer,
                    "device": device_str,
                    "name": arch_dict["name"],
                    "k": k_val,
                }
            )

    total_steps = train_cfg.num_tokens // data_cfg.batch_size
    print("Training all SAEs...")

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(autoencoders))

    try:
        for step_idx in tqdm.trange(total_steps, desc="Training SAEs"):
            acts = buffer.next()

            futures = []
            for entry in autoencoders:
                futures.append(executor.submit(train_step, entry, acts, step_idx))

            for f in futures:
                loss_val, arch_name, k_val = f.result()

                # Log periodically
                if (step_idx + 1) % 50 == 0:
                    metrics = {
                        f"{arch_name}_k={k_val}_loss": loss_val,
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
