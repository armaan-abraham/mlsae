import torch
import tqdm
import wandb
import concurrent.futures
from mlsae.model import MultiLayerSAE
from mlsae.utils import data_cfg, Buffer
from dataclasses import dataclass, field

L0_THRESHOLD = 1e-6

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
                "name": "1-0",
                "encoder_dim_mults": [2],
                "sparse_dim_mult": 32,
                "decoder_dim_mults": [],
            },
            {
                "name": "1-1",
                "encoder_dim_mults": [2],
                "sparse_dim_mult": 32,
                "decoder_dim_mults": [2],
            },
        ]
    )
    l1_values: list = field(default_factory=lambda: [1e-6, 3e-6, 1e-5])
    lr: float = 1e-4
    num_tokens: int = int(2e8)
    beta1: float = 0.9
    beta2: float = 0.99
    wandb_project: str = "mlsae"
    wandb_entity: str = "armaanabraham-independent"

train_cfg = TrainConfig()

def train_step(entry, acts):
    # Move data to correct device
    device = entry["device"]
    acts_local = acts.to(device, non_blocking=True)
    autoenc = entry["model"]
    optimizer = entry["optimizer"]
    arch_name = entry["name"]
    l1_coeff = entry["l1_coeff"]

    loss, feature_acts, l2_loss, l1_loss = autoenc(acts_local)
    l0 = (feature_acts > L0_THRESHOLD).sum().item() / feature_acts.numel()
    num_nonzero = l0 * autoenc.sparse_dim
    loss.backward()

    autoenc.make_decoder_weights_and_grad_unit_norm()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item(), l2_loss.item(), l1_loss.item(), l0, num_nonzero, arch_name, l1_coeff

def main():
    print("Starting training...")
    wandb.init(project=train_cfg.wandb_project, entity=train_cfg.wandb_entity)
    wandb.run.name = "multi_sae_single_buffer"

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
        for l1_coeff in train_cfg.l1_values:
            device_id = idx % n_gpus
            device_str = f"cuda:{device_id}"
            idx += 1

            autoenc = MultiLayerSAE(
                encoder_dim_mults=arch_dict["encoder_dim_mults"],
                sparse_dim_mult=arch_dict["sparse_dim_mult"],
                decoder_dim_mults=arch_dict["decoder_dim_mults"],
                act_size=data_cfg.act_size,
                l1_coeff=l1_coeff,
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
                    "l1_coeff": l1_coeff,
                }
            )

    total_steps = train_cfg.num_tokens // data_cfg.batch_size
    print("Training all SAEs...")

    # Create a thread pool as large as number of SAEs (or smaller if you prefer)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(autoencoders))

    try:
        for step_idx in tqdm.trange(total_steps, desc="Training SAEs"):
            acts = buffer.next()

            # Kick off each model's training step in its own thread
            futures = []
            for entry in autoencoders:
                futures.append(executor.submit(train_step, entry, acts))

            # Collect results
            for f in futures:
                loss_val, l2_val, l1_val, l0, num_nonzero, arch_name, l1_coeff = f.result()

                if (step_idx + 1) % 50 == 0:
                    metrics = {
                        f"{arch_name}_{l1_coeff}_loss": loss_val,
                        f"{arch_name}_{l1_coeff}_l2_loss": l2_val,
                        f"{arch_name}_{l1_coeff}_l1_loss": l1_val,
                        f"{arch_name}_{l1_coeff}_l0": l0,
                        f"{arch_name}_{l1_coeff}_num_nonzero": num_nonzero,
                    },
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
