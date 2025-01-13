from dataclasses import dataclass, field
import torch
import tqdm
import wandb

from mlsae.model import MultiLayerSAE
from mlsae.utils import data_cfg, Buffer


@dataclass
class TrainConfig:
    """
    Configuration for training the sparse autoencoders.
    """

    # Specify SAE layer dims as multiples of the input dimension
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
    l1_values: list = field(default_factory=lambda: [1e-4, 3e-4])
    lr: float = 1e-4
    num_tokens: int = int(2e9)
    beta1: float = 0.9
    beta2: float = 0.99
    wandb_project: str = "mlsae"
    wandb_entity: str = "armaanabraham-independent"


train_cfg = TrainConfig()


def main():
    """
    Main training entrypoint:
      - Builds exactly ONE Buffer
      - Builds an SAE for each (architecture, l1) combo
      - Runs training steps in a single loop
    """
    print("Starting training...")
    print(f"Initializing W&B project {train_cfg.wandb_project}...")
    wandb.init(project=train_cfg.wandb_project, entity=train_cfg.wandb_entity)
    wandb.run.name = "multi_sae_single_buffer"

    print("Building buffer...")
    # 1) Build one Buffer
    buffer = Buffer()

    print("Building SAEs...")
    # 2) Create all SAE variants + their optimizers
    autoencoders = []
    for arch_dict in train_cfg.architectures:
        for l1_coeff in train_cfg.l1_values:
            arch_name = arch_dict["name"]
            # Build the model
            autoenc = MultiLayerSAE(
                encoder_dim_mults=arch_dict["encoder_dim_mults"],
                sparse_dim_mult=arch_dict["sparse_dim_mult"],
                decoder_dim_mults=arch_dict["decoder_dim_mults"],
                act_size=data_cfg.act_size,
                l1_coeff=l1_coeff,
                enc_dtype=data_cfg.enc_dtype,
                device=data_cfg.device,
            )
            # Create its optimizer
            optimizer = torch.optim.Adam(
                autoenc.parameters(),
                lr=train_cfg.lr,
                betas=(train_cfg.beta1, train_cfg.beta2),
            )
            autoencoders.append(
                {
                    "model": autoenc,
                    "optimizer": optimizer,
                    "name": arch_name,
                    "l1_coeff": l1_coeff,
                }
            )

    # 3) We'll do the same total number of training steps for each SAE.
    total_steps = train_cfg.num_tokens // data_cfg.batch_size

    print("Training all SAEs...")
    # 4) Single loop that fetches from the buffer and trains all SAEs
    for step_idx in tqdm.trange(total_steps, desc="Training SAEs"):
        # Get a batch of activations (this may trigger buffer.refresh() behind the scenes)
        acts = buffer.next()

        # Now train each SAE for one step
        for entry in autoencoders:
            autoenc = entry["model"]
            optimizer = entry["optimizer"]
            arch_name = entry["name"]
            l1_coeff = entry["l1_coeff"]

            loss, feature_acts, l2_loss, l1_loss = autoenc(acts)
            loss.backward()

            # Unit-norm the final decoder weights
            autoenc.make_decoder_weights_and_grad_unit_norm()

            optimizer.step()
            optimizer.zero_grad()

            # Optionally log metrics for each model
            # (use a distinctive key so each variant is tracked separately in wandb)
            if (step_idx + 1) % 100 == 0:
                metrics = {
                    f"{arch_name}_{l1_coeff}_loss": loss.item(),
                    f"{arch_name}_{l1_coeff}_l2_loss": l2_loss.item(),
                    f"{arch_name}_{l1_coeff}_l1_loss": l1_loss.item(),
                }
                wandb.log(metrics)

    print("Saving all SAEs...")
    # 5) Done training, save each model
    for entry in autoencoders:
        autoenc = entry["model"]
        arch_name = entry["name"]
        autoenc.save(arch_name)

    print("Finishing W&B project...")
    wandb.finish()


if __name__ == "__main__":
    main()
