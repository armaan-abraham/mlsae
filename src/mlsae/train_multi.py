from dataclasses import dataclass, field
import torch
import tqdm
import wandb

from mlsae.mlsae import MultiLayerSAE
from mlsae.utils import data_cfg, Buffer

@dataclass
class TrainConfig:
    """
    Configuration class for training the sparse autoencoder.
    This includes hyperparams relevant to training only.
    """
    # Specify SAE layer dims as multiples of the input dimension
    architectures: list = field(default_factory=lambda: [
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
    ])
    l1_values: list = field(default_factory=lambda: [1e-4, 3e-4])
    lr: float = 1e-4
    num_tokens: int = int(2e9)
    beta1: float = 0.9
    beta2: float = 0.99
    wandb_project: str = "mlsae"
    wandb_entity: str = "armaanabraham-independent"

train_cfg = TrainConfig()

def train_one_autoencoder(
    architecture_name,
    encoder_dim_mults,
    sparse_dim_mult,
    decoder_dim_mults,
    l1_coeff,
):
    """
    Trains one SAE with the given architecture and L1 coefficient.
    Returns the trained model.
    """
    # Construct the autoencoder with the data config's parameters
    autoenc = MultiLayerSAE(
        encoder_dims=encoder_dim_mults,
        sparse_dim=sparse_dim_mult,
        decoder_dims=decoder_dim_mults,
        act_size=data_cfg.act_size,
        l1_coeff=l1_coeff,
        enc_dtype=data_cfg.enc_dtype,
        device=data_cfg.device,
    )

    buffer = Buffer()
    optimizer = torch.optim.Adam(
        autoenc.parameters(),
        lr=train_cfg.lr,
        betas=(train_cfg.beta1, train_cfg.beta2)
    )

    total_batches = train_cfg.num_tokens // data_cfg.batch_size

    for i in tqdm.trange(total_batches, desc=f"Training {architecture_name}, L1={l1_coeff}"):
        acts = buffer.next()
        loss, feature_acts, l2_loss, l1_loss = autoenc(acts)
        loss.backward()

        autoenc.make_decoder_weights_and_grad_unit_norm()

        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 100 == 0:
            metrics = {
                "loss": loss.item(),
                "l2_loss": l2_loss.item(),
                "l1_loss": l1_loss.item(),
            }
            wandb.log(metrics)

    return autoenc

def main():
    # Initialize wandb logging
    wandb.init(project=train_cfg.wandb_project, entity=train_cfg.wandb_entity)

    for arch_dict in train_cfg.architectures:
        for l1_coeff in train_cfg.l1_values:
            arch_name = arch_dict["name"]
            run_name = f"{arch_name}_l1={l1_coeff}"
            wandb.run.name = run_name

            autoenc = train_one_autoencoder(
                architecture_name=arch_name,
                encoder_dim_mults=arch_dict["encoder_dim_mults"],
                sparse_dim_mult=arch_dict["sparse_dim_mult"],
                decoder_dim_mults=arch_dict["decoder_dim_mults"],
                l1_coeff=l1_coeff,
            )
            # Save each trained model
            autoenc.save(arch_name)

    wandb.finish()

if __name__ == "__main__":
    main()
