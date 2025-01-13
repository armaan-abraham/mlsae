import torch
import tqdm
from utils import Buffer, Config, model
from multilayer_autoencoder import MultiLayerAutoEncoder
import wandb
import argparse

def train_one_autoencoder(
    architecture_name,
    encoder_dims,
    sparse_dim,
    decoder_dims,
    l1_coeff,
    cfg: Config,
):
    """
    Trains one SAE with the given architecture and L1 coefficient.
    Returns the trained model.
    """
    autoenc = MultiLayerAutoEncoder(
        encoder_dims=encoder_dims,
        sparse_dim=sparse_dim,
        decoder_dims=decoder_dims,
        act_size=cfg.act_size,
        l1_coeff=l1_coeff,
        enc_dtype=cfg.enc_dtype,
        device=cfg.device,
    )

    buffer = Buffer()
    autoenc_optim = torch.optim.Adam(
        autoenc.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2)
    )
    total_batches = cfg.num_tokens // cfg.batch_size

    for i in tqdm.trange(total_batches, desc=f"Training {architecture_name}, L1={l1_coeff}"):
        acts = buffer.next()
        loss, feature_acts, l2_loss, l1_loss = autoenc(acts)
        loss.backward()
        autoenc.make_decoder_weights_and_grad_unit_norm()
        autoenc_optim.step()
        autoenc_optim.zero_grad()

        if (i + 1) % 100 == 0:
            metrics = {
                "loss": loss.item(),
                "l2_loss": l2_loss.item(),
                "l1_loss": l1_loss.item(),
            }
            wandb.log(metrics)

    return autoenc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architectures",
        type=str,
        required=True,
        help=(
            "JSON string with a list of architecture configs. "
            "Each item: {\"name\": str, \"encoder_dims\": [...], \"sparse_dim\": int, \"decoder_dims\": [...]}"
        ),
    )
    parser.add_argument(
        "--l1_values",
        type=str,
        required=True,
        help="JSON string with a list of L1 coefficients to try, e.g. [1e-4, 3e-4]",
    )
    parser.add_argument("--wandb_project", type=str, default="autoencoder")
    parser.add_argument("--wandb_entity", type=str, default="armaanabraham-independent")
    args = parser.parse_args()

    import json
    arch_list = json.loads(args.architectures)
    l1_list = json.loads(args.l1_values)

    cfg = Config()

    wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    for arch_dict in arch_list:
        for l1_coeff in l1_list:
            arch_name = arch_dict["name"]
            # Build a unique run name for wandb
            wandb_run_name = f"{arch_name}_l1={l1_coeff}"
            wandb.run.name = wandb_run_name

            autoenc = train_one_autoencoder(
                architecture_name=arch_name,
                encoder_dims=arch_dict["encoder_dims"],
                sparse_dim=arch_dict["sparse_dim"],
                decoder_dims=arch_dict["decoder_dims"],
                l1_coeff=l1_coeff,
                cfg=cfg,
            )
            # Save model
            autoenc.save(arch_name)

    wandb.finish()


if __name__ == "__main__":
    main()
