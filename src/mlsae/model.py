import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlsae.utils import DTYPES

model_dir = Path(__file__).parent / "checkpoints"


class DeepSAE(nn.Module):
    """
    Multi-layer sparse autoencoder with a single sparse representation layer,
    using ReLU + an L1 sparsity penalty on the hidden layer.
    """

    def __init__(
        self,
        encoder_dim_mults: list[int],
        sparse_dim_mult: int,
        decoder_dim_mults: list[int],
        act_size: int,
        enc_dtype: str = "fp32",
        device: str = "cuda:0",
        l1_coeff: float = 0.0,
    ):
        super().__init__()

        self.encoder_dims = [dim * act_size for dim in encoder_dim_mults]
        self.decoder_dims = [dim * act_size for dim in decoder_dim_mults]
        self.sparse_dim = sparse_dim_mult * act_size
        self.act_size = act_size
        self.enc_dtype = enc_dtype
        self.dtype = DTYPES[enc_dtype]
        self.device = device
        self.l1_coeff = l1_coeff

        print(f"Encoder dims: {self.encoder_dims}")
        print(f"Decoder dims: {self.decoder_dims}")
        print(f"Sparse dim: {self.sparse_dim}")
        print(f"L1 coefficient: {self.l1_coeff}")
        print(f"Device: {self.device}")
        print(f"Dtype: {self.dtype}")

        # Build encoder
        layers = []
        in_dim = self.act_size
        for dim in self.encoder_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU())
            in_dim = dim

        # Sparse representation
        layers.append(nn.Linear(in_dim, self.sparse_dim))
        # We'll apply ReLU in forward()

        self.encoder = nn.Sequential(*layers)

        # Build decoder
        dec_layers = []
        out_dim = self.sparse_dim
        for dim in self.decoder_dims:
            dec_layers.append(nn.Linear(out_dim, dim))
            dec_layers.append(nn.ReLU())
            out_dim = dim

        dec_layers.append(nn.Linear(out_dim, self.act_size))
        self.decoder = nn.Sequential(*dec_layers)

        self.init_weights()
        self.to(self.device, self.dtype)

    def init_weights(self):
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        feature_acts = F.relu(encoded)

        # Decode
        reconstructed = self.decoder(feature_acts)

        # Compute MSE reconstruction loss
        mse_loss = (reconstructed.float() - x.float()).pow(2).mean()

        # Add L1 penalty on the sparse hidden activation
        l1_loss = feature_acts.abs().mean()
        loss = mse_loss + self.l1_coeff * l1_loss

        # Calculate number of nonzero activations (>1e-6)
        assert feature_acts.shape[0] == x.shape[0], "Batch size mismatch"
        nonzero_acts = (feature_acts > 1e-6).float().sum(dim=1).mean()

        return loss, mse_loss, l1_loss, nonzero_acts, feature_acts

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        # Keep final decoder layer weights unit norm in each row
        if hasattr(self.decoder[-1], "weight"):
            w = self.decoder[-1].weight
            w_normed = w / w.norm(dim=-1, keepdim=True)
            if w.grad is not None:
                w_dec_grad_proj = (w.grad * w_normed).sum(-1, keepdim=True) * w_normed
                w.grad -= w_dec_grad_proj
            w.data = w_normed

    def get_version(self, save_path: Path):
        version_list = []
        for file in save_path.glob("*.pt"):
            try:
                version_list.append(int(file.stem))
            except ValueError:
                pass
        return max(version_list, default=-1) + 1

    def save(self, architecture_name, version=None):
        save_path = model_dir / architecture_name
        save_path.mkdir(exist_ok=True, parents=True)

        if version is None:
            version = self.get_version(save_path)

        torch.save(self.state_dict(), save_path / f"{version}.pt")

        config_dict = {
            "encoder_dims": self.encoder_dims,
            "decoder_dims": self.decoder_dims,
            "sparse_dim": self.sparse_dim,
            "act_size": self.act_size,
            "enc_dtype": self.enc_dtype,
            "device": self.device,
            "l1_coeff": self.l1_coeff,
        }
        with open(save_path / f"{version}_cfg.json", "w") as f:
            json.dump(config_dict, f)

        print(f"Saved version {version} for architecture {architecture_name}")

    @classmethod
    def load(cls, architecture_name, version):
        load_path = model_dir / architecture_name
        with open(load_path / f"{version}_cfg.json", "r") as f:
            config_dict = json.load(f)
        new_model = cls(
            encoder_dim_mults=[
                dim // config_dict["act_size"] for dim in config_dict["encoder_dims"]
            ],
            sparse_dim_mult=config_dict["sparse_dim"] // config_dict["act_size"],
            decoder_dim_mults=[
                dim // config_dict["act_size"] for dim in config_dict["decoder_dims"]
            ],
            act_size=config_dict["act_size"],
            enc_dtype=config_dict["enc_dtype"],
            device=config_dict["device"],
            l1_coeff=config_dict["l1_coeff"],
        )

        state_dict = torch.load(load_path / f"{version}.pt", map_location="cpu")
        new_model.load_state_dict(state_dict)
        new_model.to(new_model.dtype)

        return new_model
