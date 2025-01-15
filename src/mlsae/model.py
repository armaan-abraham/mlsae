import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlsae.utils import DTYPES

model_dir = Path(__file__).parent / "checkpoints"


class DeepSAE(nn.Module):
    """
    Multi-layer sparse autoencoder with a single sparse representation layer.
    """

    def __init__(
        self,
        encoder_dim_mults: list[int],
        sparse_dim_mult: int,
        decoder_dim_mults: list[int],
        act_size: int,
        top_k: int,
        enc_dtype: str = "fp32",
        device: str = "cuda:0",
    ):
        super().__init__()

        self.encoder_dims = [dim * act_size for dim in encoder_dim_mults]
        self.decoder_dims = [dim * act_size for dim in decoder_dim_mults]
        self.sparse_dim = sparse_dim_mult * act_size
        self.act_size = act_size
        self.k = top_k
        assert self.k < self.sparse_dim, "top_k must be less than sparse_dim"
        self.enc_dtype = enc_dtype
        self.dtype = DTYPES[enc_dtype]
        self.device_name = device

        print(f"Encoder dims: {self.encoder_dims}")
        print(f"Decoder dims: {self.decoder_dims}")
        print(f"Sparse dim: {self.sparse_dim}")
        print(f"K: {self.k}")
        print(f"Device: {self.device_name}")
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
        # We'll apply ReLU and top-k in forward()

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
        self.to(self.device_name, self.dtype)

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

        # Top-k (per-batch-element) in the sparse layer
        if self.k < feature_acts.shape[1]:
            # mask out everything except top k in each row
            _, idxs = torch.topk(feature_acts, self.k, dim=1)
            assert idxs.shape == (
                feature_acts.shape[0],
                self.k,
            ), f"Top-k indices must have shape (batch_size, k), got {idxs.shape}"
            mask = torch.zeros_like(feature_acts, dtype=feature_acts.dtype).scatter_(
                1, idxs, 1.0
            )
            assert mask.sum(dim=1).allclose(
                torch.tensor(self.k, dtype=feature_acts.dtype)
            )
            feature_acts = feature_acts * mask
            feature_acts = torch.relu(feature_acts)

        # Decode
        reconstructed = self.decoder(feature_acts)
        # MSE loss as reconstruction loss
        loss = (reconstructed.float() - x.float()).pow(2).mean()

        return loss, feature_acts

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
            "top_k": self.k,
            "enc_dtype": self.enc_dtype,
            "device": self.device_name,
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
            encoder_dim_mults=config_dict["encoder_dims"],
            sparse_dim_mult=config_dict["sparse_dim"] // config_dict["act_size"],
            decoder_dim_mults=config_dict["decoder_dims"],
            act_size=config_dict["act_size"],
            top_k=config_dict["top_k"],
            enc_dtype=config_dict["enc_dtype"],
            device=config_dict["device"],
        )

        state_dict = torch.load(load_path / f"{version}.pt")
        new_model.load_state_dict(state_dict)
        new_model.to(new_model.device, new_model.dtype)

        return new_model
