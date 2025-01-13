import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import DTYPES, cfg
import json
from pathlib import Path

model_dir = Path(__file__).parent / "checkpoints"


class MultiLayerSAE(nn.Module):
    """
    Multi-layer sparse autoencoder with a single sparse representation layer.
    The user specifies (dimensions are specified as multiples of act_size):
    - A list of encoder-layer sizes: e.g., [dim1, dim2, ..., dim_n].
    - A single sparse representation dimension (largest dimension).
    - A list of decoder-layer sizes: e.g., [dim1, dim2, ..., dim_m].
    Only the sparse representation layer has an L1 penalty to promote sparsity.
    """

    def __init__(
        self,
        encoder_dim_mults: list[int],
        sparse_dim_mult: int,
        decoder_dim_mults: list[int],
        act_size: int,
        l1_coeff: float,
        enc_dtype: str = "fp32",
        device: str = "cuda:0",
    ):
        super().__init__()

        self.encoder_dims = [dim * act_size for dim in encoder_dim_mults]
        self.decoder_dims = [dim * act_size for dim in decoder_dim_mults]
        self.sparse_dim = sparse_dim_mult * act_size
        self.act_size = act_size
        self.l1_coeff = l1_coeff
        self.enc_dtype = enc_dtype  # store the string key (e.g. "fp32")
        self.dtype = DTYPES[enc_dtype]
        self.device_name = device
        print(f"Encoder dims: {self.encoder_dims}")
        print(f"Decoder dims: {self.decoder_dims}")
        print(f"Sparse dim: {self.sparse_dim}")

        # Build encoder
        layers = []
        in_dim = self.act_size
        for dim in self.encoder_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU())
            in_dim = dim

        # Sparse representation
        layers.append(nn.Linear(in_dim, self.sparse_dim))
        # We apply ReLU in forward() below

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

        # Finally, ensure the module is on the right device/dtype
        self.to(self.device, self.dtype)

    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        # Sparse representation with ReLU
        feature_acts = F.relu(encoded)

        # Decode
        reconstructed = self.decoder(feature_acts)
        # Compute reconstruction loss (MSE)
        l2_loss = (reconstructed.float() - x.float()).pow(2).mean()

        # L1 penalty only on the sparse layer
        l1_loss = self.l1_coeff * feature_acts.float().abs().sum()

        loss = l2_loss + l1_loss
        return loss, feature_acts, l2_loss, l1_loss

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        """
        Unit norm only on final decoder layer
        """
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
        """
        Saves model state dict and config as JSON. If version is None, auto-increment it.
        """
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
            "l1_coeff": self.l1_coeff,
            # Save the string key ("fp32", "fp16", etc.) rather than "torch.float32"
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
            encoder_dims=config_dict["encoder_dims"],
            sparse_dim=config_dict["sparse_dim"],
            decoder_dims=config_dict["decoder_dims"],
            act_size=config_dict["act_size"],
            l1_coeff=config_dict["l1_coeff"],
            enc_dtype=config_dict["enc_dtype"],
            device=config_dict["device"],
        )

        state_dict = torch.load(
            load_path / f"{version}.pt",
        )
        new_model.load_state_dict(state_dict)
        new_model.to(new_model.device, new_model.dtype)

        return new_model
