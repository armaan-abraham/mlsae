import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlsae.data import DTYPES

model_dir = Path(__file__).parent / "checkpoints"

ZERO_ACT_THRESHOLD = 1e-6

class DeepSAE(nn.Module):
    """
    Multi-layer sparse autoencoder with a single sparse representation layer,
    where we replace L1 regularization with a top-k activation mask.
    """

    def __init__(
        self,
        encoder_dim_mults: list[int],
        sparse_dim_mult: int,
        decoder_dim_mults: list[int],
        act_size: int,
        name: str = None,
        enc_dtype: str = "fp32",
        device: str = "cuda:0",
        l1_lambda: float = 0.1,
        leaky_relu_slope: float = 0.05
    ):
        super().__init__()

        assert all(mult == 1 for mult in encoder_dim_mults)
        self.name = name
        self.encoder_dims = [dim * act_size for dim in encoder_dim_mults]
        self.decoder_dims = [dim * act_size for dim in decoder_dim_mults]
        self.sparse_dim = sparse_dim_mult * act_size
        self.act_size = act_size
        self.enc_dtype = enc_dtype
        self.dtype = DTYPES[enc_dtype]
        self.device = device
        self.l1_lambda = l1_lambda
        self.leaky_relu_slope = leaky_relu_slope
        print(f"Encoder dims: {self.encoder_dims}")
        print(f"Decoder dims: {self.decoder_dims}")
        print(f"Sparse dim: {self.sparse_dim}")
        print(f"L1 lambda: {self.l1_lambda}")
        print(f"Device: {self.device}")
        print(f"Dtype: {self.dtype}")

        # Parameter groups for L2: stored in class-level lists
        self.params_with_decay = []
        self.params_no_decay = []

        # --------------------------------------------------------
        # Build encoder
        # --------------------------------------------------------
        encoder_layers = []
        in_dim = self.act_size

        # Bulid the repeated (Linear -> ReLU -> LayerNorm) blocks
        for dim in self.encoder_dims:
            linear_layer = self._create_linear_layer(
                in_dim, dim, apply_weight_decay=True
            )
            encoder_layers.append(linear_layer)
            encoder_layers.append(nn.Tanh())
            encoder_layers.append(nn.LayerNorm(dim))
            in_dim = dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Final encoder layer creates sparse representation
        self.sparse_layer = self._create_linear_layer(
            in_dim, self.sparse_dim, apply_weight_decay=False
        )


        # --------------------------------------------------------
        # Build decoder
        # --------------------------------------------------------
        decoder_layers = []
        out_dim = self.sparse_dim

        for dim in self.decoder_dims:
            linear_layer = self._create_linear_layer(
                out_dim, dim, apply_weight_decay=True
            )
            decoder_layers.append(linear_layer)
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.LayerNorm(dim))
            out_dim = dim

        # Final decoder layer -> output
        linear_out = self._create_linear_layer(
            out_dim, self.act_size, apply_weight_decay=False
        )
        decoder_layers.append(linear_out)

        self.decoder = nn.Sequential(*decoder_layers)

        # Initialize weights, move to device/dtype
        self.init_weights()
        self.to(self.device, self.dtype)

    def _create_linear_layer(self, in_dim, out_dim, apply_weight_decay: bool):
        """
        Creates a Linear(in_dim, out_dim) and assigns the weight to
        params_with_decay or params_no_decay accordingly. The bias
        is always placed in params_no_decay (commonly done in PyTorch).
        """
        layer = nn.Linear(in_dim, out_dim)

        if apply_weight_decay:
            self.params_with_decay.append(layer.weight)
        else:
            self.params_no_decay.append(layer.weight)

        if layer.bias is not None:
            self.params_no_decay.append(layer.bias)

        return layer

    def get_param_groups(self, weight_decay=1e-4):
        """
        Return parameter groups for the optimizer:
          - One group with weight_decay
          - One group without weight_decay
        """
        return [
            {"params": self.params_with_decay, "weight_decay": weight_decay},
            {"params": self.params_no_decay, "weight_decay": 0.0},
        ]

    def init_weights(self):
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        # Encode
        if self.encoder_dims:
            resid = self.encoder(x)
            resid += x
        else:
            resid = x

        feature_acts = F.relu(self.sparse_layer(resid))

        # Decode
        reconstructed = self.decoder(feature_acts)

        # MSE reconstruction loss
        mse_loss = (reconstructed.float() - x.float()).pow(2).mean()
        
        # L1 penalty on feature activations
        l1_loss = self.l1_lambda * feature_acts.abs().mean()
        loss = mse_loss + l1_loss

        # Number of nonzero activations (for monitoring)
        nonzero_acts = (feature_acts > ZERO_ACT_THRESHOLD).float().sum(dim=1).mean()

        return loss, mse_loss, l1_loss, nonzero_acts, feature_acts, reconstructed

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
            "l1_lambda": self.l1_lambda,
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
            l1_lambda=config_dict["l1_lambda"],
        )

        state_dict = torch.load(load_path / f"{version}.pt", map_location="cpu")
        new_model.load_state_dict(state_dict)
        new_model.to(new_model.dtype)

        return new_model

    @torch.no_grad()
    def resample_sparse_features(self, idx):
        print("Resampling sparse features...")
        new_W_enc = torch.zeros_like(self.encoder[-1].weight)
        new_W_dec = torch.zeros_like(self.decoder[0].weight)
        nn.init.kaiming_normal_(new_W_enc, nonlinearity="relu")
        nn.init.kaiming_normal_(new_W_dec, nonlinearity="relu")

        new_b_enc = torch.zeros_like(self.encoder[-1].bias)

        self.encoder[-1].weight.data[idx] = new_W_enc[idx]
        self.encoder[-1].bias.data[idx] = new_b_enc[idx]
        self.decoder[0].weight.data[:, idx] = new_W_dec[:, idx]
