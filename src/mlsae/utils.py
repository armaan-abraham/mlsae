import os
import json
import torch
import numpy as np
import einops
import pprint
from pathlib import Path
import transformer_lens
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from dataclasses import dataclass
import tqdm

this_dir = Path(__file__).parent

site_to_size = {
    "mlp_out": 512,
    "post": 2048,
    "resid_pre": 512,
    "resid_mid": 512,
    "resid_post": 512,
}


@dataclass
class Config:
    seed: int = 49
    batch_size: int = 4096
    buffer_mult: int = 384
    lr: float = 1e-4
    num_tokens: int = int(2e9)
    l1_coeff: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.99
    dict_mult: int = 32
    seq_len: int = 128
    enc_dtype: str = "fp32"
    remove_rare_dir: bool = False
    model_name: str = "gelu-2l"
    site: str = "mlp_out"
    layer: int = 0
    device: str = "cuda:0"

    @property
    def model_batch_size(self) -> int:
        return self.batch_size // self.seq_len * 16

    @property
    def buffer_size(self) -> int:
        return self.batch_size * self.buffer_mult

    @property
    def buffer_batches(self) -> int:
        return self.buffer_size // self.seq_len

    @property
    def act_name(self) -> str:
        return transformer_lens.utils.get_act_name(self.site, self.layer)

    @property
    def act_size(self) -> int:
        return site_to_size[self.site]

    @property
    def dict_size(self) -> int:
        return self.act_size * self.dict_mult

    @property
    def name(self) -> str:
        return f"{self.model_name}_{self.layer}_{self.dict_size}_{self.site}"


cfg = Config()
pprint.pprint(cfg)

# Create directories
cache_dir = this_dir / "cache"
data_dir = this_dir / "data"
model_dir = this_dir / "checkpoints"

for dir_path in [cache_dir, data_dir, model_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Set environment variables
os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
os.environ["DATASETS_CACHE"] = str(cache_dir)

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
np.random.seed(cfg.seed)

model = (
    transformer_lens.HookedTransformer.from_pretrained(cfg.model_name)
    .to(DTYPES[cfg.enc_dtype])
    .to(cfg.device)
)

n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab


def load_encoder_training_data(shuffle=True):
    training_data_path = data_dir / "tokens.pt"
    if not training_data_path.exists():
        print("Fetching training data...")
        data = load_dataset(
            "NeelNanda/c4-code-tokenized-2b", split="train", cache_dir=cache_dir
        )
        data.save_to_disk(data_dir / "c4_code_tokenized_2b.hf")
        data.set_format(type="torch", columns=["tokens"])
        all_tokens = data["tokens"]
        all_tokens = einops.rearrange(
            all_tokens, "batch (x seq_len) -> (batch x) seq_len", x=8, seq_len=128
        )
        all_tokens[:, 0] = model.tokenizer.bos_token_id
        all_tokens = all_tokens[torch.randperm(all_tokens.shape[0])]
        torch.save(all_tokens, training_data_path)
    else:
        all_tokens = torch.load(training_data_path)

    if shuffle:
        all_tokens = all_tokens[torch.randperm(all_tokens.shape[0])]
    return all_tokens


class Buffer:
    def __init__(self):
        self.all_tokens = load_encoder_training_data()
        self.buffer = torch.zeros(
            (cfg.buffer_size, cfg.act_size), dtype=torch.bfloat16, device=cfg.device
        )
        self.token_pointer = 0
        self.first = True
        self.refresh()

    @torch.no_grad()
    def refresh(self):
        self.pointer = 0
        with torch.autocast("cuda", torch.bfloat16):
            if self.first:
                num_batches = cfg.buffer_batches
            else:
                num_batches = cfg.buffer_batches // 2
            self.first = False
            for _ in range(0, num_batches, cfg.model_batch_size):
                tokens = self.all_tokens[
                    self.token_pointer : self.token_pointer + cfg.model_batch_size
                ]
                _, cache = model.run_with_cache(
                    tokens, stop_at_layer=cfg.layer + 1, names_filter=cfg.act_name
                )
                acts = cache[cfg.act_name].reshape(-1, cfg.act_size)

                self.buffer[self.pointer : self.pointer + acts.shape[0]] = acts
                self.pointer += acts.shape[0]
                self.token_pointer += cfg.model_batch_size

        self.pointer = 0
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0]).to(cfg.device)]

    @torch.no_grad()
    def next(self):
        out = self.buffer[self.pointer : self.pointer + cfg.batch_size]
        self.pointer += cfg.batch_size
        if self.pointer > self.buffer.shape[0] // 2 - cfg.batch_size:
            self.refresh()
        return out
