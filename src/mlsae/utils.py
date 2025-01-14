import os
import torch
import numpy as np
import einops
import pprint
from pathlib import Path
import transformer_lens
from datasets import load_dataset
from dataclasses import dataclass


this_dir = Path(__file__).parent

site_to_size = {
    "mlp_out": 512,
    "post": 2048,
    "resid_pre": 512,
    "resid_mid": 512,
    "resid_post": 512,
}


@dataclass
class DataConfig:
    """
    Configuration for data buffering and dataset loading.
    This config should only contain parameters relevant to
    dataset handling and buffering logic, not autoencoder training.
    """

    seed: int = 49
    batch_size: int = 8192
    buffer_mult: int = 384
    seq_len: int = 128
    dataset_seq_len: int = 1024
    enc_dtype: str = "fp32"
    remove_rare_dir: bool = False
    model_name: str = "gpt2-small"
    site: str = "hook_resid_pre"
    layer: int = 6
    device: str = "cuda:0"
    rows_to_load: int = -1
    dataset_name: str = "apollo-research/Skylion007-openwebtext-tokenizer-gpt2"
    cache_name: str = "openwebtext_1m"
    test_data_ratio: float = 0.05

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


data_cfg = DataConfig()
pprint.pprint(data_cfg)

# Create directories
cache_dir = this_dir / "cache"
data_dir = this_dir / "data"
model_dir = this_dir / "checkpoints"

for dir_path in [cache_dir, data_dir, model_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Set environment variables for caching
os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
os.environ["DATASETS_CACHE"] = str(cache_dir)

# Map string dtype keys to torch dtypes
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

# Seed for reproducibility
np.random.seed(data_cfg.seed)

# Load the transformer model used for data encoding
model = (
    transformer_lens.HookedTransformer.from_pretrained(data_cfg.model_name)
    .to(DTYPES[data_cfg.enc_dtype])
    .to(data_cfg.device)
)

n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab


def load_encoder_training_data(shuffle=True):
    """
    Loads or downloads the tokenized dataset. Returns a tensor of tokens
    for training. Also splits off a given number of test samples and
    saves them to a separate file in the same cache folder.
    """
    split_cache_dir = data_dir / data_cfg.cache_name
    split_cache_dir.mkdir(parents=True, exist_ok=True)

    train_data_path = split_cache_dir / "tokens_train.pt"
    test_data_path = split_cache_dir / "tokens_test.pt"

    # If both train and test files already exist, simply load the train set and return it.
    if train_data_path.exists() and test_data_path.exists():
        print("Detected existing split train/test data; loading from disk...")
        train_tokens = torch.load(train_data_path)
        # (Optional) If you also want to load test data in the future, you can do:
        # test_tokens = torch.load(test_data_path)
        # For now, we only return train_tokens to keep consistent with original usage.
        if shuffle:
            train_tokens = train_tokens[torch.randperm(train_tokens.shape[0])]
        print(f"Loaded {train_tokens.shape[0]} training rows")
        return train_tokens

    # Otherwise, fetch data and split.
    # We won't use the old single-file path, so skip that logic entirely.
    print("Fetching training data...")
    data = load_dataset(
        data_cfg.dataset_name, split="train", cache_dir=cache_dir
    )
    if data_cfg.rows_to_load > 0:
        data = data.select(range(min(data_cfg.rows_to_load, len(data))))
    data.save_to_disk(data_dir / f"{data_cfg.cache_name}.hf")
    data.set_format(type="torch", columns=["input_ids"])
    all_tokens = data["input_ids"]
    assert (
        all_tokens.shape[1] == data_cfg.dataset_seq_len
    ), "Dataset sequence length must match dataset_seq_len"

    # Rearrange to shape (N, seq_len) for contiguous sequences
    all_tokens = einops.rearrange(
        all_tokens,
        "batch (x seq_len) -> (batch x) seq_len",
        x=data_cfg.dataset_seq_len // data_cfg.seq_len,
        seq_len=data_cfg.seq_len,
    )

    # Set first token to bos_token for each sequence
    all_tokens[:, 0] = model.tokenizer.bos_token_id

    # Shuffle once overall
    all_tokens = all_tokens[torch.randperm(all_tokens.shape[0])]

    # Split off test samples
    num_test = int(all_tokens.shape[0] * data_cfg.test_data_ratio)
    test_tokens = all_tokens[:num_test]
    train_tokens = all_tokens[num_test:]

    # Save both splits
    torch.save(train_tokens, train_data_path)
    torch.save(test_tokens, test_data_path)
    print(
        f"Created train/test split: "
        f"{train_tokens.shape[0]} train rows, "
        f"{test_tokens.shape[0]} test rows"
    )

    # Return only the training portion for downstream usage
    return train_tokens


class Buffer:
    """
    A buffer that streams activation data from the model's intermediate representations.
    Pulls new data in chunks, caches it, and shuffles it.
    """

    def __init__(self):
        self.all_tokens = load_encoder_training_data()
        # Create a buffer for the activations
        self.buffer = torch.zeros(
            (data_cfg.buffer_size, data_cfg.act_size),
            dtype=DTYPES[data_cfg.enc_dtype],
            device=data_cfg.device,
        )
        self.token_pointer = 0
        self.first = True
        self.refresh()

    @torch.no_grad()
    def refresh(self):
        """
        Refill and shuffle the buffer with fresh activations from the model.
        """
        self.pointer = 0
        with torch.autocast("cuda", DTYPES[data_cfg.enc_dtype]):
            if self.first:
                num_batches = data_cfg.buffer_batches
            else:
                num_batches = data_cfg.buffer_batches // 2
            self.first = False

            for _ in range(0, num_batches, data_cfg.model_batch_size):
                tokens = self.all_tokens[
                    self.token_pointer : self.token_pointer + data_cfg.model_batch_size
                ]
                _, cache = model.run_with_cache(
                    tokens,
                    stop_at_layer=data_cfg.layer + 1,
                    names_filter=data_cfg.act_name,
                )
                print(cache[data_cfg.act_name].shape)
                acts = cache[data_cfg.act_name].reshape(-1, data_cfg.act_size)

                # Copy these activations to the buffer
                self.buffer[self.pointer : self.pointer + acts.shape[0]] = acts
                self.pointer += acts.shape[0]
                self.token_pointer += data_cfg.model_batch_size

        # Reset pointer and shuffle buffer
        self.pointer = 0
        self.buffer = self.buffer[
            torch.randperm(self.buffer.shape[0]).to(data_cfg.device)
        ]

    @torch.no_grad()
    def next(self):
        """
        Fetch the next batch from the buffer. If we're halfway through the buffer,
        we refresh the buffer (to keep half of it always fresh).
        """
        out = self.buffer[self.pointer : self.pointer + data_cfg.batch_size]
        self.pointer += data_cfg.batch_size
        if self.pointer > self.buffer.shape[0] // 2 - data_cfg.batch_size:
            self.refresh()
        return out
