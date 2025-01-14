import os
import torch
import numpy as np
import einops
import pprint
from pathlib import Path
import transformer_lens
from datasets import load_dataset
from dataclasses import dataclass
import time

this_dir = Path(__file__).parent

"""
- How does this work between buffer size and rows_to_load?
"""

@dataclass
class DataConfig:
    seed: int = 49
    buffer_batch_size_tokens: int = 8192
    buffer_size_buffer_batch_size_mult: int = 256
    seq_len: int = 64
    model_batch_size_seqs: int = 64
    dataset_row_len: int = 1024
    enc_dtype: str = "fp32"
    remove_rare_dir: bool = False
    model_name: str = "gpt2-small"
    site: str = "resid_pre"
    layer: int = 6
    act_size: int = 768 # This is checked when we run the model
    device: str = "cuda:0"
    dataset_name: str = "apollo-research/Skylion007-openwebtext-tokenizer-gpt2"
    test_data_ratio: float = 0.05

    @property
    def seqs_per_dataset_row(self) -> int:
        return self.dataset_row_len // self.seq_len

    @property
    def buffer_size_tokens(self) -> int:
        return self.buffer_batch_size_tokens * self.buffer_size_buffer_batch_size_mult

    @property
    def buffer_size_seqs(self) -> int:
        return self.buffer_size_tokens // self.seq_len

    @property
    def buffer_refresh_size_seqs(self) -> int:
        return self.buffer_size_seqs // 2

    @property
    def act_name(self) -> str:
        return transformer_lens.utils.get_act_name(self.site, self.layer)


data_cfg = DataConfig()
# Print regular attributes
print("Regular attributes:")
for k, v in data_cfg.__dict__.items():
    print(f"{k}: {v}")

# Print properties
print("\nProperties:")
properties = [attr for attr in dir(DataConfig) if isinstance(getattr(DataConfig, attr), property)]
for prop in properties:
    print(f"{prop}: {getattr(data_cfg, prop)}")

assert data_cfg.dataset_row_len % data_cfg.seq_len == 0
assert data_cfg.buffer_refresh_size_seqs % data_cfg.model_batch_size_seqs == 0

# Create directories
cache_dir = this_dir / "cache"
data_dir = this_dir / "data"
model_dir = this_dir / "checkpoints"

for dir_path in [cache_dir, data_dir, model_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Set environment variables for caching
os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
os.environ["DATASETS_CACHE"] = str(cache_dir)

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
np.random.seed(data_cfg.seed)

# Load the transformer
model = (
    transformer_lens.HookedTransformer.from_pretrained(data_cfg.model_name)
    .to(DTYPES[data_cfg.enc_dtype])
    .to(data_cfg.device)
)

def stream_training_chunks():
    # Load a streaming dataset
    dataset_iter = load_dataset(
        data_cfg.dataset_name,
        split="train",
        streaming=True,
        cache_dir=cache_dir,
    )

    row_batch_iter = dataset_iter.batch(data_cfg.buffer_refresh_size_seqs // data_cfg.seqs_per_dataset_row)

    for row_batch in row_batch_iter:
        yield torch.tensor(row_batch["input_ids"], dtype=torch.int32, device=data_cfg.device)


class Buffer:
    """
    Streams tokens from the huggingface dataset in seq_len chunks.
    On refresh, runs them through the model (up to buffer_seqs
    worth) and saves the activations in self.buffer.
    """
    def __init__(self):

        self.token_stream = stream_training_chunks()

        # Buffer to hold activations
        self.buffer = torch.zeros(
            (data_cfg.buffer_size_tokens, data_cfg.act_size),
            dtype=DTYPES[data_cfg.enc_dtype],
            device=data_cfg.device,
        )
        self.pointer = 0
        self.first = True

        # Fill up the buffer at startup
        self.refresh()

    @torch.no_grad()
    def refresh(self):
        """
        Refill and shuffle the buffer with fresh activations from the model.
        Instead of loading tokens in batches of size equal to data_cfg.model_batch_size, 
        we now load all required tokens for an entire refresh in one go, then process 
        them in chunks.
        """
        print("Refreshing buffer...")
        start_time = time.time()
        self.pointer = 0

        # Gather all rows needed for this refresh
        rows = next(self.token_stream)
        assert rows is not None, "No more data available from token_stream."

        if self.first:
            rows_next = next(self.token_stream)
            rows = torch.cat([rows, rows_next], dim=0)

        self.first = False

        print("Preprocessing rows...")
        seqs = einops.rearrange(
            rows,
            "row (seq token) -> (row seq) token",
            token=data_cfg.seq_len,
            seq=data_cfg.seqs_per_dataset_row,
        )

        print(f"seqs.shape: {seqs.shape}")

        # Force BOS token
        seqs[:, 0] = model.tokenizer.bos_token_id

        print("Running model...")

        assert seqs.shape[0] % data_cfg.buffer_refresh_size_seqs == 0

        num_model_batches = seqs.shape[0] // data_cfg.model_batch_size_seqs

        with torch.autocast("cuda", DTYPES[data_cfg.enc_dtype]):
            loaded_batches = 0
            while loaded_batches < num_model_batches:
                start_idx = loaded_batches * data_cfg.model_batch_size_seqs
                end_idx = start_idx + data_cfg.model_batch_size_seqs

                model_batch = seqs[start_idx:end_idx]

                # Forward pass with caching
                _, cache = model.run_with_cache(
                    model_batch,
                    stop_at_layer=data_cfg.layer + 1,
                    names_filter=data_cfg.act_name,
                )

                acts = cache[data_cfg.act_name]
                # Expect shape: (data_cfg.model_batch_size_seqs, data_cfg.seq_len, data_cfg.act_size)
                assert acts.shape == (data_cfg.model_batch_size_seqs, data_cfg.seq_len, data_cfg.act_size)
                # Flatten from [batch, seq_len, act_size] -> [batch*seq_len, act_size]
                acts = acts.reshape(
                    data_cfg.model_batch_size_seqs * data_cfg.seq_len, 
                    data_cfg.act_size
                )

                self.buffer[self.pointer : self.pointer + acts.shape[0]] = acts
                self.pointer += acts.shape[0]

                assert self.pointer <= self.buffer.shape[0], "Buffer overflow"

                loaded_batches += 1

        # Shuffle the buffer
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0], device=data_cfg.device)]
        self.pointer = 0
        print(f"Buffer refreshed in {time.time() - start_time:.2f} seconds.")

    @torch.no_grad()
    def next(self):
        """
        Fetch the next batch from the buffer. If pointer goes beyond half,
        refresh the buffer so it never goes stale.
        """
        out = self.buffer[self.pointer : self.pointer + data_cfg.buffer_batch_size_tokens]
        self.pointer += data_cfg.buffer_batch_size_tokens
        if self.pointer > self.buffer.shape[0] // 2 - data_cfg.buffer_batch_size_tokens:
            self.refresh()
        return out
