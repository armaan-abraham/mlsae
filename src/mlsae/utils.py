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
import threading
import concurrent.futures  # We'll use ThreadPoolExecutor
# ... all your other imports ...

this_dir = Path(__file__).parent

@dataclass
class DataConfig:
    seed: int = 49
    buffer_batch_size_tokens: int = 8192
    buffer_size_buffer_batch_size_mult: int = 256
    seq_len: int = 64
    model_batch_size_seqs: int = 128
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

for k, v in data_cfg.__dict__.items():
    print(f"{k}: {v}")
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

# ---- Prepare multiple model replicas (one per GPU) ----
# device_count = torch.cuda.device_count()
device_count = 4
print(f"Device count: {device_count}")
    
models_on_each_gpu = []
for dev_id in range(device_count):
    dev_str = f"cuda:{dev_id}"
    model_i = (
        transformer_lens.HookedTransformer
        .from_pretrained(data_cfg.model_name, device=dev_str)
        .to(DTYPES[data_cfg.enc_dtype])
    )
    models_on_each_gpu.append(model_i)

for i, model_i in enumerate(models_on_each_gpu):
    # Check the first parameter just to confirm which device it lives on
    param_name, param = next(model_i.named_parameters())
    print(f"Model {i}, param {param_name} => {param.device}")

def stream_training_chunks():
    dataset_iter = load_dataset(
        data_cfg.dataset_name,
        split="train",
        streaming=True,
        cache_dir=cache_dir,
    )
    row_batch_iter = dataset_iter.batch(
        data_cfg.buffer_refresh_size_seqs // data_cfg.seqs_per_dataset_row
    )
    for row_batch in row_batch_iter:
        yield torch.tensor(row_batch["input_ids"], dtype=torch.int32, device=data_cfg.device)

class Buffer:
    """
    Streams tokens and fills self.buffer with model activations.
    """
    def __init__(self):
        self.token_stream = stream_training_chunks()
        self.buffer = torch.zeros(
            (data_cfg.buffer_size_tokens, data_cfg.act_size),
            dtype=DTYPES[data_cfg.enc_dtype],
            device="cpu",
        )
        self.pointer = 0
        self.first = True
        self.refresh()

    @torch.no_grad()
    def refresh(self):
        print("Refreshing buffer...")
        start_time = time.time()
        self.pointer = 0

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
        seqs[:, 0] = models_on_each_gpu[0].tokenizer.bos_token_id
        assert seqs.shape[0] % data_cfg.buffer_refresh_size_seqs == 0

        # Each GPU processes an equal portion of seqs
        total_seqs = seqs.shape[0]
        num_devices = len(models_on_each_gpu)
        chunk_size = total_seqs // num_devices

        with torch.autocast("cuda", DTYPES[data_cfg.enc_dtype]):
            # Collect partial activations from each thread
            def run_inference_on_device(device_idx, seq_chunk):
                device_str = f"cuda:{device_idx}"
                local_model = models_on_each_gpu[device_idx]
                batch_size_seqs = data_cfg.model_batch_size_seqs
                all_acts = []

                start_time = time.time()
                seq_chunk = seq_chunk.to(device_str)

                for start in range(0, seq_chunk.shape[0], batch_size_seqs):
                    start_time = time.time()
                    sub_chunk = seq_chunk[start : start + batch_size_seqs]
                    _, cache = local_model.run_with_cache(
                        sub_chunk,
                        stop_at_layer=data_cfg.layer + 1,
                        names_filter=data_cfg.act_name,
                        return_cache_object=True,
                    )
                    print(f"Cache device {device_idx}: {dir(cache)}")
                    acts = cache.cache_dict[data_cfg.act_name]  # shape: [batch_size, seq_len, act_size]
                    print(f"Acts device {device_idx}: {acts.device}")
                    acts = acts.reshape(batch_size_seqs * data_cfg.seq_len, data_cfg.act_size)
                    all_acts.append(acts)
                    print(f"Running inference on device {device_idx}, iter {start}... {time.time() - start_time:.2f}")

                return torch.cat(all_acts, dim=0)

            seqs_list = []
            for i in range(num_devices):
                start_idx = i * chunk_size
                # Last chunk takes the remainder in case total_seqs not divisible
                end_idx = total_seqs if i == (num_devices - 1) else (start_idx + chunk_size)
                seqs_list.append(seqs[start_idx:end_idx])

            print("Running inference...")
            # Fire up threads
            results = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for i in range(num_devices):
                    print(f"Running inference on device {i}, {seqs_list[i].shape}...")
                    futures.append(
                        executor.submit(
                            run_inference_on_device,
                            i,  # device_idx
                            seqs_list[i],
                        )
                    )
                for f in futures:
                    results.append(f.result())

        print("Combining partial results...")
        # Combine partial results
        # Each element of results is shape [chunk_size * seq_len, act_size] (except for possibly the last chunk)
        # We will concat them into a single tensor
        all_acts = torch.cat([acts.to("cpu") for acts in results], dim=0)

        # Now place combined acts in self.buffer
        acts_count = all_acts.shape[0]
        assert acts_count <= self.buffer.shape[0], "Buffer overflow"
        self.buffer[:acts_count] = all_acts
        self.pointer = acts_count

        # Shuffle
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0])]
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
