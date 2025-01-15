import os
import time
from dataclasses import dataclass
from pathlib import Path

import einops
import torch
import torch.multiprocessing as mp
import transformer_lens
from datasets import load_dataset

this_dir = Path(__file__).parent


@dataclass
class DataConfig:
    seed: int = 49
    buffer_batch_size_tokens: int = 16384
    buffer_size_buffer_batch_size_mult: int = 1024
    seq_len: int = 64
    model_batch_size_seqs: int = 128
    dataset_row_len: int = 1024
    enc_dtype: str = "fp32"
    remove_rare_dir: bool = False
    model_name: str = "gpt2-small"
    site: str = "resid_pre"
    layer: int = 6
    act_size: int = 768
    device: str = "cuda:0"
    dataset_name: str = "apollo-research/Skylion007-openwebtext-tokenizer-gpt2"
    llm_device_count: int = 1

    eval_data_seed: int = 59
    eval_tokens: int = int(1e4)

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
    def buffer_refresh_size_rows(self) -> int:
        return self.buffer_refresh_size_seqs // self.seqs_per_dataset_row

    @property
    def act_name(self) -> str:
        return transformer_lens.utils.get_act_name(self.site, self.layer)
    

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

data_cfg = DataConfig()

# Create directories
this_dir = Path(__file__).parent
cache_dir = this_dir / "cache"
data_dir = this_dir / "data"
model_dir = this_dir / "checkpoints"

for dir_path in [cache_dir, data_dir, model_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Set environment variables for caching
os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
os.environ["DATASETS_CACHE"] = str(cache_dir)


def worker(tasks: mp.Queue, results: mp.Queue, device_id: int, data_cfg_dict: dict):
    device = f"cuda:{device_id}"
    local_data_cfg = DataConfig(**data_cfg_dict)

    DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    local_dtype = DTYPES[local_data_cfg.enc_dtype]

    local_model = transformer_lens.HookedTransformer.from_pretrained(
        local_data_cfg.model_name, device=device
    ).to(local_dtype)

    try:
        while True:
            task = tasks.get()
            if task is None:
                break
            seq_chunk = task

            # BOS
            seq_chunk[:, 0] = local_model.tokenizer.bos_token_id

            batch_size_seqs = local_data_cfg.model_batch_size_seqs
            all_acts = []
            seq_chunk = seq_chunk.to(device)

            with torch.autocast("cuda", local_dtype):
                for start in range(0, seq_chunk.shape[0], batch_size_seqs):
                    sub_chunk = seq_chunk[start : start + batch_size_seqs]
                    _, cache = local_model.run_with_cache(
                        sub_chunk,
                        stop_at_layer=local_data_cfg.layer + 1,
                        names_filter=local_data_cfg.act_name,
                        return_cache_object=True,
                    )
                    acts = cache.cache_dict[local_data_cfg.act_name]
                    acts = acts.reshape(
                        acts.shape[0] * acts.shape[1], local_data_cfg.act_size
                    )
                    all_acts.append(acts.cpu())

            results.put(torch.cat(all_acts, dim=0).to("cpu"))

    except Exception as e:
        results.put(e)
        raise


def stream_training_chunks(data_cfg, cache_dir, buffer_size: int, eval: bool = False):
    dataset_iter = load_dataset(
        data_cfg.dataset_name,
        split="train",
        streaming=True,
        cache_dir=cache_dir,
    )
    dataset_iter = dataset_iter.shuffle(
        buffer_size=buffer_size,
        seed=data_cfg.seed if not eval else data_cfg.eval_data_seed,
    )
    row_batch_iter = dataset_iter.batch(
        buffer_size / 2
    )
    for row_batch in row_batch_iter:
        yield torch.tensor(
            row_batch["input_ids"], dtype=torch.int32, device=data_cfg.device
        )


class Buffer:
    """
    Streams tokens and fills self.buffer with model activations.
    """

    def __init__(self, eval: bool = False):
        print("Initializing buffer...")
        self.token_stream = stream_training_chunks(
            data_cfg,
            cache_dir,
            data_cfg.buffer_refresh_size_rows * 2,
            eval=eval,
        )
        self.buffer = torch.zeros(
            (data_cfg.buffer_size_tokens, data_cfg.act_size),
            dtype=torch.float32,  # or DTYPES[data_cfg.enc_dtype], etc.
            device="cpu",
        )
        mp.set_start_method("spawn", force=True)

        self.tasks = mp.Queue()
        self.results = mp.Queue()
        self.workers = []

        for i in range(data_cfg.llm_device_count):
            p = mp.Process(
                target=worker,
                args=(
                    self.tasks,
                    self.results,
                    i,
                    data_cfg.__dict__,
                ),
            )
            p.start()
            self.workers.append(p)

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

        total_seqs = seqs.shape[0]
        num_devices = data_cfg.llm_device_count
        chunk_size = total_seqs // num_devices

        seqs_list = []
        for i in range(num_devices):
            start_idx = i * chunk_size
            end_idx = total_seqs if i == (num_devices - 1) else (start_idx + chunk_size)
            seqs_list.append(seqs[start_idx:end_idx])

        print("Running inference...")

        for i in range(num_devices):
            self.tasks.put(seqs_list[i])

        results = []
        for i in range(num_devices):
            result = self.results.get()
            if isinstance(result, Exception):
                raise result
            results.append(result)

        print("Combining partial results...")
        all_acts = torch.cat(results, dim=0)

        acts_count = all_acts.shape[0]
        assert acts_count <= self.buffer.shape[0], "Buffer overflow"
        self.buffer[:acts_count] = all_acts
        self.pointer = acts_count

        # Shuffle
        idx = torch.randperm(acts_count)
        self.buffer[:acts_count] = self.buffer[idx]
        self.pointer = 0
        print(f"Buffer refreshed in {time.time() - start_time:.2f} seconds.")

    @torch.no_grad()
    def next(self):
        out = self.buffer[
            self.pointer : self.pointer + data_cfg.buffer_batch_size_tokens
        ]
        self.pointer += data_cfg.buffer_batch_size_tokens
        if self.pointer > self.buffer.shape[0] // 2 - data_cfg.buffer_batch_size_tokens:
            self.refresh()
        return out

    def __del__(self):
        print("Cleaning up Buffer resources...")
        # Send termination signal to all workers
        for _ in self.workers:
            self.tasks.put(None)  # None is the termination signal

        # Join all worker processes
        for worker in self.workers:
            worker.join()

        # Close the queues
        self.tasks.close()
        self.results.close()
