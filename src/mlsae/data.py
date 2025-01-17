import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import uuid
import time
from concurrent.futures import ThreadPoolExecutor

import einops
import torch
import torch.multiprocessing as mp
import transformer_lens
from datasets import load_dataset
from tqdm import tqdm
from line_profiler import profile

from transformers import AutoTokenizer

this_dir = Path(__file__).parent


@dataclass
class DataConfig:
    seed: int = 49
    buffer_batch_size_tokens: int = 131072
    buffer_size_buffer_batch_size_mult: int = 512
    seq_len: int = 64
    model_batch_size_seqs: int = 1024
    dataset_row_len: int = 256
    enc_dtype: str = "fp32"
    save_dtype: str = "bf16"
    model_name: str = "pythia-31m"
    model_tokenizer_name: str = "EleutherAI/pythia-31m"
    site: str = "resid_pre"
    layer: int = 3
    act_size: int = 256
    device: str = "cuda:0"
    buffer_device: str = "cpu"
    dataset_name: str = "Skylion007/openwebtext"
    dataset_subset_size: int = int(5e6)
    data_is_tokenized: bool = False
    llm_device_count: int = 2

    eval_data_seed: int = 59
    eval_tokens_buffer_batch_size_mult: int = 512

    caching: bool = True
    cache_id_fields: list = field(default_factory=lambda: [
        "seed",
        "model_name",
        "layer",
        "site",
        "seq_len",
        "act_size",
        "enc_dtype",
        "dataset_name",
        "dataset_row_len",
        "buffer_refresh_size_tokens",
        "save_dtype",
    ])

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
    def buffer_refresh_size_tokens(self) -> int:
        return self.buffer_refresh_size_seqs * self.seq_len

    @property
    def act_name(self) -> str:
        return transformer_lens.utils.get_act_name(self.site, self.layer)

    @property
    def eval_tokens(self) -> int:
        return self.buffer_batch_size_tokens * self.eval_tokens_buffer_batch_size_mult


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

data_cfg = DataConfig()

# Create directories
this_dir = Path(__file__).parent
cache_dir = this_dir / "cache"
activations_dir = cache_dir / "activations"
data_dir = this_dir / "data"
model_dir = this_dir / "checkpoints"

for dir_path in [cache_dir, activations_dir, data_dir, model_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Set environment variables for caching
os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
os.environ["DATASETS_CACHE"] = str(cache_dir)


def get_relevant_config_subset(cfg: DataConfig) -> dict:
    subset = {}
    for field_name in cfg.cache_id_fields:
        subset[field_name] = getattr(cfg, field_name)
    return subset


def find_existing_cache_folder(cfg: DataConfig, base_cache_dir: Path) -> Path:
    desired_subset = get_relevant_config_subset(cfg)
    for subdir in base_cache_dir.iterdir():
        if not subdir.is_dir():
            continue
        config_path = subdir / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                stored_cfg_subset = json.load(f)
            if stored_cfg_subset == desired_subset:
                return subdir
    return None


def create_new_cache_folder(cfg: DataConfig, base_cache_dir: Path) -> Path:
    new_cache_folder = base_cache_dir / str(uuid.uuid4())
    new_cache_folder.mkdir(parents=True, exist_ok=True)

    relevant_cfg = get_relevant_config_subset(cfg)
    with open(new_cache_folder / "config.json", "w") as f:
        json.dump(relevant_cfg, f, indent=2)

    return new_cache_folder


def worker(tasks: mp.Queue, results: mp.Queue, device_id: int, data_cfg_dict: dict):
    device = f"cuda:{device_id}"
    local_data_cfg = DataConfig(**data_cfg_dict)
    local_dtype = DTYPES[local_data_cfg.enc_dtype]

    local_model = transformer_lens.HookedTransformer.from_pretrained(
        local_data_cfg.model_name, device=device
    ).to(local_dtype)

    try:
        while True:
            task = tasks.get()
            if task is None:
                break

            seq_chunk = task  # task is expected to be a tensor of token IDs

            # BOS token (optional, but often helpful even if data is tokenized)
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
                    assert acts.shape[-1] == local_data_cfg.act_size, f"Expected {local_data_cfg.act_size} act size, got {acts.shape[-1]}"
                    acts = acts.reshape(
                        acts.shape[0] * acts.shape[1], local_data_cfg.act_size
                    )
                    all_acts.append(acts)

            results.put(torch.cat(all_acts, dim=0).to(local_data_cfg.buffer_device))

    except Exception as e:
        results.put(e)
        raise

assert not data_cfg.data_is_tokenized

def stream_training_chunks(
    data_cfg,
    cache_dir,
    eval: bool = False,
    thread_count: int = 32,
):
    start_time = time.time()
    
    # Load a subset of the dataset using either select or train_test_split
    # Method 1: Using select with a range
    dataset = load_dataset(
        data_cfg.dataset_name,
        split=f"train[:{data_cfg.dataset_subset_size}]",  # This syntax selects first n_samples
        cache_dir=cache_dir,
        trust_remote_code=True,
        num_proc=thread_count,
    )
    
    # Batch the data
    dataset = dataset.batch(data_cfg.buffer_refresh_size_rows, num_proc=thread_count)
    print(f"Time taken to initialize dataset: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(data_cfg.model_tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    for batch in dataset:
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            chunk_size = max(1, len(batch["text"]) // thread_count)
            futures = []

            for i in range(thread_count):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(batch["text"]))
                if start_idx >= len(batch["text"]):
                    break
                sub_texts = batch["text"][start_idx:end_idx]
                futures.append(
                    executor.submit(
                        tokenizer,
                        sub_texts,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=data_cfg.dataset_row_len,
                        padding="max_length",
                        return_tensors="pt",
                    )
                )

            partial_encodings = [f.result() for f in futures]

        combined = {}
        for key in partial_encodings[0].keys():
            combined[key] = torch.cat([p[key] for p in partial_encodings], dim=0)

        batch_tokens = combined["input_ids"]

        padding_token = tokenizer.pad_token_id
        total_tokens = batch_tokens.numel()
        padded_tokens = (batch_tokens == padding_token).sum().item()
        padding_percentage = (padded_tokens / total_tokens) * 100
        print(f"Padding percentage: {padding_percentage:.2f}%")
        
        assert isinstance(batch_tokens, torch.Tensor), f"Batch tokens are not a tensor: {type(batch_tokens)}"
        assert batch_tokens.shape == (data_cfg.buffer_refresh_size_rows, data_cfg.dataset_row_len), (
            f"Batch tokens shape is {batch_tokens.shape}, expected "
            f"{(data_cfg.buffer_refresh_size_rows, data_cfg.dataset_row_len)}"
        )
        
        batch_tokens = einops.rearrange(
            batch_tokens,
            "row (x seq_len) -> (row x) seq_len",
            seq_len=data_cfg.seq_len,
            x=data_cfg.seqs_per_dataset_row,
            row=data_cfg.buffer_refresh_size_rows
        )
        print(f"Time taken to stream: {time.time() - start_time:.2f} seconds")
        yield batch_tokens
        start_time = time.time()

class Buffer:
    def __init__(self, eval: bool = False, use_multiprocessing: bool = True):
        print("Initializing buffer...")
        self.use_multiprocessing = use_multiprocessing
        self.token_stream = stream_training_chunks(
            data_cfg,
            cache_dir,
            eval=eval,
        )
        self.buffer = torch.zeros(
            (data_cfg.buffer_size_tokens, data_cfg.act_size),
            dtype=DTYPES[data_cfg.enc_dtype],
            device=data_cfg.buffer_device,
        )

        if self.use_multiprocessing:
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
        else:
            device = data_cfg.device
            local_dtype = DTYPES[data_cfg.enc_dtype]
            self.local_model = transformer_lens.HookedTransformer.from_pretrained(
                data_cfg.model_name, device=device
            ).to(local_dtype)

        self.chunk_index = 0
        self.cache_path = None
        if data_cfg.caching:
            existing_cache = find_existing_cache_folder(data_cfg, activations_dir)
            if existing_cache is not None:
                print(f"Found existing cache: {existing_cache}")
                self.cache_path = existing_cache
            else:
                self.cache_path = create_new_cache_folder(data_cfg, activations_dir)
                print(f"Created new cache folder: {self.cache_path}")

        self.pointer = 0
        self.first = True
        self.refresh()

    @torch.no_grad()
    def refresh(self):
        print("Refreshing buffer...")
        start_time = time.time()

        all_acts = None
        if data_cfg.caching and self.cache_path is not None:
            chunk_path = self.cache_path / f"chunk_{self.chunk_index}.pt"
            if chunk_path.exists():
                print(f"Loading from cache chunk {self.chunk_index}")
                all_acts = torch.load(chunk_path, map_location=data_cfg.buffer_device).to(DTYPES[data_cfg.enc_dtype])
            
        start = time.time()
        if all_acts is None:
            seqs = next(self.token_stream)
            assert seqs is not None, "No more data available from token_stream."
            assert seqs.shape[0] == data_cfg.buffer_refresh_size_seqs, f"Expected {data_cfg.buffer_refresh_size_seqs} sequences, got {seqs.shape[0]}"

            if self.first:
                seqs_next = next(self.token_stream)
                seqs = torch.cat([seqs, seqs_next], dim=0)
                self.first = False

            if self.use_multiprocessing:
                all_acts = self.run_inference_multi(seqs)
            else:
                all_acts = self.run_inference_mono(seqs)

            if data_cfg.caching and self.cache_path is not None:
                torch.save(all_acts.to(DTYPES[data_cfg.save_dtype]), self.cache_path / f"chunk_{self.chunk_index}.pt")

        print(f"Time taken to run inference: {time.time() - start:.2f} seconds")

        start = time.time()
        acts_count = all_acts.shape[0]
        assert acts_count <= self.buffer.shape[0], "Buffer overflow"
        self.buffer[:acts_count] = all_acts
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0])]

        self.pointer = 0
        self.chunk_index += 1
        print(f"Time taken to refresh buffer: {time.time() - start:.2f} seconds")

        print(f"Buffer refreshed in {time.time() - start_time:.2f} seconds.")

    @torch.no_grad()
    def run_inference_multi(self, seqs: torch.Tensor) -> torch.Tensor:
        print("Running inference with multiprocessing...")
        total_seqs = seqs.shape[0]
        num_devices = data_cfg.llm_device_count
        chunk_size = total_seqs // num_devices
        seqs_list = []

        for i in range(num_devices):
            start_idx = i * chunk_size
            end_idx = total_seqs if i == (num_devices - 1) else (start_idx + chunk_size)
            seqs_list.append(seqs[start_idx:end_idx])

        for i in range(num_devices):
            self.tasks.put(seqs_list[i])

        results = []
        for _ in range(num_devices):
            result = self.results.get()
            if isinstance(result, Exception):
                raise result
            results.append(result.to("cuda:0"))

        print("Combining partial results...")
        return torch.cat(results, dim=0)

    @torch.no_grad()
    def run_inference_mono(self, seqs: torch.Tensor) -> torch.Tensor:
        print("Running inference in single process...")
        seqs[:, 0] = self.local_model.tokenizer.bos_token_id

        batch_size_seqs = data_cfg.model_batch_size_seqs
        all_acts_list = []
        seqs = seqs.to(data_cfg.device)

        local_dtype = DTYPES[data_cfg.enc_dtype]
        with torch.autocast("cuda", local_dtype):
            for start_idx in range(0, seqs.shape[0], batch_size_seqs):
                sub_chunk = seqs[start_idx : start_idx + batch_size_seqs]
                _, cache = self.local_model.run_with_cache(
                    sub_chunk,
                    stop_at_layer=data_cfg.layer + 1,
                    names_filter=data_cfg.act_name,
                    return_cache_object=True,
                )
                acts = cache.cache_dict[data_cfg.act_name]
                acts = acts.reshape(acts.shape[0] * acts.shape[1], data_cfg.act_size)
                all_acts_list.append(acts.to(data_cfg.buffer_device))

        return torch.cat(all_acts_list, dim=0)

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
        if self.use_multiprocessing:
            for _ in self.workers:
                self.tasks.put(None)
            for worker in self.workers:
                worker.join()
            self.tasks.close()
            self.results.close()
