import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import uuid

import einops
import torch
import torch.multiprocessing as mp
import transformer_lens
from datasets import load_dataset

# Add a tokenizer import for the fallback path
from transformers import AutoTokenizer

this_dir = Path(__file__).parent


@dataclass
class DataConfig:
    seed: int = 49
    buffer_batch_size_tokens: int = 131072
    buffer_size_buffer_batch_size_mult: int = 256
    seq_len: int = 64
    model_batch_size_seqs: int = 512
    dataset_row_len: int = 1024
    enc_dtype: str = "fp32"
    save_dtype: str = "bf16"
    model_name: str = "pythia-31m"
    site: str = "resid_pre"
    layer: int = 6
    act_size: int = 768
    device: str = "cuda:0"
    dataset_name: str = "EleutherAI/pile"
    data_is_tokenized: bool = False  # New field
    llm_device_count: int = 4

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
    def buffer_refresh_size_tokens(self) -> int:
        return self.buffer_refresh_size_seqs * self.seq_len

    @property
    def buffer_refresh_size_rows(self) -> int:
        return self.buffer_refresh_size_seqs // self.seqs_per_dataset_row

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
                    acts = acts.reshape(
                        acts.shape[0] * acts.shape[1], local_data_cfg.act_size
                    )
                    all_acts.append(acts.cpu())

            results.put(torch.cat(all_acts, dim=0).to("cpu"))

    except Exception as e:
        results.put(e)
        raise


def stream_training_chunks(data_cfg, cache_dir, buffer_size: int, eval: bool = False):
    """
    Pulls data from the specified dataset. 
    If data_is_tokenized is False, applies a tokenizer to raw text before yielding.
    If data_is_tokenized is True, uses row_batch["input_ids"] directly.
    """
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
    row_batch_iter = dataset_iter.batch(buffer_size // 2)

    # Instantiate a tokenizer if needed
    tokenizer = None
    if not data_cfg.data_is_tokenized:
        tokenizer = AutoTokenizer.from_pretrained(data_cfg.model_name)

    for row_batch in row_batch_iter:
        if data_cfg.data_is_tokenized:
            # data should already have ["input_ids"]
            token_tensors = torch.tensor(
                row_batch["input_ids"], dtype=torch.int32, device=data_cfg.device
            )
        else:
            # data is not tokenized, so tokenize raw text here
            # some versions of The Pile store text under row_batch["text"]
            tokenized = tokenizer(
                list(row_batch["text"]),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=data_cfg.dataset_row_len,
            )
            token_tensors = tokenized["input_ids"].to(data_cfg.device)

        yield token_tensors


class Buffer:
    def __init__(self, eval: bool = False, use_multiprocessing: bool = True):
        print("Initializing buffer...")
        self.use_multiprocessing = use_multiprocessing
        self.token_stream = stream_training_chunks(
            data_cfg,
            cache_dir,
            data_cfg.buffer_refresh_size_rows * 2,
            eval=eval,
        )
        self.buffer = torch.zeros(
            (data_cfg.buffer_size_tokens, data_cfg.act_size),
            dtype=DTYPES[data_cfg.enc_dtype],
            device="cpu",
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
        self.pointer = 0

        rows = next(self.token_stream)
        assert rows is not None, "No more data available from token_stream."

        if self.first:
            rows_next = next(self.token_stream)
            rows = torch.cat([rows, rows_next], dim=0)
            self.first = False

        all_acts = None
        if data_cfg.caching and self.cache_path is not None:
            chunk_path = self.cache_path / f"chunk_{self.chunk_index}.pt"
            if chunk_path.exists():
                print(f"Loading from cache chunk {self.chunk_index}")
                all_acts = torch.load(chunk_path, map_location="cpu").to(DTYPES[data_cfg.enc_dtype])
            
        if all_acts is None:
            print("Preprocessing rows...")
            seqs = einops.rearrange(
                rows,
                "row (seq token) -> (row seq) token",
                token=data_cfg.seq_len,
                seq=data_cfg.seqs_per_dataset_row,
            )

            if self.use_multiprocessing:
                all_acts = self.run_inference_multi(seqs)
            else:
                all_acts = self.run_inference_mono(seqs)

            if data_cfg.caching and self.cache_path is not None:
                torch.save(all_acts.to(DTYPES[data_cfg.save_dtype]), self.cache_path / f"chunk_{self.chunk_index}.pt")

        acts_count = all_acts.shape[0]
        assert acts_count <= self.buffer.shape[0], "Buffer overflow"
        self.buffer[:acts_count] = all_acts
        self.pointer = acts_count

        idx = torch.randperm(acts_count)
        self.buffer[:acts_count] = self.buffer[idx]
        self.pointer = 0
        self.chunk_index += 1

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
                all_acts_list.append(acts.cpu())

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
