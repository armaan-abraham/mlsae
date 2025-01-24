import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import einops
import torch
import torch.multiprocessing as mp
import transformer_lens
from datasets import load_dataset
from transformers import AutoTokenizer

this_dir = Path(__file__).parent


@dataclass
class DataConfig:
    seed: int = 49
    buffer_batch_size_tokens: int = 65536
    seq_len: int = 128
    model_batch_size_seqs: int = 512
    enc_dtype: str = "fp32"
    cache_dtype: str = "bf16"
    model_name: str = "gpt2-small"
    site: str = "resid_pre"
    layer: int = 9
    act_size: int = 768
    dataset_name: str = "Skylion007/openwebtext"
    dataset_column_name: str = "text"
    dataset_batch_size: int = 2048

    eval_data_seed: int = 59
    eval_batches: int = 500

    caching: bool = True
    # These are the fields that must match before reusing an existing cache
    cache_id_fields: list = field(
        default_factory=lambda: [
            # TODO: eval data seed
            "seed",
            "model_name",
            "layer",
            "site",
            "seq_len",
            "act_size",
            "enc_dtype",
            "dataset_name",
            "dataset_batch_size",
        ]
    )

    @property
    def act_name(self) -> str:
        return transformer_lens.utils.get_act_name(self.site, self.layer)

    @property
    def eval_tokens(self) -> int:
        return self.buffer_batch_size_tokens * self.eval_tokens_buffer_batch_size_mult


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

DEVICE_COUNT = torch.cuda.device_count()
logging.info(f"Using {DEVICE_COUNT} GPUs")

data_cfg = DataConfig()

# Create directories
this_dir = Path(__file__).parent
cache_dir = this_dir / "cache"
activations_dir = cache_dir / "activations"
data_dir = this_dir / "data"
model_dir = this_dir / "checkpoints"

for dir_path in [cache_dir, data_dir, model_dir, activations_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Set environment variables for caching
os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
os.environ["DATASETS_CACHE"] = str(cache_dir)

tokenizer = AutoTokenizer.from_pretrained(data_cfg.model_name)


def get_relevant_config_subset(cfg: DataConfig) -> dict:
    """
    Returns a dictionary of only those fields in cfg.cache_id_fields.
    Used to identify matching cache directories.
    """
    subset = {}
    for field_name in cfg.cache_id_fields:
        subset[field_name] = getattr(cfg, field_name)
    return subset


def find_existing_cache_folder(
    cfg: DataConfig, base_cache_dir: Path, eval=False
) -> Path:
    """
    Checks each subdirectory in base_cache_dir for a config.json that matches
    the relevant fields of cfg. Returns the path if found, else None.
    """
    desired_subset = get_relevant_config_subset(cfg)
    for subdir in base_cache_dir.iterdir():
        if not subdir.is_dir():
            continue
        config_path = subdir / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                stored_cfg_subset = json.load(f)
            if stored_cfg_subset.get("eval", False) == eval:
                stored_cfg_subset.pop("eval")
                if stored_cfg_subset == desired_subset:
                    return subdir
    return None


def create_new_cache_folder(cfg: DataConfig, base_cache_dir: Path, eval=False) -> Path:
    """
    Creates a new UUID-based folder under base_cache_dir,
    writes config.json with relevant fields, and returns path.
    """
    new_cache_folder = base_cache_dir / str(uuid.uuid4())
    new_cache_folder.mkdir(parents=True, exist_ok=True)

    relevant_cfg = get_relevant_config_subset(cfg)
    relevant_cfg["eval"] = eval
    with open(new_cache_folder / "config.json", "w") as f:
        json.dump(relevant_cfg, f, indent=2)

    return new_cache_folder


def worker(tasks: mp.Queue, results: mp.Queue, device_id: int):
    device = f"cuda:{device_id}"

    local_model = transformer_lens.HookedTransformer.from_pretrained(
        data_cfg.model_name, device="cpu"
    ).to(DTYPES[data_cfg.enc_dtype])

    try:
        while True:
            task = tasks.get()

            if task is None:
                break

            seq_chunk = task

            local_model.to(device)

            batch_size_seqs = data_cfg.model_batch_size_seqs
            all_acts = []
            seq_chunk = seq_chunk.to(device)

            with torch.autocast("cuda", DTYPES[data_cfg.enc_dtype]):
                for start in range(0, seq_chunk.shape[0], batch_size_seqs):
                    sub_chunk = seq_chunk[start : start + batch_size_seqs]
                    _, cache = local_model.run_with_cache(
                        sub_chunk,
                        stop_at_layer=data_cfg.layer + 1,
                        names_filter=data_cfg.act_name,
                        return_cache_object=True,
                    )
                    acts = cache.cache_dict[data_cfg.act_name]
                    assert (
                        acts.shape[-1] == data_cfg.act_size
                    ), f"Expected {data_cfg.act_size} act size, got {acts.shape[-1]}"
                    acts = acts.reshape(
                        acts.shape[0] * acts.shape[1], data_cfg.act_size
                    )
                    all_acts.append(acts)

            results.put(torch.cat(all_acts, dim=0).to("cpu"))

            local_model.to("cpu")

    except Exception as e:
        results.put(e)
        raise


def stream_training_chunks(data_cfg, cache_dir, eval: bool = False):
    dataset_iter = load_dataset(
        data_cfg.dataset_name,
        split="train",
        streaming=True,
        cache_dir=cache_dir,
    )
    dataset_iter = dataset_iter.shuffle(
        buffer_size=data_cfg.dataset_batch_size,
        seed=data_cfg.seed if not eval else data_cfg.eval_data_seed,
    )

    dataset_iter = dataset_iter.batch(data_cfg.dataset_batch_size)
    dataset_iter = transformer_lens.utils.tokenize_and_concatenate(
        dataset_iter,
        tokenizer,
        streaming=True,
        max_length=data_cfg.seq_len,
        add_bos_token=True,
        column_name=data_cfg.dataset_column_name,
    )

    for batch in dataset_iter:
        yield batch["tokens"].to(dtype=torch.int32, device="cpu")


class Buffer:
    """
    Streams tokens and fills self.buffer with model activations.
    """

    def __init__(self, eval: bool = False):
        logging.info("Initializing buffer...")
        self.token_stream = stream_training_chunks(
            data_cfg,
            cache_dir,
            eval=eval,
        )

        self._init_act_gen_workers()

        self._init_cache()

        self.pointer = 0

        # For cache
        self.chunk_index = 0

        # Prefetch-related members
        self._prefetch_thread = None
        self._prefetched_acts = None
        self._prefetch_lock = threading.Lock()

        # Load the very first buffer content
        self.refresh()

    def _init_cache(self):
        # Find or create cache folder if caching is enabled
        self.cache_path = None
        if data_cfg.caching:
            existing_cache = find_existing_cache_folder(
                data_cfg, activations_dir, eval=eval
            )
            if existing_cache is not None:
                logging.info(f"Found existing cache: {existing_cache}")
                self.cache_path = existing_cache
            else:
                self.cache_path = create_new_cache_folder(
                    data_cfg, activations_dir, eval=eval
                )
                logging.info(f"Created new cache folder: {self.cache_path}")

    def _init_act_gen_workers(self):
        mp.set_start_method("spawn", force=True)
        self.tasks = mp.Queue()
        self.results = mp.Queue()
        self.workers = []

        # Spin up child processes that each hold a model
        for i in range(DEVICE_COUNT):
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

    def _prefetch_chunk(self, next_index: int):
        """
        In a background thread, check if the given chunk is in the cache.
        If yes, load it into _prefetched_acts; if not, do nothing.
        """
        if not data_cfg.caching or self.cache_path is None:
            return

        chunk_path = self.cache_path / f"chunk_{next_index}.pt"
        if chunk_path.exists():
            # If it exists, load it
            logging.info(f"(Prefetch) Found chunk {next_index} in cache. Loading...")
            acts = torch.load(chunk_path, map_location="cpu").to(
                DTYPES[data_cfg.enc_dtype]
            )

            with self._prefetch_lock:
                # Store for main thread to pick up on next refresh()
                self._prefetched_acts = acts
        else:
            logging.info(
                f"(Prefetch) Chunk {next_index} not found in cache. Skipping prefetch."
            )

    @torch.no_grad()
    def refresh(self):
        """
        Refreshes the buffer with the next chunk. If a prefetched chunk is
        available, uses it. Otherwise, normal logic applies. Spawns a new thread
        to prefetch the next chunk from cache for potential future use.
        """
        # Wait for any existing prefetch thread to finish
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            self._prefetch_thread.join()

        # Check if there is a prefetched chunk available
        with self._prefetch_lock:
            if self._prefetched_acts is not None:
                acts = self._prefetched_acts
                self._prefetched_acts = None
            else:
                acts = None

        # We need to load the tokens regardless of caching, so if we get a cache
        # miss, our iter is up to date
        tokens = next(self.token_stream)
        assert tokens is not None, "No more data from token_stream."
        assert tokens.ndim == 2, f"Expected tokens to be 2D, got {tokens.ndim}D"
        assert (
            tokens.shape[1] == data_cfg.seq_len
        ), f"Expected tokens to have sequence length {data_cfg.seq_len}, got {tokens.shape[1]}"

        # If we did not have prefetched data, do the usual load/generate
        if acts is None:
            logging.info(f"Refreshing buffer for chunk {self.chunk_index}...")
            # Try to load from cache if caching
            if data_cfg.caching and self.cache_path is not None:
                chunk_path = self.cache_path / f"chunk_{self.chunk_index}.pt"
                if chunk_path.exists():
                    logging.info(f"Loading chunk {self.chunk_index} from cache...")
                    acts = torch.load(chunk_path, map_location="cpu").to(
                        DTYPES[data_cfg.enc_dtype]
                    )

            # If still None, generate from data
            if acts is None:
                acts = self.run_inference_multi(tokens)
                assert acts.ndim == 2, f"Expected acts to be 2D, got {acts.ndim}D"
                assert (
                    acts.shape[1] == data_cfg.act_size
                ), f"Expected acts to have act size {data_cfg.act_size}, got {acts.shape[1]}"
                assert (
                    acts.shape[0] > data_cfg.buffer_batch_size_tokens
                ), f"Expected acts to have at least {data_cfg.buffer_batch_size_tokens} tokens, got {acts.shape[0]}"
                logging.info(f"Generated {acts.shape[0]} tokens")

                # Save to cache if enabled
                if data_cfg.caching and self.cache_path is not None:
                    torch.save(
                        acts.to(DTYPES[data_cfg.cache_dtype]),
                        self.cache_path / f"chunk_{self.chunk_index}.pt",
                    )

        self.buffer = acts
        self.pointer = 0

        self.chunk_index += 1

        # Spawn a thread to prefetch next chunk from cache (if it exists)
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_chunk,
            args=(self.chunk_index,),  # next chunk index
        )
        self._prefetch_thread.start()

    @torch.no_grad()
    def run_inference_multi(self, seqs: torch.Tensor) -> torch.Tensor:
        """
        Distributes seqs across multiple processes (one per GPU). Collects results.
        """
        assert (
            hasattr(self, "workers") and self.workers is not None
        ), "Workers not initialized"
        total_seqs = seqs.shape[0]
        num_workers = len(self.workers)
        chunk_size = total_seqs // num_workers
        seqs_list = []

        for i in range(num_workers):
            start_idx = i * chunk_size
            end_idx = total_seqs if i == (num_workers - 1) else (start_idx + chunk_size)
            seqs_list.append(seqs[start_idx:end_idx])

        logging.info("Running inference with multiprocessing...")
        for i in range(num_workers):
            self.tasks.put(seqs_list[i])

        results = []
        for _ in range(num_workers):
            result = self.results.get()
            if isinstance(result, Exception):
                raise result
            results.append(result)

        logging.info("Combining partial results...")
        return torch.cat(results, dim=0)

    @torch.no_grad()
    def next(self):
        out = self.buffer[
            self.pointer : self.pointer + data_cfg.buffer_batch_size_tokens
        ]
        self.pointer += data_cfg.buffer_batch_size_tokens
        if self.pointer > self.buffer.shape[0]:
            self.refresh()
        return out

    def __del__(self):
        logging.info("Cleaning up Buffer resources...")
        if self.use_multiprocessing:
            # Send termination signal to workers
            for _ in self.workers:
                self.tasks.put(None)
            for worker in self.workers:
                worker.join()
            self.tasks.close()
            self.results.close()
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            self._prefetch_thread.join()
