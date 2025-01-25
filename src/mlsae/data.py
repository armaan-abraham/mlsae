import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import einops
import numpy as np
import torch
import torch.multiprocessing as mp
import transformer_lens
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from transformers import AutoTokenizer

from mlsae.config import DataConfig, data_cfg, DTYPES
from mlsae.worker import TaskType

this_dir = Path(__file__).parent


DEVICE_COUNT = torch.cuda.device_count()

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


def keep_single_column(dataset: Dataset, col_name: str):
    """
    Acts on a HuggingFace dataset to delete all columns apart from a single column name - useful when we want to tokenize and mix together different strings
    """
    for key in dataset.features:
        if key != col_name:
            dataset = dataset.remove_columns(key)
    return dataset


# This is a modified version of the tokenize_and_concatenate function from transformer_lens.utils
def tokenize_and_concatenate(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    streaming: bool = False,
    max_length: int = 1024,
    column_name: str = "text",
    add_bos_token: bool = True,
    num_proc: int = 10,
) -> Dataset:
    """Helper function to tokenizer and concatenate a dataset of text. This converts the text to tokens, concatenates them (separated by EOS tokens) and then reshapes them into a 2D array of shape (____, sequence_length), dropping the last batch. Tokenizers are much faster if parallelised, so we chop the string into 20, feed it into the tokenizer, in parallel with padding, then remove padding at the end.

    This tokenization is useful for training language models, as it allows us to efficiently train on a large corpus of text of varying lengths (without, eg, a lot of truncation or padding). Further, for models with absolute positional encodings, this avoids privileging early tokens (eg, news articles often begin with CNN, and models may learn to use early positional encodings to predict these)

    Args:
        dataset (Dataset): The dataset to tokenize, assumed to be a HuggingFace text dataset.
        tokenizer (AutoTokenizer): The tokenizer. Assumed to have a bos_token_id and an eos_token_id.
        streaming (bool, optional): Whether the dataset is being streamed. If True, avoids using parallelism. Defaults to False.
        max_length (int, optional): The length of the context window of the sequence. Defaults to 1024.
        column_name (str, optional): The name of the text column in the dataset. Defaults to 'text'.
        add_bos_token (bool, optional): . Defaults to True.

    Returns:
        Dataset: Returns the tokenized dataset, as a dataset of tensors, with a single column called "tokens"
    """
    if tokenizer.pad_token is None:
        # We add a padding token, purely to implement the tokenizer. This will be removed before inputting tokens to the model, so we do not need to increment d_vocab in the model.
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    # Define the length to chop things up into - leaving space for a bos_token if required
    if add_bos_token:
        seq_len = max_length - 1
    else:
        seq_len = max_length

    logging.info(f"Tokenize and concatenate called")

    def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        text = examples[column_name]
        # Concatenate it all into an enormous string, separated by eos_tokens
        full_text = tokenizer.eos_token.join(
            [tokenizer.eos_token.join(sub_text) for sub_text in text]
        )
        logging.info(f"Full text length: {len(full_text)}")

        # Handle the case when full_text is empty
        if not full_text.strip():
            return {"tokens": np.array([], dtype=np.int64)}

        # Divide into 20 chunks of ~ equal length
        num_chunks = 20
        chunk_length = (len(full_text) - 1) // num_chunks + 1
        chunks = [
            full_text[i * chunk_length : (i + 1) * chunk_length]
            for i in range(num_chunks)
        ]
        # Tokenize the chunks in parallel. Uses NumPy because HuggingFace map doesn't want tensors returned
        tokens = tokenizer(chunks, return_tensors="np", padding=True)[
            "input_ids"
        ].flatten()
        # Drop padding tokens
        tokens = tokens[tokens != tokenizer.pad_token_id]
        num_tokens = len(tokens)
        logging.info(f"Num tokens: {num_tokens}")

        # Handle cases where num_tokens is less than seq_len
        if num_tokens < seq_len:
            num_batches = 1
            # Pad tokens if necessary
            tokens = tokens[:seq_len]
            if len(tokens) < seq_len:
                padding_length = seq_len - len(tokens)
                padding = np.full(padding_length, tokenizer.pad_token_id)
                tokens = np.concatenate([tokens, padding], axis=0)
        else:
            num_batches = num_tokens // seq_len
            # Drop the final tokens if not enough to make a full sequence
            tokens = tokens[: seq_len * num_batches]

        tokens = einops.rearrange(
            tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len
        )
        if add_bos_token:
            prefix = np.full((num_batches, 1), tokenizer.bos_token_id)
            tokens = np.concatenate([prefix, tokens], axis=1)
        return {"tokens": tokens}

    kwargs = {
        "batched": True,
        "remove_columns": [column_name],
    }
    if not streaming:
        kwargs["num_proc"] = num_proc

    tokenized_dataset = dataset.map(
        tokenize_function,
        **kwargs,
    )
    return tokenized_dataset.with_format(type="torch")


def stream_training_chunks(data_cfg, cache_dir, tokenizer, eval: bool = False):
    dataset_iter = load_dataset(
        data_cfg.dataset_name,
        "en",
        split="train",
        streaming=True,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    dataset_iter = keep_single_column(dataset_iter, data_cfg.dataset_column_name)

    dataset_iter = dataset_iter.batch(data_cfg.dataset_batch_size_entries)

    dataset_iter = tokenize_and_concatenate(
        dataset_iter,
        tokenizer,
        streaming=True,
        max_length=data_cfg.seq_len,
        add_bos_token=True,
        column_name=data_cfg.dataset_column_name,
    )

    dataset_iter = dataset_iter.batch(data_cfg.buffer_size_tokens // data_cfg.seq_len)

    for batch in dataset_iter:
        yield batch["tokens"].to(dtype=torch.int32, device="cpu")


class StaticBuffer:
    def __init__(self):
        self.pointer = 0

    def populate(self, buffer: torch.Tensor):
        assert buffer.ndim == 2, f"Expected buffer to be 2D, got {buffer.ndim}D"
        assert (
            buffer.shape[1] == data_cfg.act_size
        ), f"Expected buffer to have act size {data_cfg.act_size}, got {buffer.shape[1]}"
        assert (
            buffer.shape[0] == data_cfg.buffer_size_tokens
        ), f"Expected buffer to have {data_cfg.buffer_size_tokens} tokens, got {buffer.shape[0]}"
        self.buffer = buffer
        self.pointer = 0

    def needs_refresh(self):
        return self.pointer >= self.buffer.shape[0]

    def to(self, device: str) -> None:
        self.buffer = self.buffer.to(device)

    @torch.no_grad()
    def next(self):
        assert not self.needs_refresh(), "Buffer is empty"
        out = self.buffer[
            self.pointer : self.pointer + data_cfg.buffer_batch_size_tokens
        ]
        self.pointer += data_cfg.buffer_batch_size_tokens
        return out

    def clone(self):
        new_buffer = StaticBuffer()
        new_buffer.buffer = self.buffer.clone()
        new_buffer.pointer = self.pointer
        return new_buffer


class Buffer:
    """
    Streams tokens and fills self.buffer with model activations.
    """

    def __init__(
        self,
        results_queue: mp.Queue,
        tasks_queue: mp.Queue,
        num_workers: int = DEVICE_COUNT,
        eval: bool = False,
    ):
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        logging.info("Initializing buffer...")
        self.token_stream = stream_training_chunks(
            data_cfg,
            cache_dir,
            self.tokenizer,
            eval=eval,
        )

        self.results_queue = results_queue
        self.tasks_queue = tasks_queue

        self._init_cache(eval=eval)

        # For cache
        self.chunk_index = 0

        # Create and store a StaticBuffer instance.
        self.static_buffer = StaticBuffer()

        # Prefetch-related members
        self._prefetch_thread = None
        self._prefetched_acts = None
        self._prefetch_lock = threading.Lock()

        # Load the very first buffer content
        self.refresh()

    def _init_cache(self, eval: bool = False):
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
        Refreshes the StaticBuffer with the next chunk. If a prefetched chunk is
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
        logging.info("Loading tokens")
        tokens = next(self.token_stream)
        logging.info(f"Loaded tokens of shape {tokens.shape}")
        assert tokens is not None, "No more data from token_stream."
        assert tokens.ndim == 2, f"Expected tokens to be 2D, got {tokens.ndim}D"
        assert (
            tokens.shape[1] == data_cfg.seq_len
        ), f"Expected tokens to have sequence length {data_cfg.seq_len}, got {tokens.shape[1]}"
        assert (
            tokens[:, 0] == self.tokenizer.bos_token_id
        ).all(), "First token is not BOS"

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

        # Populate our StaticBuffer
        self.static_buffer.populate(acts)

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
        total_seqs = seqs.shape[0]
        chunk_size = total_seqs // self.num_workers
        seqs_list = []

        for i in range(self.num_workers):
            start_idx = i * chunk_size
            end_idx = (
                total_seqs if i == (self.num_workers - 1) else (start_idx + chunk_size)
            )
            seqs_list.append(seqs[start_idx:end_idx])

        logging.info("Running inference with multiprocessing...")
        for i in range(self.num_workers):
            self.tasks_queue.put((TaskType.GENERATE, {"seq_chunk": seqs_list[i]}))

        results = []
        for _ in range(self.num_workers):
            result = self.results_queue.get()
            if isinstance(result, Exception):
                raise result
            results.append(result)

        logging.info("Combining partial results...")
        return torch.cat(results, dim=0)

    def __del__(self):
        logging.info("Cleaning up Buffer resources...")
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            self._prefetch_thread.join()
