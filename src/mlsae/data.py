import logging
import os
from pathlib import Path
from typing import Dict, List

import aiohttp
import einops
import numpy as np
import torch
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from transformers import AutoTokenizer

from mlsae.config import data_cfg

this_dir = Path(__file__).parent

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
        # This is not a bug. We are only loading datasets that are
        # structured in this way.
        assert isinstance(text, list), f"Expected list, got {type(text)}"
        assert isinstance(text[0], list), f"Expected list of lists, got {type(text[0])}"
        assert isinstance(
            text[0][0], str
        ), f"Expected list of lists of strings, got {type(text[0][0])}"
        # Concatenate it all into an enormous string, separated by eos_tokens
        # This double loop looks incorrect, but we are actually getting a list
        # of lists of strings from text, so this is required and correct
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


def stream_training_chunks(dataset_batch_size_entries: int | None = None, act_block_size_seqs: int | None = None, seed: int | None = None):
    CLIENT_TIMEOUT_SECONDS = 60 * 60 * 2
    storage_options = {
        "client_kwargs": {
            "timeout": aiohttp.ClientTimeout(total=CLIENT_TIMEOUT_SECONDS),
        }
    }
    dataset_iter = load_dataset(
        data_cfg.dataset_name,
        "en",
        split="train",
        streaming=True,
        cache_dir=cache_dir,
        trust_remote_code=True,
        storage_options=storage_options,
    )

    dataset_iter = keep_single_column(dataset_iter, data_cfg.dataset_column_name)
    dataset_batch_size = dataset_batch_size_entries or data_cfg.dataset_batch_size_entries
    dataset_iter = dataset_iter.batch(dataset_batch_size)

    if seed is not None:
        dataset_iter = dataset_iter.shuffle(seed=seed, buffer_size=dataset_batch_size)

    dataset_iter = tokenize_and_concatenate(
        dataset_iter,
        AutoTokenizer.from_pretrained(data_cfg.tokenizer_name),
        streaming=True,
        max_length=data_cfg.seq_len,
        add_bos_token=True,
        column_name=data_cfg.dataset_column_name,
    )

    dataset_iter = dataset_iter.batch(act_block_size_seqs or data_cfg.act_block_size_seqs)

    for batch in dataset_iter:
        yield batch["tokens"].to(dtype=torch.int32, device="cpu")
