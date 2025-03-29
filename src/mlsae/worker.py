import enum
import logging

from networkx import cut_size
import torch
import torch.multiprocessing as mp
import transformer_lens
import einops

from mlsae.config import DEVICE_COUNT, DTYPES, data_cfg, train_cfg
from mlsae.data import stream_training_chunks
from mlsae.model import DeepSAE
from mlsae.shared_memory import SharedMemory

import os

class TaskType(enum.Enum):
    TOKENS = 0
    ACTS = 1
    TRAIN = 2


def cpu_worker(tasks: mp.Queue, results: mp.Queue, shared_memory: SharedMemory):
    logging.info("Starting CPU worker")
    token_stream = stream_training_chunks()
    token_block_cache_idx = 0
    try:
        while True:
            logging.info("CPU worker waiting for task")
            task = tasks.get()
            logging.info(f"CPU worker got task")
            if task is None:
                break
            task_type, task_data = task
            assert task_type == TaskType.TOKENS
            token_block_idx = task_data["token_block_idx"]

            token_block_path = f"token_block_{token_block_cache_idx}.pt"
            cache_hit = os.path.exists(token_block_path)

            if cache_hit:
                token_block = torch.load(token_block_path, map_location="cpu")
                logging.info(f"Loaded token block {token_block_cache_idx} from {token_block_path}")
            else:
                token_block = next(token_stream)

            assert (
                token_block.shape == (data_cfg.act_block_size_seqs, data_cfg.seq_len)
            ), f"Expected {(data_cfg.act_block_size_seqs, data_cfg.seq_len)}, got {token_block.shape}"
            shared_memory["token_blocks"][token_block_idx].copy_(token_block)
            results.put(
                (
                    TaskType.TOKENS,
                    {"token_block_idx": token_block_idx},
                )
            )

            if not cache_hit:
                logging.info(f"Saving token block {token_block_cache_idx} to {token_block_path}")
                torch.save(token_block, token_block_path)
                logging.info(f"Successfully saved token block {token_block_cache_idx}")

            token_block_cache_idx += 1
    except Exception as e:
        results.put(e)
        raise


def gpu_worker(
    device_id: int, tasks: mp.Queue, results: mp.Queue, shared_memory: SharedMemory
):
    device = f"cuda:{device_id}"
    logging.info(f"Starting worker on device {device}")
    local_llm = None
    assert (
        device_id >= 0 and device_id < DEVICE_COUNT
    ), f"Device id {device_id} is out of range"

    try:
        while True:
            logging.info(f"GPU worker {device_id} waiting for task")
            task = tasks.get()
            logging.info(f"GPU worker {device_id} got task")
            if task is None:
                break
            task_type, task_data = task
            if task_type == TaskType.ACTS:
                if local_llm is None:
                    local_llm = transformer_lens.HookedTransformer.from_pretrained(
                        data_cfg.model_name, device="cpu"
                    ).to(DTYPES[data_cfg.sae_dtype])
                task_generate(results, device, local_llm, task_data, shared_memory)
            elif task_type == TaskType.TRAIN:
                task_train(results, device, task_data, shared_memory)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
    except Exception as e:
        results.put(e)
        raise


def generate_acts_from_tokens(
    llm: transformer_lens.HookedTransformer, tokens: torch.Tensor, device: str
):
    all_acts = []

    with torch.no_grad():
        with torch.autocast("cuda", dtype=DTYPES[data_cfg.sae_dtype]):
            for start in range(0, tokens.shape[0], data_cfg.llm_batch_size_seqs):
                subblock = tokens[start : start + data_cfg.llm_batch_size_seqs].to(
                    device
                )
                _, cache = llm.run_with_cache(
                    subblock,
                    stop_at_layer=data_cfg.layer + 1,
                    names_filter=data_cfg.act_name,
                    return_cache_object=True,
                )
                acts = cache.cache_dict[data_cfg.act_name]
                assert acts.shape[1:3] == (data_cfg.seq_len, data_cfg.act_size_full), f"Expected shape ({data_cfg.seq_len}, {data_cfg.act_size_full}), got {acts.shape[1:3]}"
                batch_size_seqs = acts.shape[0]
                
                alt_acts = einops.rearrange(acts, "batch seq (act_chunk act_size) -> batch seq act_chunk act_size", act_size=data_cfg.act_size, act_chunk=data_cfg.act_size_full_multiple)
                alt_acts = einops.rearrange(alt_acts, "batch seq act_chunk act_size -> (batch seq act_chunk) act_size", act_size=data_cfg.act_size)

                acts = einops.rearrange(acts, "batch seq (act_chunk act_size) -> (batch seq act_chunk) act_size", act_size=data_cfg.act_size)

                assert torch.allclose(acts, alt_acts)

                # We chunk up the original llm activations and treat these
                # chunks as individual SAE inputs for quicker experimentation
                assert acts.shape == (batch_size_seqs * data_cfg.seq_len * data_cfg.act_size_full_multiple, data_cfg.act_size)
                all_acts.append(acts)
        
    

    result = torch.cat(all_acts, dim=0)
    assert result.shape == (data_cfg.act_block_size_seqs * data_cfg.seq_len * data_cfg.act_size_full_multiple, data_cfg.act_size)
    return result


def task_generate(
    results: mp.Queue,
    device: str,
    local_llm: transformer_lens.HookedTransformer,
    task_data: dict,
    shared_memory: SharedMemory,
):
    current_mem = torch.cuda.memory_allocated(device) / 1024**3
    logging.info(
        f"Starting activation generation device {device}. Current GPU memory: {current_mem:.2f}GB"
    )

    try:
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    except Exception as e:
        logging.error(f"Failed to reset peak memory stats for device {device}: {e}")

    # It's not ideal that we are moving the llm to and from CPU every time we
    # get this task, but it is acceptable so we can have a GPU work on both act
    # generation and training
    local_llm.to(device)
    token_block = shared_memory["token_blocks"][task_data["token_block_idx"]]
    assert (
        token_block.shape[0] == data_cfg.act_block_size_seqs
    ), f"Expected {data_cfg.act_block_size_seqs} tokens, got {token_block.shape[0]}"
    acts_block = generate_acts_from_tokens(local_llm, token_block, device)
    shared_memory["act_blocks"][task_data["act_block_idx"]].copy_(acts_block)

    results.put(
        (
            TaskType.ACTS,
            {
                "act_block_idx": task_data["act_block_idx"],
                "token_block_idx": task_data["token_block_idx"],
            },
        )
    )

    local_llm.to("cpu")

    try:
        peak_mem = torch.cuda.max_memory_allocated(torch.device(device)) / 1024**3
        logging.info(
            f"Finished activation generation device {device}. Peak GPU memory: {peak_mem:.2f}GB"
        )
    except Exception as e:
        logging.error(f"Failed to get peak memory stats for device {device}: {e}")


def init_optimizer(model: DeepSAE):
    # Use the model's optimizer configuration if available
    optimizer_type = getattr(model, "optimizer_type", "sparse_adam")
    optimizer_config = getattr(model, "optimizer_config", {})
    
    if optimizer_type == "mixed_muon":
        from mlsae.optimizer.mixed_muon import MixedMuon
        return MixedMuon(model.parameters(), **optimizer_config)
    elif optimizer_type == "sparse_adam":
        from mlsae.optimizer.sparse_adam import SparseAdam
        return SparseAdam(model.parameters(), **optimizer_config)
    elif optimizer_type == "SGD":
        from torch.optim import SGD
        return SGD(model.parameters(), **optimizer_config)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def preprocess_acts(acts: torch.Tensor):
    acts -= acts.mean(dim=1, keepdim=True)
    acts /= acts.norm(dim=1, keepdim=True)
    return acts


def task_train(
    results: mp.Queue, device: str, task_data: dict, shared_memory: SharedMemory
):
    current_mem = torch.cuda.memory_allocated(device) / 1024**3
    logging.info(
        f"Starting training device {device}. Current GPU memory: {current_mem:.2f}GB"
    )

    try:
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    except Exception as e:
        logging.error(f"Failed to reset peak memory stats for device {device}: {e}")

    model_idx = task_data["model_idx"]
    act_block_idx = task_data["act_block_idx"]

    model = shared_memory["models"][model_idx].clone()
    model.to(device)
    optimizer = init_optimizer(model)
    optimizer.copy_state_from(shared_memory["optimizers"][model_idx])
    act_freq_history = shared_memory["act_freq_history"][model_idx].to(device)
    n_iter = shared_memory["n_iter"][model_idx].to(device)
    act_block = shared_memory["act_blocks"][act_block_idx]

    assert (
        act_block.shape == (data_cfg.act_block_size_entries, data_cfg.act_size)
    ), f"Expected shape ({data_cfg.act_block_size_entries}, {data_cfg.act_size}), got {act_block.shape}"

    metrics_list = []

    # This loop is fine, as we assert that the act block size is correct above,
    # and we set the act block size as a multiple of the SAE batch size
    for start in range(0, act_block.shape[0], data_cfg.sae_batch_size_entries):
        acts = act_block[start : start + data_cfg.sae_batch_size_entries].to(device)

        acts = preprocess_acts(acts)

        result = model.optimize(acts, optimizer, iteration=n_iter)

        loss = result["loss"]
        feature_acts = result["feature_acts"]

        if start == 0:
            logging.info(
                f"Start: Device {device}, Model {model_idx}, Loss: {loss.item()}"
            )

        act_freq_batch = (feature_acts != 0).float().mean(dim=0)
        act_freq_history += act_freq_batch
        
        # Calculate the average number of nonzero features per example in the batch
        avg_nonzero_features = (feature_acts != 0).float().sum(dim=1).mean().item()

        # store step metrics
        baseline_mse = get_baseline_mse(acts)
        metrics = {
            "loss": loss.item(),
            "mse": result["mse_loss"].item(),
            "mse_baseline": baseline_mse.item(),
            "mse_normalized": (result["mse_loss"] / baseline_mse).item(),
            "avg_nonzero_features": avg_nonzero_features,
        }

        # Log all values in result that are float, int, or tensor of size 1
        for key, value in result.items():
            # Skip metrics we've already logged
            if key in metrics:
                continue
                
            # Check if it's a scalar value we can log
            if isinstance(value, (float, int)):
                metrics[key] = value
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                metrics[key] = value.item()

        if (n_iter + 1) % train_cfg.measure_dead_over_n_batches == 0:
            dead_features = act_freq_history == 0
            assert dead_features.shape == (
                model.sparse_dim,
            ), f"Expected {model.sparse_dim} dead features, got {dead_features.shape}"

            metrics["dead_features"] = dead_features.float().sum().item()
            act_freq_history = torch.zeros(
                model.sparse_dim,
                dtype=torch.float,
                device=device,
            )

        metrics_list.append(metrics)
        n_iter += 1

    logging.info(f"End: Device {device}, Model {model_idx}, Loss: {loss.item()}")

    del acts

    # update shared memory
    shared_memory["act_freq_history"][model_idx].copy_(act_freq_history)
    shared_memory["n_iter"][model_idx].copy_(n_iter)
    shared_memory["models"][model_idx].copy_tensors_(model)
    shared_memory["optimizers"][model_idx].copy_state_from(optimizer)


    del model

    try:
        peak_mem = torch.cuda.max_memory_allocated(torch.device(device)) / 1024**3
        logging.info(
            f"Finished training device {device}. Peak GPU memory: {peak_mem:.2f}GB"
        )
    except Exception as e:
        logging.error(f"Failed to get peak memory stats for device {device}: {e}")

    result_data = {
        "metrics": metrics_list,
        "act_block_idx": act_block_idx,
        "model_idx": model_idx,
    }
    results.put((TaskType.TRAIN, result_data))


def get_baseline_mse(acts: torch.Tensor):
    """
    Get the baseline MSE loss for the given activations.
    """
    return (acts - acts.mean(dim=0)).pow(2).mean()
