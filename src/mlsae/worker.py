import enum
import logging

import torch
import torch.multiprocessing as mp
import transformer_lens

from mlsae.config import DTYPES, data_cfg, train_cfg
from mlsae.data import stream_training_chunks
from mlsae.model import DeepSAE, SparseAdam
from mlsae.shared_memory import SharedMemory


class TaskType(enum.Enum):
    TOKENS = 0
    ACTS = 1
    TRAIN = 2


def cpu_worker(tasks: mp.Queue, results: mp.Queue, shared_memory: SharedMemory):
    logging.info("Starting CPU worker")
    token_stream = stream_training_chunks()
    try:
        while True:
            task = tasks.get()
            if task is None:
                break
            task_type, task_data = task
            assert task_type == TaskType.TOKENS
            token_block_idx = task_data["token_block_idx"]
            token_block = next(token_stream)
            shared_memory.token_blocks[token_block_idx].copy_(token_block)
    except Exception as e:
        results.put(e)
        raise


def gpu_worker(
    device_id: int, tasks: mp.Queue, results: mp.Queue, shared_memory: SharedMemory
):
    device = f"cuda:{device_id}"
    logging.info(f"Starting worker on device {device}")
    local_llm = None

    try:
        while True:
            task = tasks.get()
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
                raise ValueError(f"Unknown task type: {task}")
    except Exception as e:
        results.put(e)
        raise


def task_generate(
    results: mp.Queue,
    device: str,
    local_llm: transformer_lens.HookedTransformer,
    task_data: dict,
    shared_memory: SharedMemory,
):
    local_llm.to(device)
    all_acts = []
    token_block = shared_memory.token_blocks[task_data["token_block_idx"]]
    assert (
        token_block.shape[0] == data_cfg.act_block_size_seqs
    ), f"Expected {data_cfg.act_block_size_seqs} tokens, got {token_block.shape[0]}"

    with torch.no_grad():
        with torch.autocast("cuda", DTYPES[data_cfg.sae_dtype]):
            for start in range(0, token_block.shape[0], data_cfg.llm_batch_size_seqs):
                subblock = token_block[start : start + data_cfg.llm_batch_size_seqs]
                _, cache = local_llm.run_with_cache(
                    subblock,
                    stop_at_layer=data_cfg.layer + 1,
                    names_filter=data_cfg.act_name,
                    return_cache_object=True,
                )
                acts = cache.cache_dict[data_cfg.act_name]
                assert (
                    acts.shape[-1] == data_cfg.act_size
                ), f"Expected {data_cfg.act_size} act size, got {acts.shape[-1]}"
                acts = acts.reshape(acts.shape[0] * acts.shape[1], data_cfg.act_size)
                all_acts.append(acts)

    acts_block = torch.cat(all_acts, dim=0)
    shared_memory.act_blocks[task_data["act_block_idx"]].copy_(acts_block)

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


def init_optimizer(model: DeepSAE, model_idx: int):
    weight_decay = train_cfg.architectures[model_idx]["weight_decay"]
    lr = train_cfg.architectures[model_idx]["lr"]
    return SparseAdam(model.get_param_groups(weight_decay=weight_decay), lr=lr)


def task_train(
    results: mp.Queue, device: str, task_data: dict, shared_memory: SharedMemory
):
    model_idx = task_data["model_idx"]
    act_block_idx = task_data["act_block_idx"]

    model = shared_memory.models[model_idx].to(device)
    optimizer = init_optimizer(model, model_idx)
    optimizer.copy_tensors_(shared_memory.optimizers[model_idx])
    act_freq_history = shared_memory.act_freq_history[model_idx].to(device)
    n_iter = shared_memory.n_iter[model_idx].to(device)
    act_block = shared_memory.act_blocks[act_block_idx].to(device)

    assert (
        act_block.shape[0] == data_cfg.act_block_size_tokens
    ), f"Expected {data_cfg.act_block_size_tokens} tokens, got {act_block.shape[0]}"

    metrics_list = []

    for start in range(0, act_block.shape[0], data_cfg.sae_batch_size_tokens):
        acts = act_block[start : start + data_cfg.sae_batch_size_tokens]

        loss, l2_loss, feature_acts, _ = model(acts)
        loss.backward()
        model.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        optimizer.zero_grad()

        act_freq_batch = (feature_acts > 0).float().mean(dim=0)
        act_freq_history += act_freq_batch

        # store step metrics
        metrics = {
            "loss": loss.item(),
            "l2_loss": l2_loss.item(),
        }

        if (n_iter + 1) % train_cfg.measure_dead_over_n_batches == 0:
            dead_features = act_freq_history == 0

            if (n_iter + 1) % train_cfg.resample_dead_every_n_batches == 0:
                model.resample_sparse_features(dead_features)

            metrics["dead_features"] = dead_features.float().sum().item()
            act_freq_history = torch.zeros(
                model.sparse_dim,
                dtype=torch.float,
                device=device,
            )

        metrics_list.append(metrics)
        n_iter += 1

    # update shared memory
    shared_memory.act_freq_history[model_idx].copy_(act_freq_history)
    shared_memory.n_iter[model_idx].copy_(n_iter)
    shared_memory.optimizers[model_idx].copy_tensors_(optimizer)
    shared_memory.models[model_idx].copy_tensors_(model)

    result_data = {
        "metrics": metrics_list,
        "act_block_idx": act_block_idx,
        "model_idx": model_idx,
    }
    results.put((TaskType.TRAIN, result_data))
