import enum
import logging

import torch
import torch.multiprocessing as mp
import transformer_lens

from mlsae.config import DEVICE_COUNT, DTYPES, data_cfg, train_cfg
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
            logging.info("CPU worker waiting for task")
            task = tasks.get()
            logging.info(f"CPU worker got task")
            if task is None:
                break
            task_type, task_data = task
            assert task_type == TaskType.TOKENS
            token_block_idx = task_data["token_block_idx"]
            # It is okay that we are not checking for stopiteration. We are
            # setting the max tokens based on knowledge of the dataset and even
            # if we are wrong, it's not a big deal that the exception gets
            # bubbled up.
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
    all_acts = []
    token_block = shared_memory["token_blocks"][task_data["token_block_idx"]]
    assert (
        token_block.shape[0] == data_cfg.act_block_size_seqs
    ), f"Expected {data_cfg.act_block_size_seqs} tokens, got {token_block.shape[0]}"

    with torch.no_grad():
        with torch.autocast("cuda", dtype=DTYPES[data_cfg.sae_dtype]):
            for start in range(0, token_block.shape[0], data_cfg.llm_batch_size_seqs):
                subblock = token_block[start : start + data_cfg.llm_batch_size_seqs].to(
                    device
                )
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
    return SparseAdam(model.get_param_groups())


def resample_dead_features(optimizer: SparseAdam, model: DeepSAE, idx: torch.Tensor):
    """
    Zeros out the Adam moments for re-initialized rows/columns in the sparse encoder's
    weight/bias and the first decoder layer's weight. This avoids using stale moment
    estimates for brand-new features.
    """
    if not model.should_resample_sparse_features(idx):
        return

    logging.info(f"Resampling dead features model {model.name}")

    model.resample_sparse_features(idx)

    enc_layer = model.sparse_encoder_block[0]  # Linear for the sparse encoder
    dec_layer = model.decoder_blocks[0]  # First Linear in the decoder

    # We do not reset the step counter, but we reset the momentum
    with torch.no_grad():
        for group in optimizer.param_groups:
            for p in group["params"]:
                # Safely retrieve state
                state = optimizer.state.get(p, {})
                exp_avg = state.get("exp_avg", None)
                exp_avg_sq = state.get("exp_avg_sq", None)
                if exp_avg is None or exp_avg_sq is None:
                    continue  # Some params may not have momentum buffers yet

                if p is enc_layer.weight:
                    # shape: (sparse_dim, in_dim)
                    exp_avg[idx, :] = 0
                    exp_avg_sq[idx, :] = 0

                elif p is enc_layer.bias:
                    # shape: (sparse_dim)
                    exp_avg[idx] = 0
                    exp_avg_sq[idx] = 0

                elif p is dec_layer.weight:
                    # shape: (out_dim, sparse_dim)
                    exp_avg[:, idx] = 0
                    exp_avg_sq[:, idx] = 0


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
    optimizer.copy_tensors_(shared_memory["optimizers"][model_idx])
    act_freq_history = shared_memory["act_freq_history"][model_idx].to(device)
    n_iter = shared_memory["n_iter"][model_idx].to(device)
    act_block = shared_memory["act_blocks"][act_block_idx]

    assert (
        act_block.shape[0] == data_cfg.act_block_size_tokens
    ), f"Expected {data_cfg.act_block_size_tokens} tokens, got {act_block.shape[0]}"

    metrics_list = []

    # This loop is fine, as we assert that the act block size is correct above,
    # and we set the act block size as a multiple of the SAE batch size
    for start in range(0, act_block.shape[0], data_cfg.sae_batch_size_tokens):
        acts = act_block[start : start + data_cfg.sae_batch_size_tokens].to(device)

        loss, l2, mse_loss, feature_acts, _ = model(acts, step=n_iter)
        loss.backward()
        model.process_gradients()
        optimizer.step()
        optimizer.zero_grad()

        if start == 0:
            logging.info(
                f"Start: Device {device}, Model {model_idx}, Loss: {loss.item()}, Optimizer step: {optimizer.get_step().item()}"
            )

        act_freq_batch = (feature_acts > 0).float().mean(dim=0)
        act_freq_history += act_freq_batch

        # store step metrics
        metrics = {
            "loss": loss.item(),
            "l2": l2.item(),
            "mse_loss": mse_loss.item(),
            "weight_decay_penalty": model.get_weight_decay_penalty(),
            "act_decay": model.get_act_decay(n_iter),
        }

        if (n_iter + 1) % train_cfg.measure_dead_over_n_batches == 0:
            dead_features = act_freq_history == 0

            if (n_iter + 1) % train_cfg.resample_dead_every_n_batches == 0:
                logging.info(
                    f"Possibly resampling dead features device {device} model {model.name}"
                )
                try:
                    resample_dead_features(optimizer, model, dead_features)
                except Exception as e:
                    logging.error(
                        f"Failed to resample dead features device {device} model {model.name}: {e}"
                    )

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
    shared_memory["optimizers"][model_idx].copy_tensors_(optimizer)
    shared_memory["models"][model_idx].copy_tensors_(model)

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
