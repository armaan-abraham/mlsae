import enum
import logging
import cProfile
import pstats
from io import StringIO

import torch
import torch.multiprocessing as mp
import transformer_lens

from mlsae.config import DTYPES, data_cfg, train_cfg
from line_profiler import profile


class TaskType(enum.Enum):
    TRAIN = 0
    GENERATE = 1
    TERMINATE = 2


def worker(device_id: int, tasks: mp.Queue, results: mp.Queue):
    device = f"cuda:{device_id}"
    logging.info(f"Starting worker on device {device}")

    local_llm = transformer_lens.HookedTransformer.from_pretrained(
        data_cfg.model_name, device="cpu"
    ).to(DTYPES[data_cfg.enc_dtype])

    profiler = cProfile.Profile()
    profiler.enable()

    try:
        while True:
            task = tasks.get()
            task_type, task_data = task
            if task_type == TaskType.TERMINATE:
                break
            elif task_type == TaskType.GENERATE:
                task_generate(results, device, local_llm, task_data)
            elif task_type == TaskType.TRAIN:
                task_train(results, device, task_data)
            else:
                raise ValueError(f"Unknown task type: {task}")
    except Exception as e:
        results.put(e)
        raise
    finally:
        profiler.disable()
        
        # Option 1: print stats in the child process log
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumtime")
        ps.print_stats()
        logging.info(f"Worker {device_id} profile:\n{s.getvalue()}")

        # Option 2: dump to .prof file to analyze later
        profiler.dump_stats(f"worker_device_{device_id}.prof")


@profile
def task_generate(
    results: mp.Queue,
    device: str,
    local_llm: transformer_lens.HookedTransformer,
    task_data: dict,
):
    seq_chunk = task_data["seq_chunk"]
    batch_size_seqs = data_cfg.model_batch_size_seqs
    all_acts = []
    seq_chunk = seq_chunk.to(device)
    local_llm.to(device)

    with torch.autocast("cuda", DTYPES[data_cfg.enc_dtype]):
        for start in range(0, seq_chunk.shape[0], batch_size_seqs):
            sub_chunk = seq_chunk[start : start + batch_size_seqs]
            _, cache = local_llm.run_with_cache(
                sub_chunk,
                stop_at_layer=data_cfg.layer + 1,
                names_filter=data_cfg.act_name,
                return_cache_object=True,
            )
            acts = cache.cache_dict[data_cfg.act_name]
            assert (
                acts.shape[-1] == data_cfg.act_size
            ), f"Expected {data_cfg.act_size} act size, got {acts.shape[-1]}"
            acts = acts.reshape(acts.shape[0] * acts.shape[1], data_cfg.act_size)
            all_acts.append(acts.to("cpu"))

    results.put(torch.cat(all_acts, dim=0))
    local_llm.to("cpu")


@profile
def model_step(model_entry, acts):
    autoenc = model_entry["model"]
    optimizer = model_entry["optimizer"]

    assert (
        acts.device == autoenc.device
    ), f"Acts on device {acts.device}, expected {autoenc.device}"

    loss, l2_loss, feature_acts, _ = autoenc(acts)
    loss.backward()
    autoenc.make_decoder_weights_and_grad_unit_norm()
    optimizer.step()
    optimizer.zero_grad()

    return {
        "loss": loss.item(),
        "l2_loss": l2_loss.item(),
        "feature_acts": feature_acts.detach(),
    }


@profile
def task_train(results: mp.Queue, device: str, task_data: dict):
    model_entry = task_data["model_entry"]
    static_buffer = task_data["static_buffer"]
    model_entry["model"].to(device)
    model_entry["act_freq_history"] = model_entry["act_freq_history"].to(device)
    for param_group in model_entry["optimizer"].state.values():
        for state_key, state_value in param_group.items():
            if torch.is_tensor(state_value):
                param_group[state_key] = state_value.to(device)

    metrics_list = []

    while not static_buffer.needs_refresh():
        acts = static_buffer.next().to(device)
        step_res = model_step(model_entry, acts)
        act_freq_batch = (step_res["feature_acts"] > 0).float().mean(dim=0)
        model_entry["act_freq_history"] += act_freq_batch

        # store step metrics
        metrics = {
            "arch_name": model_entry["name"],
            "loss": step_res["loss"],
            "l2_loss": step_res["l2_loss"],
            "act_freq": act_freq_batch.sum().item(),
            "n_iter": model_entry["n_iter"],
        }

        if (model_entry["n_iter"] + 1) % train_cfg.measure_dead_over_n_batches == 0:
            dead_features = model_entry["act_freq_history"] == 0

            if (
                model_entry["n_iter"] + 1
            ) % train_cfg.resample_dead_every_n_batches == 0:
                model_entry["model"].resample_sparse_features(dead_features)

            metrics["dead_features"] = dead_features.float().sum().item()
            model_entry["act_freq_history"] = torch.zeros(
                model_entry["model"].sparse_dim,
                dtype=torch.float,
                device=device,
            )

        metrics_list.append(metrics)
        model_entry["n_iter"] += 1

    for param_group in model_entry["optimizer"].state.values():
        for state_key, state_value in param_group.items():
            if torch.is_tensor(state_value):
                param_group[state_key] = state_value.cpu()
    # move back to CPU and return
    model_entry["model"].to("cpu")
    model_entry["act_freq_history"] = model_entry["act_freq_history"].to("cpu")
    results.put((model_entry, metrics_list))
