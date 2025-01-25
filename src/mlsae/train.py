# train.py

import logging
from dataclasses import dataclass, field

import torch
import torch.multiprocessing as mp
import tqdm
import wandb

from mlsae.data import Buffer, StaticBuffer, data_cfg
from mlsae.model import DeepSAE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@dataclass
class TrainConfig:
    architectures: list = field(
        default_factory=lambda: [
            {
                "name": "0",
                "encoder_dim_mults": [],
                "sparse_dim_mult": 8,
                "decoder_dim_mults": [],
                "weight_decay": 5e-4,
                "l1_lambda": 0.25,
                "lr": 4e-3,
            },
            {
                "name": "1",
                "encoder_dim_mults": [1],
                "sparse_dim_mult": 8,
                "decoder_dim_mults": [],
                "weight_decay": 5e-4,
                "l1_lambda": 0.25,
                "lr": 4e-3,
            },
            {
                "name": "2",
                "encoder_dim_mults": [1],
                "sparse_dim_mult": 8,
                "decoder_dim_mults": [1],
                "weight_decay": 5e-4,
                "l1_lambda": 0.25,
                "lr": 4e-3,
            },
            {
                "name": "3",
                "encoder_dim_mults": [1.5, 1],
                "sparse_dim_mult": 8,
                "decoder_dim_mults": [1, 1.5],
                "weight_decay": 5e-4,
                "l1_lambda": 0.25,
                "lr": 4e-3,
            },
        ]
    )

    num_tokens: int = int(2e9)
    wandb_project: str = "mlsae"
    wandb_entity: str = "armaanabraham-independent"
    save_to_s3: bool = False

    measure_dead_over_n_batches: int = 10
    resample_dead_every_n_batches: int = 350


train_cfg = TrainConfig()

assert (
    train_cfg.resample_dead_every_n_batches % train_cfg.measure_dead_over_n_batches == 0
)


def model_step(model_entry, acts):
    device = model_entry["device"]
    autoenc = model_entry["model"]
    optimizer = model_entry["optimizer"]

    acts = acts.to(device, non_blocking=True)
    loss, mse_loss, l1_loss, feature_acts, _ = autoenc(acts)
    loss.backward()
    autoenc.make_decoder_weights_and_grad_unit_norm()
    optimizer.step()
    optimizer.zero_grad()

    return {
        "loss": loss.item(),
        "mse_loss": mse_loss.item(),
        "l1_loss": l1_loss.item(),
        "feature_acts": feature_acts.detach(),
    }


def train_worker(device_id, task_queue, results_queue):
    """
    Single worker pinned to device_id. Receives (model_entry, static_buffer) from
    a single shared task queue. Moves them to 'cuda:device_id', runs training,
    returns (updated_model_entry, metrics_list) in results_queue.
    """
    device_str = f"cuda:{device_id}"
    while True:
        task = task_queue.get()
        if task is None:
            break
        model_entry, static_buffer = task
        try:
            model_entry["model"].to(device_str)
            static_buffer.to(device_str)
            model_entry["act_freq_history"] = model_entry["act_freq_history"].to(
                device_str
            )

            metrics_list = []

            while not static_buffer.needs_refresh():
                acts = static_buffer.next()
                step_res = model_step(model_entry, acts)
                act_freq_batch = (step_res["feature_acts"] == 0).mean(dim=0)
                model_entry["act_freq_history"] += act_freq_batch

                # store step metrics
                metrics = {
                    "arch_name": model_entry["name"],
                    "loss": step_res["loss"],
                    "mse_loss": step_res["mse_loss"],
                    "l1_loss": step_res["l1_loss"],
                    "act_freq": act_freq_batch.mean().item(),
                    "n_iter": model_entry["n_iter"],
                }

                if (
                    model_entry["n_iter"] + 1
                ) % train_cfg.measure_dead_over_n_batches == 0:
                    dead_features = act_freq_history == 0

                    if (
                        model_entry["n_iter"] + 1
                    ) % train_cfg.resample_dead_every_n_batches == 0:
                        model_entry["model"].resample_sparse_features(dead_features)

                    metrics["dead_features"] = dead_features.mean().item()
                    act_freq_history = torch.zeros(
                        model_entry["model"].sparse_dim, dtype=torch.float
                    )

                metrics_list.append(metrics)
                model_entry["n_iter"] += 1

            # move back to CPU
            model_entry["model"].to("cpu")
            model_entry["act_freq_history"] = model_entry["act_freq_history"].to("cpu")
            results_queue.put((model_entry, metrics_list))
        except Exception as e:
            results_queue.put(e)


def main():
    wandb.init(project=train_cfg.wandb_project, entity=train_cfg.wandb_entity)
    wandb.config.update(train_cfg)
    wandb.config.update(data_cfg)

    logging.info("Building buffer...")
    buffer = Buffer()

    logging.info("Building models in CPU...")
    n_gpus = torch.cuda.device_count()
    autoencoders = []
    for i, arch_dict in enumerate(train_cfg.architectures):
        autoenc = DeepSAE(
            encoder_dim_mults=arch_dict["encoder_dim_mults"],
            sparse_dim_mult=arch_dict["sparse_dim_mult"],
            decoder_dim_mults=arch_dict["decoder_dim_mults"],
            act_size=data_cfg.act_size,
            enc_dtype=data_cfg.enc_dtype,
            device="cpu",
            l1_lambda=arch_dict["l1_lambda"],
            name=arch_dict["name"],
        )
        optimizer = torch.optim.SGD(
            autoenc.get_param_groups(weight_decay=arch_dict["weight_decay"]),
            lr=arch_dict["lr"],
        )

        autoencoders.append(
            {
                "model": autoenc,
                "optimizer": optimizer,
                "name": arch_dict["name"],
                "n_iter": 0,
            }
        )

    # Number of times we refresh the buffer
    num_buffer_refreshes = train_cfg.num_tokens // data_cfg.buffer_size_tokens
    logging.info(f"Will refresh buffer {num_buffer_refreshes} times")

    # Create a single shared task queue and a single shared results queue
    task_queue = mp.Queue()
    results_queue = mp.Queue()

    # Start as many workers as GPUs
    mp.set_start_method("spawn", force=True)
    workers = []
    for device_id in range(n_gpus):
        p = mp.Process(target=train_worker, args=(device_id, task_queue, results_queue))
        p.start()
        workers.append(p)

    try:
        for outer_idx in tqdm.trange(num_buffer_refreshes, desc="Buffer refreshes"):
            # Refresh buffer
            buffer.refresh()

            # For each model, clone the static buffer and enqueue a task
            for model_entry in autoencoders:
                buf_clone = buffer.static_buffer.clone()
                task_queue.put((model_entry, buf_clone))

            # Collect results for each model
            updated_autoencoders = []
            all_metrics = []
            for _ in range(len(autoencoders)):
                result = results_queue.get()
                if isinstance(result, Exception):
                    logging.error(f"Worker encountered an error: {result}")
                    raise RuntimeError(result)
                model_entry, metrics_list = result
                updated_autoencoders.append(model_entry)
                all_metrics.extend(metrics_list)

            # Reassign
            autoencoders = updated_autoencoders

            # Group step metrics and log them
            metrics_by_step = {}
            for m in all_metrics:
                arch_name = m["arch_name"]
                metrics_by_step[m["step_index"]].update(
                    {
                        f"{arch_name}_{k}": v
                        for k, v in m.items()
                        if k not in ["arch_name", "step_index"]
                    }
                )

            for step_idx in sorted(metrics_by_step.keys()):
                log_dict = metrics_by_step[step_idx]
                wandb.log(log_dict)

    finally:
        logging.info("Saving all models...")
        for entry in autoencoders:
            try:
                entry["model"].save(entry["name"], save_to_s3=train_cfg.save_to_s3)
            except Exception as e:
                logging.error(f"Error saving {entry['name']}: {e}")
        wandb.finish()

        # Signal workers to stop, then join
        for _ in workers:
            task_queue.put(None)
        for w in workers:
            w.join()


if __name__ == "__main__":
    main()
