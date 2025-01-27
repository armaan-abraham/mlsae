# train.py

import logging
from dataclasses import dataclass, field

import torch
import torch.multiprocessing as mp
import tqdm
import wandb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from mlsae.config import data_cfg, train_cfg
from mlsae.data import Buffer
from mlsae.model import DeepSAE
from mlsae.worker import TaskType, worker


def main():
    wandb.init(project=train_cfg.wandb_project, entity=train_cfg.wandb_entity)
    wandb.config.update(train_cfg)
    wandb.config.update(data_cfg)

    # ===== Instantiate workers =====
    mp.set_start_method("spawn", force=True)

    tasks_queue = mp.Queue()
    results_queue = mp.Queue()
    workers = []
    for device_id in range(torch.cuda.device_count()):
        p = mp.Process(
            target=worker,
            args=(device_id, tasks_queue, results_queue),
        )
        p.start()
        workers.append(p)

    logging.info("Building buffer...")
    buffer = Buffer(results_queue, tasks_queue)

    logging.info("Building models in CPU...")
    autoencoders = []
    for i, arch_dict in enumerate(train_cfg.architectures):
        autoenc = DeepSAE(
            encoder_dim_mults=arch_dict["encoder_dim_mults"],
            sparse_dim_mult=arch_dict["sparse_dim_mult"],
            decoder_dim_mults=arch_dict["decoder_dim_mults"],
            act_size=data_cfg.act_size,
            enc_dtype=data_cfg.enc_dtype,
            device="cpu",
            topk=arch_dict["topk"],
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
                "act_freq_history": torch.zeros(autoenc.sparse_dim, dtype=torch.float),
            }
        )

    # Number of times we refresh the buffer
    num_buffer_refreshes = train_cfg.num_tokens // data_cfg.buffer_size_tokens
    logging.info(f"Will refresh buffer {num_buffer_refreshes} times")

    try:
        for outer_idx in tqdm.trange(num_buffer_refreshes, desc="Buffer refreshes"):
            # Refresh buffer
            buffer.refresh()

            # For each model, clone the static buffer and enqueue a task
            for model_entry in autoencoders:
                logging.info(f"Enqueuing task for {model_entry['name']}")
                buf_clone = buffer.static_buffer.clone()
                tasks_queue.put(
                    (
                        TaskType.TRAIN,
                        {"model_entry": model_entry, "static_buffer": buf_clone},
                    )
                )

            # Collect results for each model
            updated_autoencoders = []
            all_metrics = []
            for _ in range(len(autoencoders)):
                result = results_queue.get()
                if isinstance(result, tuple) and result[0] == "error":
                    error_info = result[1]
                    error_msg = (
                        f"Worker error on device {error_info['device_id']} "
                        f"for model {error_info['model_name']}:\n"
                        f"{error_info['error_type']}: {error_info['error_message']}\n"
                        f"Traceback:\n{error_info['traceback']}"
                    )
                    logging.error(error_msg)
                    raise RuntimeError(error_msg)
                model_entry, metrics_list = result
                updated_autoencoders.append(model_entry)
                all_metrics.extend(metrics_list)
                logging.info(f"Collected results for {model_entry['name']}")

            # Reassign
            autoencoders = updated_autoencoders

            # Group step metrics and log them
            metrics_by_step = {}
            logging.info(f"Grouping metrics for {len(all_metrics)} steps")
            logging.info(f"Metrics[0]: {all_metrics[0]}")
            for m in all_metrics:
                arch_name = m["arch_name"]
                step_idx = m["n_iter"]
                # Initialize the dict for this step if it doesn't exist
                if step_idx not in metrics_by_step:
                    metrics_by_step[step_idx] = {}
                metrics_by_step[step_idx].update(
                    {
                        f"{arch_name}_{k}": v
                        for k, v in m.items()
                        if k not in ["arch_name", "n_iter"]
                    }
                )

            for step_idx in sorted(metrics_by_step.keys()):
                log_dict = metrics_by_step[step_idx]
                wandb.log(log_dict)

    finally:
        # Signal workers to stop
        for _ in workers:
            tasks_queue.put(None)

        logging.info("Saving all models...")
        for entry in autoencoders:
            try:
                entry["model"].save(entry["name"], save_to_s3=train_cfg.save_to_s3)
            except Exception as e:
                logging.error(f"Error saving {entry['name']}: {e}")
        wandb.finish()

        for w in workers:
            w.join()


if __name__ == "__main__":
    main()
