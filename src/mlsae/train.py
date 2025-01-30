import logging
from collections import defaultdict
from dataclasses import dataclass, field
from queue import Queue

import torch
import torch.multiprocessing as mp
import tqdm
import wandb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from mlsae.config import DEVICE_COUNT, data_cfg, train_cfg
from mlsae.model import DeepSAE, SparseAdam
from mlsae.shared_memory import SharedMemory
from mlsae.worker import TaskType, cpu_worker, gpu_worker, init_optimizer


class Trainer:
    def train(self):
        wandb.init(project=train_cfg.wandb_project, entity=train_cfg.wandb_entity)
        wandb.config.update(train_cfg)
        wandb.config.update(data_cfg)

        mp.set_start_method("spawn", force=True)

        # Instantiate models and optimizers
        self.saes = []
        self.optimizers = []
        for i, arch_dict in enumerate(train_cfg.architectures):
            sae = DeepSAE(
                encoder_dim_mults=arch_dict["encoder_dim_mults"],
                sparse_dim_mult=arch_dict["sparse_dim_mult"],
                decoder_dim_mults=arch_dict["decoder_dim_mults"],
                act_size=data_cfg.act_size,
                enc_dtype=data_cfg.sae_dtype,
                device="cpu",
                topk=arch_dict["topk"],
                act_l2_coeff=arch_dict["act_l2_coeff"],
                name=arch_dict["name"],
            )
            optimizer = init_optimizer(sae, i)
            self.saes.append(sae)
            self.optimizers.append(optimizer)

        # Instantiate shared memory
        self.shared_memory = SharedMemory(self.saes, self.optimizers)

        # Instantiate queues
        self.gpu_tasks_queue = mp.Queue()
        self.cpu_tasks_queue = mp.Queue()
        self.results_queue = mp.Queue()

        self.gpu_workers = []
        # Instantiate workers
        for device_id in range(DEVICE_COUNT):
            gpu_worker_p = mp.Process(
                target=gpu_worker,
                args=(
                    device_id,
                    self.gpu_tasks_queue,
                    self.results_queue,
                    self.shared_memory,
                ),
            )
            self.gpu_workers.append(gpu_worker_p)
            gpu_worker_p.start()
        self.n_gpu_outstanding_tasks = 0

        self.cpu_worker_busy = False
        self.cpu_worker = mp.Process(
            target=cpu_worker,
            args=(self.cpu_tasks_queue, self.results_queue, self.shared_memory),
        )
        self.cpu_worker.start()

        # Two queues: act write queue and act read queue
        # Whatever indices are not in the read and write queue are currently being written to
        # If a block is done being read by all of the models, then add it to the write queue

        # When a train task takes place, do we just add tasks for all of the models?
        # Yes, because as long as there is one read block available, all of those models read from the same block
        # so they will not spin until they are all done.

        # When we come across the read queue and it is below a certain threshold size, then we submit blocks from the write queue
        # to workers.

        self.train_iter = 0
        self.max_train_iter = train_cfg.num_tokens // data_cfg.act_block_size_tokens

        self.tokens_write_queue = Queue(maxsize=data_cfg.n_token_blocks)
        self.tokens_read_queue = Queue(maxsize=data_cfg.n_token_blocks)
        for i in range(data_cfg.n_token_blocks):
            self.tokens_write_queue.put(i)

        self.acts_write_queue = Queue(maxsize=data_cfg.n_act_blocks)
        self.acts_read_queue = Queue(maxsize=data_cfg.n_act_blocks)
        for i in range(data_cfg.n_act_blocks):
            self.acts_write_queue.put(i)

        self.training_needed_for_model_idx = set(range(len(train_cfg.architectures)))
        self.training_completed_for_model_idx = set()
        self.current_act_block_idx = None

        self.metrics_aggregator = []

        try:
            self.pbar = tqdm.tqdm(total=self.max_train_iter, desc="Training")
            while self.train_iter < self.max_train_iter:
                assert (
                    self.acts_write_queue.qsize() + self.acts_read_queue.qsize()
                    <= data_cfg.n_act_blocks
                )
                assert (
                    self.tokens_read_queue.qsize() + self.tokens_write_queue.qsize()
                    <= data_cfg.n_token_blocks
                )
                assert self.n_gpu_outstanding_tasks <= DEVICE_COUNT
                if self.should_add_tokens_task():
                    self.add_tokens_task()
                elif self.should_add_train_task():
                    self.add_train_task()
                elif self.should_add_acts_task():
                    self.add_acts_task()
                else:
                    result = self.results_queue.get()
                    self.handle_result(result)
        finally:
            self.finish()

    def should_add_tokens_task(self):
        if self.cpu_worker_busy:
            return False
        return not self.tokens_write_queue.empty()

    def add_tokens_task(self):
        token_block_idx = self.tokens_write_queue.get(block=False)
        task = (TaskType.TOKENS, {"token_block_idx": token_block_idx})
        self.cpu_tasks_queue.put(task)
        self.cpu_worker_busy = True

    def should_add_acts_task(self):
        if (
            self.n_gpu_outstanding_tasks == DEVICE_COUNT
            or self.tokens_read_queue.empty()
            or self.acts_write_queue.empty()
        ):
            return False
        return (
            # We are waiting on the generation of an act block to train models
            (self.acts_read_queue.empty() and len(self.training_needed_for_model_idx) == len(train_cfg.architectures))

            # We have a GPU to spare as other GPUs are busy training models but
            # there are no more models to train for this iteration
            or (not self.training_needed_for_model_idx and not self.acts_write_queue.empty())
        )

    def add_acts_task(self):
        # Allocate token block (read) to this task
        token_block_idx = self.tokens_read_queue.get(block=False)
        # Allocate act block (write) to this task
        act_block_idx = self.acts_write_queue.get(block=False)
        task = (
            TaskType.ACTS,
            {"token_block_idx": token_block_idx, "act_block_idx": act_block_idx},
        )
        self.gpu_tasks_queue.put(task)
        self.n_gpu_outstanding_tasks += 1

    def should_add_train_task(self):
        if self.n_gpu_outstanding_tasks == DEVICE_COUNT:
            return False
        return not self.acts_read_queue.empty() and self.training_needed_for_model_idx

    def add_train_task(self):
        # If we haven't started training models for a new act block, we choose
        # an act block to read from
        if self.current_act_block_idx is None:
            self.current_act_block_idx = self.acts_read_queue.get(block=False)
            assert len(self.training_needed_for_model_idx) == len(train_cfg.architectures)

        # Take one model that needs to be trained, and it to the task queue
        model_idx = self.training_needed_for_model_idx.pop()
        task = (
            TaskType.TRAIN,
            {"model_idx": model_idx, "act_block_idx": self.current_act_block_idx},
        )
        self.gpu_tasks_queue.put(task)

        self.n_gpu_outstanding_tasks += 1

    def handle_tokens_result(self, result_data):
        token_block_idx = result_data["token_block_idx"]
        self.tokens_read_queue.put(token_block_idx)
        self.cpu_worker_busy = False

    def handle_acts_result(self, result_data):
        act_block_idx = result_data["act_block_idx"]
        token_block_idx = result_data["token_block_idx"]
        self.acts_read_queue.put(act_block_idx)
        self.tokens_write_queue.put(token_block_idx)
        self.n_gpu_outstanding_tasks -= 1

    def handle_train_result(self, result_data):
        model_idx = result_data["model_idx"]
        act_block_idx = result_data["act_block_idx"]
        self.training_completed_for_model_idx.add(model_idx)
        should_log = False
        if len(self.training_completed_for_model_idx) == len(train_cfg.architectures):
            self.acts_write_queue.put(act_block_idx)
            self.pbar.update(1)
            self.train_iter += 1
            # Add models back to train idx
            self.training_needed_for_model_idx = set(
                range(len(train_cfg.architectures))
            )
            self.training_completed_for_model_idx = set()
            self.current_act_block_idx = None
            should_log = True
        self.aggregate_and_log_metrics(result_data, should_log)
        self.n_gpu_outstanding_tasks -= 1

    def aggregate_and_log_metrics(self, result_data, should_log):
        metrics = result_data["metrics"]
        model_idx = result_data["model_idx"]
        if not self.metrics_aggregator:
            self.metrics_aggregator = [
                {} for _ in range(data_cfg.act_block_size_sae_batch_size_mult)
            ]
        for metrics_for_batch, aggregate_metrics_for_batch in zip(
            metrics, self.metrics_aggregator
        ):
            for k, v in metrics_for_batch.items():
                aggregate_metrics_for_batch[
                    f"{train_cfg.architectures[model_idx]['name']}_{k}"
                ] = v

        if should_log:
            # All train tasks for this act block have been completed, log all
            # together
            for i, aggregate_metrics_for_batch in enumerate(self.metrics_aggregator):
                log = aggregate_metrics_for_batch
                if i == 0:
                    log.update(self.get_training_state())
                wandb.log(log)
            self.metrics_aggregator = []

    def get_training_state(self):
        return {
            "cpu_worker_busy": self.cpu_worker_busy,
            "train_iter": self.train_iter,
            "n_gpu_outstanding_tasks": self.n_gpu_outstanding_tasks,
            "acts_read_queue_size": self.acts_read_queue.qsize(),
            "acts_write_queue_size": self.acts_write_queue.qsize(),
            "tokens_read_queue_size": self.tokens_read_queue.qsize(),
            "tokens_write_queue_size": self.tokens_write_queue.qsize(),
        }

    def handle_result(self, result):
        if isinstance(result, Exception):
            raise result
        task_type: TaskType = result[0]
        match task_type:
            case TaskType.TOKENS:
                self.handle_tokens_result(result[1])
            case TaskType.ACTS:
                self.handle_acts_result(result[1])
            case TaskType.TRAIN:
                self.handle_train_result(result[1])

    def finish(self):
        self.pbar.close()

        # Signal workers to stop
        self.cpu_tasks_queue.put(None)
        for _ in self.gpu_workers:
            self.gpu_tasks_queue.put(None)

        logging.info("Saving all models...")
        for arch_dict, sae in zip(train_cfg.architectures, self.saes):
            try:
                sae.save(arch_dict["name"], save_to_s3=train_cfg.save_to_s3)
            except Exception as e:
                logging.error(f"Error saving {arch_dict['name']}: {e}")
        wandb.finish()

        for w in self.gpu_workers:
            w.join()
        self.cpu_worker.join()


if __name__ == "__main__":
    Trainer().train()
