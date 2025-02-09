import logging
import os
from queue import Queue

import torch.multiprocessing as mp
import tqdm
import wandb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from mlsae.config import DEVICE_COUNT, data_cfg, train_cfg
from mlsae.model import models
from mlsae.shared_memory import SharedMemory
from mlsae.worker import TaskType, cpu_worker, gpu_worker, init_optimizer


class Trainer:
    def train(self):
        wandb.init(project=train_cfg.wandb_project, entity=train_cfg.wandb_entity)
        wandb.config.update(train_cfg)
        wandb.config.update(data_cfg)

        mp.set_start_method("spawn", force=True)

        # Instantiate models and optimizers
        logging.info("Instantiating models and optimizers")
        self.saes = []
        self.optimizers = []
        assert models
        for model in models:
            sae = model(
                act_size=data_cfg.act_size,
                device="cpu",
            )
            optimizer = init_optimizer(sae)
            self.saes.append(sae)
            self.optimizers.append(optimizer)

        wandb.config.update({"models": [sae.get_config_dict() for sae in self.saes]})

        # Instantiate shared memory
        logging.info("Instantiating shared memory")
        self.shared_memory = SharedMemory(self.saes, self.optimizers)

        # Instantiate queues
        logging.info("Instantiating queues and workers")
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

        self.training_needed_for_model_idx = set(range(len(self.saes)))
        self.training_completed_for_model_idx = set()
        self.act_block_idx_current_training = None

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
                assert not (
                    set(self.tokens_write_queue.queue)
                    & set(self.tokens_read_queue.queue)
                ), "Token queues have overlapping indices"
                assert not (
                    set(self.acts_write_queue.queue) & set(self.acts_read_queue.queue)
                ), "Act queues have overlapping indices"
                assert (
                    self.n_gpu_outstanding_tasks <= DEVICE_COUNT
                    and self.n_gpu_outstanding_tasks >= 0
                ), f"n_gpu_outstanding_tasks is {self.n_gpu_outstanding_tasks}, but should be between 0 and {DEVICE_COUNT}"
                assert self.training_needed_for_model_idx.isdisjoint(
                    self.training_completed_for_model_idx
                ), (
                    f"Overlap between needed and completed sets: "
                    f"{self.training_needed_for_model_idx & self.training_completed_for_model_idx}"
                )
                if self.should_add_tokens_task():
                    self.add_tokens_task()
                elif self.should_add_train_task():
                    self.add_train_task()
                elif self.should_add_acts_task():
                    self.add_acts_task()
                else:
                    logging.info("Waiting for result")
                    result = self.results_queue.get()
                    logging.info("Got result")
                    self.handle_result(result)
                logging.info(
                    f"Token read queue: {list(self.tokens_read_queue.queue)}, "
                    f"Token write queue: {list(self.tokens_write_queue.queue)}, "
                    f"Act read queue: {list(self.acts_read_queue.queue)}, "
                    f"Act write queue: {list(self.acts_write_queue.queue)}, "
                    f"GPU outstanding tasks: {self.n_gpu_outstanding_tasks}, "
                    f"CPU worker busy: {self.cpu_worker_busy}, "
                    f"Training needed for model idx: {self.training_needed_for_model_idx}, "
                    f"Training completed for model idx: {self.training_completed_for_model_idx}, "
                    f"Act block idx current training: {self.act_block_idx_current_training}"
                )
        finally:
            self.finish()

    def should_add_tokens_task(self):
        if self.cpu_worker_busy:
            return False
        return not self.tokens_write_queue.empty()

    def add_tokens_task(self):
        token_block_idx = self.tokens_write_queue.get(block=False)
        task_data = {"token_block_idx": token_block_idx}
        task = (TaskType.TOKENS, task_data)
        self.cpu_tasks_queue.put(task)
        self.cpu_worker_busy = True
        logging.info(f"Added tokens task {task_data}")

    def should_add_acts_task(self):
        if (
            self.n_gpu_outstanding_tasks == DEVICE_COUNT
            or self.tokens_read_queue.empty()
            or self.acts_write_queue.empty()
        ):
            return False
        return (
            # We are waiting on the generation of an act block to train models
            (
                self.acts_read_queue.empty()
                and len(self.training_needed_for_model_idx) == len(self.saes)
            )
            # We have a GPU to spare as other GPUs are busy training models but
            # there are no more models to train for this iteration
            or (
                not self.training_needed_for_model_idx
                and not self.acts_write_queue.empty()
            )
        )

    def add_acts_task(self):
        # Allocate token block (read) to this task
        token_block_idx = self.tokens_read_queue.get(block=False)
        # Allocate act block (write) to this task
        act_block_idx = self.acts_write_queue.get(block=False)
        task_data = {
            "token_block_idx": token_block_idx,
            "act_block_idx": act_block_idx,
        }
        task = (TaskType.ACTS, task_data)
        self.gpu_tasks_queue.put(task)
        self.n_gpu_outstanding_tasks += 1
        logging.info(f"Added acts task {task_data}")

    def should_add_train_task(self):
        if self.n_gpu_outstanding_tasks == DEVICE_COUNT:
            return False
        return (
            not self.acts_read_queue.empty()
            or self.act_block_idx_current_training is not None
        ) and self.training_needed_for_model_idx

    def add_train_task(self):
        # If we haven't started training models for a new act block, we choose
        # an act block to read from
        if self.act_block_idx_current_training is None:
            self.act_block_idx_current_training = self.acts_read_queue.get(block=False)
            assert len(self.training_needed_for_model_idx) == len(self.saes)

        # Take one model that needs to be trained, and it to the task queue
        model_idx = self.training_needed_for_model_idx.pop()
        task_data = {
            "model_idx": model_idx,
            "act_block_idx": self.act_block_idx_current_training,
        }
        task = (TaskType.TRAIN, task_data)
        self.gpu_tasks_queue.put(task)

        self.n_gpu_outstanding_tasks += 1
        logging.info(f"Added train task {task_data}")

    def handle_tokens_result(self, result_data):
        token_block_idx = result_data["token_block_idx"]
        logging.info(f"Got tokens result token block idx: {token_block_idx}")
        self.tokens_read_queue.put(token_block_idx)
        self.cpu_worker_busy = False

    def handle_acts_result(self, result_data):
        act_block_idx = result_data["act_block_idx"]
        token_block_idx = result_data["token_block_idx"]
        logging.info(
            f"Got acts result act block idx: {act_block_idx}, token block idx: {token_block_idx}"
        )
        self.acts_read_queue.put(act_block_idx)
        self.tokens_write_queue.put(token_block_idx)
        self.n_gpu_outstanding_tasks -= 1

    def handle_train_result(self, result_data):
        model_idx = result_data["model_idx"]
        act_block_idx = result_data["act_block_idx"]
        logging.info(
            f"Got train result model idx: {model_idx}, act block idx: {act_block_idx}"
        )
        self.training_completed_for_model_idx.add(model_idx)
        should_log = False
        if len(self.training_completed_for_model_idx) == len(self.saes):
            self.acts_write_queue.put(act_block_idx)
            self.pbar.update(1)
            self.train_iter += 1
            # Add models back to train idx
            self.training_needed_for_model_idx = set(range(len(self.saes)))
            self.training_completed_for_model_idx = set()
            self.act_block_idx_current_training = None
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
                aggregate_metrics_for_batch[f"{self.saes[model_idx].name}_{k}"] = v

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
        for sae in self.saes:
            try:
                sae.save(sae.name, save_to_s3=train_cfg.save_to_s3)
            except Exception as e:
                logging.error(f"Error saving {sae.name}: {e}")
        wandb.finish()

        for w in self.gpu_workers:
            w.join()
        self.cpu_worker.join()


if __name__ == "__main__":
    assert "HF_TOKEN" in os.environ, "HF_TOKEN must be set"
    assert "WANDB_API_KEY" in os.environ, "WANDB_API_KEY must be set"
    if train_cfg.save_to_s3:
        assert "AWS_ACCESS_KEY_ID" in os.environ, "AWS_ACCESS_KEY_ID must be set"
        assert (
            "AWS_SECRET_ACCESS_KEY" in os.environ
        ), "AWS_SECRET_ACCESS_KEY must be set"
    Trainer().train()
