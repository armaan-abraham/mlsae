from copy import deepcopy
from dataclasses import dataclass, field

import torch
import transformer_lens


@dataclass
class DataConfig:
    seed: int = 49
    sae_batch_size_tokens: int = 100_000
    act_block_size_sae_batch_size_mult: int = 200

    seq_len: int = 64
    llm_batch_size_seqs: int = 2000

    sae_dtype: str = "fp32"
    cache_dtype: str = "bf16"
    model_name: str = "pythia-31m"
    tokenizer_name: str = "EleutherAI/pythia-31m"
    site: str = "resid_pre"
    layer: int = 2
    act_size: int = 256
    dataset_name: str = "allenai/c4"
    dataset_column_name: str = "text"
    dataset_batch_size_entries: int = 50

    n_token_blocks: int = 5
    n_act_blocks: int = 1

    @property
    def act_block_size_tokens(self) -> int:
        return int(self.sae_batch_size_tokens * self.act_block_size_sae_batch_size_mult)

    @property
    def act_block_size_seqs(self) -> int:
        return self.act_block_size_tokens // self.seq_len

    @property
    def act_name(self) -> str:
        return transformer_lens.utils.get_act_name(self.site, self.layer)


@dataclass
class TrainConfig:
    num_tokens: int = int(4e9)
    wandb_project: str = "mlsae"
    wandb_entity: str = "armaanabraham-independent"
    save_to_s3: bool = False

    measure_dead_over_n_batches: int = 15
    resample_dead_every_n_batches: int = 3015


train_cfg = TrainConfig()
data_cfg = DataConfig()

assert (
    train_cfg.resample_dead_every_n_batches % train_cfg.measure_dead_over_n_batches == 0
)
assert data_cfg.act_block_size_tokens % data_cfg.seq_len == 0

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

DEVICE_COUNT = torch.cuda.device_count()
