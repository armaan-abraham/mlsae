from copy import deepcopy
from dataclasses import dataclass, field

import torch
import transformer_lens


@dataclass
class DataConfig:
    seed: int = 49
    sae_batch_size_entries: int = 131072
    act_block_size_sae_batch_size_mult: int = 256

    seq_len: int = 16
    llm_batch_size_seqs: int = 500

    sae_dtype: str = "fp32"
    model_name: str = "pythia-14m"
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"
    site: str = "resid_pre"
    layer: int = 1
    act_size_full: int = 128
    act_size_full_multiple: int = 16
    dataset_name: str = "allenai/c4"
    dataset_column_name: str = "text"
    dataset_batch_size_entries: int = 20

    n_token_blocks: int = 5
    n_act_blocks: int = 2

    @property
    def act_size(self) -> int:
        return self.act_size_full // self.act_size_full_multiple

    @property
    def act_block_size_tokens(self) -> int:
        return int(self.sae_batch_size_tokens * self.act_block_size_sae_batch_size_mult)

    @property
    def act_block_size_entries(self) -> int:
        return self.act_block_size_tokens * self.act_size_full_multiple

    @property
    def act_block_size_seqs(self) -> int:
        return self.act_block_size_tokens // self.seq_len

    @property
    def sae_batch_size_tokens(self) -> int:
        return self.sae_batch_size_entries // self.act_size_full_multiple

    @property
    def act_name(self) -> str:
        return transformer_lens.utils.get_act_name(self.site, self.layer)


data_cfg = DataConfig()


@dataclass
class TrainConfig:
    num_entries: int = int(8e9)
    wandb_project: str = "mlsae"
    wandb_entity: str = "armaanabraham-independent"
    save_to_s3: bool = False

    measure_dead_over_n_batches: int = 15

    @property
    def num_tokens(self) -> int:
        return self.num_entries // data_cfg.act_size_full_multiple


train_cfg = TrainConfig()

assert data_cfg.act_block_size_tokens % data_cfg.seq_len == 0
assert data_cfg.sae_batch_size_tokens % data_cfg.act_size_full_multiple == 0
assert (
    data_cfg.act_block_size_seqs
    == data_cfg.sae_batch_size_entries
    / data_cfg.act_size_full_multiple
    / data_cfg.seq_len
    * data_cfg.act_block_size_sae_batch_size_mult
)

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

DEVICE_COUNT = torch.cuda.device_count()
