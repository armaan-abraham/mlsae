from copy import deepcopy
from dataclasses import dataclass, field

import torch
import transformer_lens


@dataclass
class DataConfig:
    seed: int = 49
    sae_batch_size_tokens: int = 131072
    act_block_size_sae_batch_size_mult: int = 16

    seq_len: int = 128
    llm_batch_size_seqs: int = 256

    sae_dtype: str = "fp32"
    cache_dtype: str = "bf16"
    model_name: str = "gpt2-small"
    site: str = "resid_pre"
    layer: int = 9
    act_size: int = 768
    dataset_name: str = "allenai/c4"
    dataset_column_name: str = "text"
    dataset_batch_size_entries: int = 50

    n_token_blocks: int = 10
    get_token_blocks_threshold: int = 4
    n_act_blocks: int = 5
    get_act_blocks_threshold: int = 2

    eval_data_seed: int = 59
    eval_batches: int = 500

    caching: bool = False
    # These are the fields that must match before reusing an existing cache
    cache_id_fields: list = field(
        default_factory=lambda: [
            # TODO: eval data seed
            "seed",
            "model_name",
            "layer",
            "site",
            "seq_len",
            "act_size",
            "enc_dtype",
            "dataset_name",
        ]
    )

    @property
    def act_block_size_tokens(self) -> int:
        return self.sae_batch_size_tokens * self.act_block_size_sae_batch_size_mult

    @property
    def act_block_size_seqs(self) -> int:
        return self.act_block_size_tokens // self.seq_len

    @property
    def act_name(self) -> str:
        return transformer_lens.utils.get_act_name(self.site, self.layer)

    @property
    def eval_tokens(self) -> int:
        return self.sae_batch_size_tokens * self.eval_tokens_buffer_batch_size_mult


@dataclass
class TrainConfig:
    architectures: list = field(
        default_factory=lambda: [
            # === 0-0 ===
            {
                "name": "0-0.0",
                "encoder_dim_mults": [],
                "sparse_dim_mult": 16,
                "decoder_dim_mults": [],
                "weight_decay": 1e-4,
                "lr": 1e-4,
                "topk": 64,
                "act_l2_coeff": 0.3,
            },
            {
                "name": "0-0.1",
                "encoder_dim_mults": [],
                "sparse_dim_mult": 16,
                "decoder_dim_mults": [],
                "weight_decay": 1e-4,
                "lr": 1e-4,
                "topk": 64,
                "act_l2_coeff": 0.6,
            },
            {
                "name": "0-0.2",
                "encoder_dim_mults": [],
                "sparse_dim_mult": 16,
                "decoder_dim_mults": [],
                "weight_decay": 1e-4,
                "lr": 1e-4,
                "topk": 64,
                "act_l2_coeff": 1.2,
            },
            # === 1-0 ===
            # Varying l2 coeff
            {
                "name": "1-0.0",
                "encoder_dim_mults": [1],
                "sparse_dim_mult": 16,
                "decoder_dim_mults": [],
                "weight_decay": 1e-4,
                "lr": 1e-4,
                "topk": 64,
                "act_l2_coeff": 0.3,
            },
            {
                "name": "1-0.1",
                "encoder_dim_mults": [1],
                "sparse_dim_mult": 16,
                "decoder_dim_mults": [],
                "weight_decay": 1e-4,
                "lr": 1e-4,
                "topk": 64,
                "act_l2_coeff": 0.6,
            },
            {
                "name": "1-0.2",
                "encoder_dim_mults": [1],
                "sparse_dim_mult": 16,
                "decoder_dim_mults": [],
                "weight_decay": 1e-4,
                "lr": 1e-4,
                "topk": 64,
                "act_l2_coeff": 1.2,
            },
            # Varying weight decay
            {
                "name": "1-0.3",
                "encoder_dim_mults": [1],
                "sparse_dim_mult": 16,
                "decoder_dim_mults": [],
                "weight_decay": 2e-4,
                "lr": 1e-4,
                "topk": 64,
                "act_l2_coeff": 0.3,
            },
            {
                "name": "1-0.4",
                "encoder_dim_mults": [1],
                "sparse_dim_mult": 16,
                "decoder_dim_mults": [],
                "weight_decay": 4e-4,
                "lr": 1e-4,
                "topk": 64,
                "act_l2_coeff": 0.3,
            },
            # === 1-1 ===
            # Varying l2 coeff
            {
                "name": "1-1.0",
                "encoder_dim_mults": [1],
                "sparse_dim_mult": 16,
                "decoder_dim_mults": [1],
                "weight_decay": 1e-4,
                "lr": 1e-4,
                "topk": 64,
                "act_l2_coeff": 0.3,
            },
            {
                "name": "1-1.1",
                "encoder_dim_mults": [1],
                "sparse_dim_mult": 16,
                "decoder_dim_mults": [1],
                "weight_decay": 1e-4,
                "lr": 1e-4,
                "topk": 64,
                "act_l2_coeff": 0.6,
            },
            {
                "name": "1-1.2",
                "encoder_dim_mults": [1],
                "sparse_dim_mult": 16,
                "decoder_dim_mults": [1],
                "weight_decay": 1e-4,
                "lr": 1e-4,
                "topk": 64,
                "act_l2_coeff": 1.2,
            },
            # Varying weight decay
            {
                "name": "1-1.3",
                "encoder_dim_mults": [1],
                "sparse_dim_mult": 16,
                "decoder_dim_mults": [1],
                "weight_decay": 2e-4,
                "lr": 1e-4,
                "topk": 64,
                "act_l2_coeff": 0.3,
            },
            {
                "name": "1-1.4",
                "encoder_dim_mults": [1],
                "sparse_dim_mult": 16,
                "decoder_dim_mults": [1],
                "weight_decay": 4e-4,
                "lr": 1e-4,
                "topk": 64,
                "act_l2_coeff": 0.3,
            },
        ]
    )

    num_tokens: int = int(3e9)
    wandb_project: str = "mlsae"
    wandb_entity: str = "armaanabraham-independent"
    save_to_s3: bool = True

    measure_dead_over_n_batches: int = 15
    resample_dead_every_n_batches: int = int(15e9)


train_cfg = TrainConfig()
data_cfg = DataConfig()

assert (
    train_cfg.resample_dead_every_n_batches % train_cfg.measure_dead_over_n_batches == 0
)

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

DEVICE_COUNT = torch.cuda.device_count()
