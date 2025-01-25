from dataclasses import dataclass, field

import transformer_lens
import torch

@dataclass
class DataConfig:
    seed: int = 49
    buffer_batch_size_tokens: int = 65536
    buffer_size_buffer_batch_size_mult: int = 16
    seq_len: int = 128
    model_batch_size_seqs: int = 256
    enc_dtype: str = "fp32"
    cache_dtype: str = "bf16"
    model_name: str = "gpt2-small"
    site: str = "resid_pre"
    layer: int = 9
    act_size: int = 768
    dataset_name: str = "allenai/c4"
    dataset_column_name: str = "text"
    dataset_batch_size_entries: int = 50

    eval_data_seed: int = 59
    eval_batches: int = 500

    caching: bool = True
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
    def buffer_size_tokens(self) -> int:
        return self.buffer_batch_size_tokens * self.buffer_size_buffer_batch_size_mult

    @property
    def act_name(self) -> str:
        return transformer_lens.utils.get_act_name(self.site, self.layer)

    @property
    def eval_tokens(self) -> int:
        return self.buffer_batch_size_tokens * self.eval_tokens_buffer_batch_size_mult


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
                "lr": 4e-4,
            },
            {
                "name": "1",
                "encoder_dim_mults": [1],
                "sparse_dim_mult": 8,
                "decoder_dim_mults": [],
                "weight_decay": 5e-4,
                "l1_lambda": 0.25,
                "lr": 4e-4,
            },
            {
                "name": "2",
                "encoder_dim_mults": [1],
                "sparse_dim_mult": 8,
                "decoder_dim_mults": [1],
                "weight_decay": 5e-4,
                "l1_lambda": 0.25,
                "lr": 2e-4,
            },
            {
                "name": "3",
                "encoder_dim_mults": [1.5, 1],
                "sparse_dim_mult": 8,
                "decoder_dim_mults": [1, 1],
                "weight_decay": 5e-4,
                "l1_lambda": 0.25,
                "lr": 1e-4,
            },
        ]
    )

    num_tokens: int = int(2e9)
    wandb_project: str = "mlsae"
    wandb_entity: str = "armaanabraham-independent"
    save_to_s3: bool = False

    measure_dead_over_n_batches: int = 15
    resample_dead_every_n_batches: int = 375


train_cfg = TrainConfig()
data_cfg = DataConfig()

assert (
    train_cfg.resample_dead_every_n_batches % train_cfg.measure_dead_over_n_batches == 0
)

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}