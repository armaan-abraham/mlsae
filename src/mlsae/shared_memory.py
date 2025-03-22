import torch

from mlsae.config import DTYPES, data_cfg
from mlsae.model import DeepSAE
from mlsae.optimizer import SparseAdam


class SharedMemory:
    def __init__(
        self,
        saes: list[DeepSAE],
        optimizers: list[SparseAdam],
        n_act_blocks: int = data_cfg.n_act_blocks,
        n_token_blocks: int = data_cfg.n_token_blocks,
    ):
        assert len(saes) == len(optimizers)
        for sae in saes:
            sae.share_memory()
        self.shared_memory = {
            "act_blocks": [
                torch.zeros(
                    (data_cfg.act_block_size_entries, data_cfg.act_size),
                    dtype=DTYPES[data_cfg.sae_dtype],
                ).share_memory_()
                for _ in range(n_act_blocks)
            ],
            "token_blocks": [
                torch.zeros(
                    (data_cfg.act_block_size_seqs, data_cfg.seq_len),
                    dtype=torch.int32,
                ).share_memory_()
                for _ in range(n_token_blocks)
            ],
            "models": saes,
            "optimizers": [optimizer.share_memory_() for optimizer in optimizers],
            "act_freq_history": [
                torch.zeros(sae.sparse_dim, dtype=torch.float).share_memory_()
                for sae in saes
            ],
            "n_iter": [
                torch.tensor(0, dtype=torch.int64).share_memory_() for _ in saes
            ],
        }

    def __getitem__(self, key: str):
        return self.shared_memory[key]

    def __setitem__(self, key: str, value):
        self.shared_memory[key] = value
