import torch

from mlsae.config import data_cfg
from mlsae.model import DeepSAE, SparseAdam


class SharedMemory:
    def __init__(
        self,
        saes: list[DeepSAE],
        optimizers: list[SparseAdam],
        n_act_blocks: int = data_cfg.n_act_blocks,
        n_token_blocks: int = data_cfg.n_token_blocks,
    ):
        assert len(saes) == len(optimizers)
        self.shared_memory = {
            "act_blocks": [
                torch.zeros(
                    data_cfg.act_block_size_tokens,
                    data_cfg.act_size,
                    dtype=data_cfg.sae_dtype,
                ).share_memory_()
                for _ in range(n_act_blocks)
            ],
            "token_blocks": [
                torch.zeros(
                    data_cfg.act_block_size_seqs, data_cfg.seq_len, dtype=torch.int32
                ).share_memory_()
                for _ in range(n_token_blocks)
            ],
            "models": [sae.share_memory_() for sae in saes],
            "optimizers": [optimizer.share_memory_() for optimizer in optimizers],
            "act_freq_history": [
                torch.zeros(sae.sparse_dim, dtype=torch.float).share_memory_()
                for sae in saes
            ],
            "n_iter": [torch.tensor(0, dtype=torch.int64).share_memory_() for _ in saes],
        }
