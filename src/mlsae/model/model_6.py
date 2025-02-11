from mlsae.model.model import DeepSAE
import torch.nn as nn

class DeepSAE6(DeepSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[1],
            sparse_dim_mult=8,
            decoder_dim_mults=[1],
            name="1-1-6",
            enc_dtype="fp32",
            device=device,
            topk=4,
            act_decay=1e-4,
            lr=2e-4,
        )

    def _init_params(self):
        super()._init_params()
        self._tie_weights()

    def _tie_weights(self):
        # Get the encoder linear layers:
        # For each dense encoder block, the linear layer is the first module in the Sequential.
        encoder_linears = [block[0] for block in self.dense_encoder_blocks]
        # Append the linear layer from the sparse encoder block (its first module)
        encoder_linears.append(self.sparse_encoder_block[0])

        # Get the decoder linear layers.
        # Note that in _init_decoder_params, we add linear layers interleaved with ReLU.
        # Extract only the Linear modules.
        decoder_linears = [module for module in self.decoder_blocks if isinstance(module, nn.Linear)]

        if len(encoder_linears) != len(decoder_linears):
            raise ValueError("The number of encoder and decoder linear layers must match.")

        # Pair them: the first encoder layer corresponds to the last decoder layer,
        # the second encoder layer to the second-to-last decoder layer, and so on.
        for i, enc_layer in enumerate(encoder_linears):
            dec_layer = decoder_linears[len(decoder_linears) - 1 - i]
            # Set the decoder weight to be the transpose of the encoder weight.
            dec_layer.weight.data.copy_(enc_layer.weight.data.t())
            dec_layer.weight.data = dec_layer.weight.data / dec_layer.weight.data.norm(dim=-1, keepdim=True)
