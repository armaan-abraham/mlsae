from mlsae.model.model import ExperimentSAEBase, TopKActivation
import torch
import math
import torch.nn as nn

class ResSAE(ExperimentSAEBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.act_squeeze == 0
        assert len(set(self.encoder_dims)) == 1
        assert len(set(self.decoder_dims)) == 1

    def _forward(self, x, iteration=None):
        # Encode
        resid = x

        if self.encoder_dims:

            for block in self.dense_encoder_blocks:
                out = block(resid)

                if out.shape[1] == resid.shape[1]:
                    # Simple case: shapes match, just add
                    resid = resid + out
                else:
                    assert out.shape[1] > resid.shape[1]
                    padding = torch.zeros_like(out)
                    padding[:, :resid.shape[1]] = resid
                    resid = out + padding

            # Dead neuron counts are very sensitive to initial scaling. I have
            # found that dividing by the L2 norm of the input activations helps
            # for shallow SAEs. This division makes it so that we initially
            # divide by the L2 norm in combination with the layernorm (as std
            # and L2 norm are proportional).
            resid = resid / torch.sqrt(torch.tensor(resid.shape[-1]))
        
        # Continue with topk activation and the rest of the network
        feature_acts = self.sparse_encoder_block(resid)
        resid = feature_acts
        
        assert (
            (feature_acts == 0).float().sum(dim=-1) >= (self.sparse_dim - self.topk)
        ).all()

        def apply_decoder_block(resid, block):
            out = block(resid)
            return out
            # # Divide by decoder output vector norms on forward pass
            # def apply_linear_layer(resid, linear_layer):
            #     assert linear_layer.weight.data.shape[1] == resid.shape[1]
            #     # W = linear_layer.weight / torch.norm(linear_layer.weight, dim=0, keepdim=True)
            #     W = linear_layer.weight
            #     out = resid @ W.T
            #     out += linear_layer.bias
            #     return out

            # if isinstance(block, nn.Sequential):
            #     linear_layer, activation = block[0], block[1]
            #     out = apply_linear_layer(resid, linear_layer)
            #     out = activation(out)
            # else:
            #     assert isinstance(block, nn.Linear)
            #     out = apply_linear_layer(resid, block)

            # return out

        resid = apply_decoder_block(resid, self.decoder_blocks[0])

        if len(self.decoder_blocks) > 1:
            for block in self.decoder_blocks[1:-1]:
                resid = apply_decoder_block(resid, block) + resid
            
            resid = apply_decoder_block(resid, self.decoder_blocks[-1])

        reconstructed = resid

        # MSE reconstruction loss
        mse_loss = (reconstructed.float() - x.float()).pow(2).mean()
        
        loss = mse_loss

        return {
            "loss": loss,
            "mse_loss": mse_loss,
            "feature_acts": feature_acts,
            "reconstructed": reconstructed,
        }

class ExperimentSAERes1(ResSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[4],
            sparse_dim_mult=16,
            decoder_dim_mults=[4],
            device=device,
            topk_init=32,
            topk_final=32,
            topk_decay_iter=2000,
            act_squeeze=0,
            weight_decay=1e-2,
            optimizer_type="sparse_adam",
            optimizer_config={
                "lr": 5e-4,
            }
        )

class ExperimentSAERes2(ResSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[4, 4],
            sparse_dim_mult=16,
            decoder_dim_mults=[4, 4],
            device=device,
            topk_init=32,
            topk_final=32,
            topk_decay_iter=2000,
            act_squeeze=0,
            weight_decay=1e-2,
            optimizer_type="sparse_adam",
            optimizer_config={
                "lr": 5e-4,
            }
        )

class ExperimentSAERes3(ResSAE):
    def __init__(self, act_size: int, device: str = "cpu"):
        super().__init__(
            act_size=act_size,
            encoder_dim_mults=[4, 4, 4],
            sparse_dim_mult=16,
            decoder_dim_mults=[4, 4, 4],
            device=device,
            topk_init=32,
            topk_final=32,
            topk_decay_iter=2000,
            act_squeeze=0,
            weight_decay=1e-2,
            optimizer_type="sparse_adam",
            optimizer_config={
                "lr": 5e-4,
            }
        )