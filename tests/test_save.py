from mlsae.model.model import DeepSAE


def test_save():
    model = DeepSAE(
        encoder_dim_mults=[1], sparse_dim_mult=2, decoder_dim_mults=[1], act_size=100
    )
    model.save("test", save_to_s3=True, model_id="test")
