from mlsae.model import DeepSAE

model = DeepSAE(encoder_dim_mults=[1], sparse_dim_mult=2, decoder_dim_mults=[1], act_size=100)
model.save("test", save_to_s3=True)
