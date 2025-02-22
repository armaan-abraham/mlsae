import torch
from mlsae.data import stream_training_chunks
from mlsae.config import data_cfg
from mlsae.model import DeepSAE
from mlsae.worker import generate_acts_from_tokens, preprocess_acts, get_baseline_mse

MAX_TOKENS = int(1e6)

iterator = stream_training_chunks(
    dataset_batch_size_entries=2,
    act_block_size_seqs=MAX_TOKENS // data_cfg.seq_len + 2,
)

arch_to_model_id = {
    "2-2_64": "badly-valid-crab",
    "2-2_128": "kindly-fleet-gopher",
    "2-2_256": "sadly-nearby-dragon",
    "0-0_64": "barely-fond-dane",
    "0-0_128": "subtly-unique-goose",
    "0-0_256": "surely-able-grouse",
}

chunks = []
num_tokens = 0


while num_tokens < MAX_TOKENS:
    chunk = next(iterator)
    chunks.append(chunk)
    num_tokens += chunk.numel()

tokens = torch.cat(chunks)

device = "cuda"


for arch in list(arch_to_model_id.keys())[:1]:
    sae = DeepSAE.load(
        arch, load_from_s3=True, model_id=arch_to_model_id[arch]
    ).eval()

    sae.to(device)

    tokens = tokens.to(device)

    acts = generate_acts_from_tokens(sae, tokens)
    print(f"{acts.shape=}")
    acts = preprocess_acts(acts)
    print(f"{acts.shape=}")

    normalized_mse_list = []
    # Process in batches
    for start in range(0, acts.shape[0], data_cfg.sae_batch_size_tokens):
        acts_batch = acts[start : start + data_cfg.sae_batch_size_tokens]
        sae_mse = sae(acts_batch)[2]
        print(f"{sae_mse=}")

        baseline_mse = get_baseline_mse(acts_batch)
        print(f"{baseline_mse=}")

        normalized_mse = sae_mse / baseline_mse
        print(f"{normalized_mse=}")

        normalized_mse_list.append(normalized_mse)

    normalized_mse_list = torch.tensor(normalized_mse_list)
    print(f"{normalized_mse_list.mean()=}")
    print(f"{normalized_mse_list.std()=}")
