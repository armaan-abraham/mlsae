import torch
from mlsae.data import stream_training_chunks
from mlsae.config import data_cfg, DTYPES
from mlsae.model import DeepSAE
from mlsae.worker import generate_acts_from_tokens, preprocess_acts, get_baseline_mse
import transformer_lens

MAX_TOKENS = int(2e6)

iterator = stream_training_chunks(
    dataset_batch_size_entries=2,
    act_block_size_seqs=MAX_TOKENS // data_cfg.seq_len + 2,
    seed=42,
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

llm = transformer_lens.HookedTransformer.from_pretrained(
    data_cfg.model_name, device=device
).to(DTYPES[data_cfg.sae_dtype])

acts = generate_acts_from_tokens(llm, tokens, device)
acts = preprocess_acts(acts)

for arch in list(arch_to_model_id.keys()):
    sae = DeepSAE.load(arch, load_from_s3=False, model_id=arch_to_model_id[arch]).eval()

    sae.to(device)

    tokens = tokens.to(device)

    normalized_mse_list = []
    # Process in batches
    for start in range(0, acts.shape[0], data_cfg.sae_batch_size_tokens):
        acts_batch = acts[start : start + data_cfg.sae_batch_size_tokens]
        sae_mse = sae(acts_batch)[2]

        baseline_mse = get_baseline_mse(acts_batch)

        normalized_mse = sae_mse / baseline_mse

        normalized_mse_list.append(normalized_mse.item())

    normalized_mse_list = torch.tensor(normalized_mse_list)
    print(f"{arch=}, {normalized_mse_list.mean()=}")
