import transformer_lens
from transformers import AutoTokenizer

model = transformer_lens.HookedTransformer.from_pretrained("pythia-14m", device="cpu")

print(model.tokenizer)


tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

