# %%
from transformers import AutoTokenizer

# Load tokenizers
gptneox_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
# %%

# Example string to tokenize
example_str = "Hello, world!"

# Tokenize with GPT-NeoX
tokens_gptneox = gptneox_tokenizer.encode(example_str, add_special_tokens=False)
print("GPT-NeoX token IDs:", tokens_gptneox)

# Tokenize with GPT-2
tokens_gpt2 = gpt2_tokenizer.encode(example_str, add_special_tokens=False)
print("GPT-2 token IDs:", tokens_gpt2)
