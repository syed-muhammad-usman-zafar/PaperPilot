from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from evaluate import load as load_metric
from sentence_transformers import SentenceTransformer

# 1) Pre‑cache GPT-2
_ = GPT2LMHeadModel.from_pretrained("gpt2")
_ = GPT2TokenizerFast.from_pretrained("gpt2")

# 2) Pre‑cache BERTScore metric
_ = load_metric("bertscore")

# 3) Pre‑cache your sentence‑transformer
_ = SentenceTransformer("paraphrase-MiniLM-L6-v2")

print("All models and metrics are now cached.")
