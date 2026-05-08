import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct"

print("=" * 60)
print("Llama 70B NF4 smoke test")
print("=" * 60)

t0 = time.time()
print("[1/4] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print("  vocab_size: {}".format(tokenizer.vocab_size))
print("  eos_token: {}".format(tokenizer.eos_token))
print("  elapsed: {:.1f}s".format(time.time() - t0))

t1 = time.time()
print("[2/4] Building NF4 config...")
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
print("  load_in_4bit: {}".format(nf4_config.load_in_4bit))
print("  quant_type: {}".format(nf4_config.bnb_4bit_quant_type))
print("  elapsed: {:.1f}s".format(time.time() - t1))

t2 = time.time()
print("[3/4] Loading model NF4 (1-3 min expected)...")
torch.cuda.reset_peak_memory_stats()
mem_before = torch.cuda.memory_allocated() / 1e9
print("  GPU mem before: {:.2f} GB".format(mem_before))

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=nf4_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()

mem_after = torch.cuda.memory_allocated() / 1e9
mem_peak = torch.cuda.max_memory_allocated() / 1e9
print("  GPU mem after: {:.2f} GB".format(mem_after))
print("  GPU mem delta: {:.2f} GB".format(mem_after - mem_before))
print("  GPU mem peak: {:.2f} GB".format(mem_peak))
print("  elapsed: {:.1f}s".format(time.time() - t2))

print("[4/4] Model structure check...")
cfg = model.config
print("  num_hidden_layers: {}".format(cfg.num_hidden_layers))
print("  num_attention_heads: {}".format(cfg.num_attention_heads))
print("  num_key_value_heads: {}".format(cfg.num_key_value_heads))
print("  hidden_size: {}".format(cfg.hidden_size))
print("  head_dim: {}".format(cfg.hidden_size // cfg.num_attention_heads))
print("  max_position_embeddings: {}".format(cfg.max_position_embeddings))
print("  vocab_size: {}".format(cfg.vocab_size))

print("=" * 60)
print("Total elapsed: {:.1f}s".format(time.time() - t0))
print("Done")
