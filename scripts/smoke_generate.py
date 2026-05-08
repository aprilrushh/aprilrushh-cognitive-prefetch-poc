import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct"

print("=" * 60)
print("Llama 70B generate smoke test (8-3a + 8-3b)")
print("=" * 60)

t0 = time.time()
print("[setup] Loading tokenizer + model NF4...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=nf4_config,
    device_map="auto",
    dtype=torch.bfloat16,
)
model.eval()
print("  setup elapsed: {:.1f}s".format(time.time() - t0))
print("  GPU mem after load: {:.2f} GB".format(torch.cuda.memory_allocated() / 1e9))

print("")
print("=" * 60)
print("[8-3a] Short-context generate (prompt ~50 tok, gen 64 tok)")
print("=" * 60)

prompt_a = "Explain in one paragraph what a Modern Hopfield Network is and why it relates to attention mechanism in transformers."
inputs_a = tokenizer(prompt_a, return_tensors="pt").to(model.device)
prompt_len_a = inputs_a.input_ids.shape[1]
print("  prompt_len: {} tokens".format(prompt_len_a))

torch.cuda.synchronize()
ta = time.time()
with torch.no_grad():
    out_a = model.generate(
        **inputs_a,
        max_new_tokens=64,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
torch.cuda.synchronize()
elapsed_a = time.time() - ta
new_tokens_a = out_a.shape[1] - prompt_len_a
print("  generated: {} tokens".format(new_tokens_a))
print("  elapsed: {:.2f}s".format(elapsed_a))
print("  throughput: {:.2f} tok/s".format(new_tokens_a / elapsed_a))
print("  GPU mem after gen: {:.2f} GB".format(torch.cuda.memory_allocated() / 1e9))
print("  --- generated text (last 200 chars) ---")
text_a = tokenizer.decode(out_a[0][prompt_len_a:], skip_special_tokens=True)
print("  " + text_a[-200:].replace("\n", " "))

print("")
print("=" * 60)
print("[8-3b] Long-context prefill (ctx ~32K) + 1 tok gen")
print("=" * 60)

# Synthetic 32K-ish context: repeat a paragraph to fill ~32000 tokens
seed = "The transformer architecture revolutionized natural language processing by introducing self-attention. " * 8
seed_ids = tokenizer(seed, return_tensors="pt").input_ids
seed_len = seed_ids.shape[1]
target_ctx = 32000
n_repeats = target_ctx // seed_len + 1
long_text = seed * n_repeats
inputs_b = tokenizer(long_text, return_tensors="pt", truncation=True, max_length=target_ctx).to(model.device)
prompt_len_b = inputs_b.input_ids.shape[1]
print("  prompt_len: {} tokens".format(prompt_len_b))

torch.cuda.reset_peak_memory_stats()
mem_before_b = torch.cuda.memory_allocated() / 1e9
torch.cuda.synchronize()
tb = time.time()
with torch.no_grad():
    out_b = model.generate(
        **inputs_b,
        max_new_tokens=1,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
torch.cuda.synchronize()
elapsed_b = time.time() - tb
mem_after_b = torch.cuda.memory_allocated() / 1e9
mem_peak_b = torch.cuda.max_memory_allocated() / 1e9
print("  prefill + 1 gen elapsed: {:.2f}s".format(elapsed_b))
print("  prefill throughput: {:.2f} tok/s".format(prompt_len_b / elapsed_b))
print("  GPU mem before: {:.2f} GB".format(mem_before_b))
print("  GPU mem after: {:.2f} GB".format(mem_after_b))
print("  GPU mem peak: {:.2f} GB".format(mem_peak_b))
print("  KV cache delta (approx): {:.2f} GB".format(mem_after_b - mem_before_b))

print("")
print("=" * 60)
print("Total elapsed: {:.1f}s".format(time.time() - t0))
print("Done")
