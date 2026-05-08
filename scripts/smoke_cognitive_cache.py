"""Cognitive Cache smoke test: 5 cache modes side-by-side."""
import sys
import time

import torch

sys.path.insert(0, "/home/ubuntu/cognitive-prefetch-poc")

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DynamicCache
from src.cognitive_cache import CognitiveCache

MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct"
PROMPT = "Explain Modern Hopfield Networks in one sentence."
MAX_NEW = 32


def sequential_predictor(layer_idx, layers):
    return [layer_idx]


def make_top_k_predictor(k):
    def predictor(layer_idx, layers):
        return list(range(layer_idx, min(layer_idx + k, len(layers))))
    return predictor


def measure(model, tokenizer, cache_factory, label):
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    cache = cache_factory()

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            past_key_values=cache,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    n_new = out.shape[1] - inputs.input_ids.shape[1]
    text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    snippet = text[:120].replace(chr(10), " | ")

    print("[{}]".format(label))
    print("  elapsed: {:.3f}s  throughput: {:.2f} tok/s".format(elapsed, n_new / elapsed))
    if hasattr(cache, "prefetch_history"):
        n_calls = len(cache.prefetch_history)
        print("  prefetch calls: {}".format(n_calls))
        if n_calls > 0:
            print("  first 2 calls: {}".format(cache.prefetch_history[:2]))
    print("  text: {}".format(snippet))
    print("")
    return elapsed, snippet


def main():
    print("=" * 60)
    print("Cognitive Cache smoke test (Step 9-4)")
    print("=" * 60)

    t0 = time.time()
    print("[setup] Loading tokenizer + Llama 70B NF4...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    nf4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=nf4,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.eval()
    print("  setup elapsed: {:.1f}s".format(time.time() - t0))
    print("  GPU mem after load: {:.2f} GB".format(torch.cuda.memory_allocated() / 1e9))
    print("")

    results = {}

    def f1():
        return DynamicCache(config=model.config)
    results["T1_vanilla"] = measure(model, tokenizer, f1, "T1 vanilla DynamicCache (no offload)")

    def f2():
        return DynamicCache(config=model.config, offloading=True, offload_only_non_sliding=False)
    results["T2_offloaded"] = measure(model, tokenizer, f2, "T2 HF DynamicCache offloading=True")

    def f3():
        return CognitiveCache(predictor=None, config=model.config, log_prefetch=True)
    results["T3_cognitive_none"] = measure(model, tokenizer, f3, "T3 CognitiveCache predictor=None")

    def f4():
        return CognitiveCache(predictor=sequential_predictor, config=model.config, log_prefetch=True)
    results["T4_cognitive_seq"] = measure(model, tokenizer, f4, "T4 CognitiveCache sequential predictor")

    def f5():
        return CognitiveCache(predictor=make_top_k_predictor(5), config=model.config, log_prefetch=True)
    results["T5_cognitive_top5"] = measure(model, tokenizer, f5, "T5 CognitiveCache top-5 predictor")

    print("=" * 60)
    print("Sanity check: text consistency across all 5 modes")
    snippets = [v[1] for v in results.values()]
    all_match = all(s == snippets[0] for s in snippets)
    print("  all texts identical: {}".format(all_match))
    if not all_match:
        for k, (_, s) in results.items():
            print("  {}: {}".format(k, s))

    print("")
    print("=" * 60)
    print("Timing summary")
    base = results["T1_vanilla"][0]
    for k, (e, _) in results.items():
        rel = e / base
        print("  {:<28s}  {:.3f}s  ({:.2f}x vanilla)".format(k, e, rel))

    print("")
    print("Total elapsed: {:.1f}s".format(time.time() - t0))
    print("Done")


if __name__ == "__main__":
    main()
