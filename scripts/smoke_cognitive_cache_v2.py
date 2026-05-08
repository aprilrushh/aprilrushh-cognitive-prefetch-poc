"""Cognitive Cache 9-5 smoke: 9 modes, multi-trial, 128 token gen."""
import sys
import time
import statistics

import torch

sys.path.insert(0, "/home/ubuntu/cognitive-prefetch-poc")

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DynamicCache
from src.cognitive_cache import CognitiveCache

MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct"
PROMPT = "Explain Modern Hopfield Networks in one sentence."
MAX_NEW = 128
N_TRIALS = 3


def sequential_pred(layer_idx, layers):
    return [layer_idx]


def make_top_k_pred(k):
    def predictor(layer_idx, layers):
        return list(range(layer_idx, min(layer_idx + k, len(layers))))
    return predictor


def skip_pred(layer_idx, layers):
    # selective: only prefetch when layer_idx is multiple of 80 (i.e., never inside a layer pass)
    # effectively: prefetch only first layer occasionally
    if layer_idx == 0:
        return [0]
    return []


def half_pred(layer_idx, layers):
    # prefetch every other layer
    if layer_idx % 2 == 0:
        return [layer_idx]
    return []


def measure(model, tokenizer, cache_factory, label, n_trials=N_TRIALS):
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    elapseds = []
    snippet = None
    n_prefetches_total = None

    for trial in range(n_trials):
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
        elapseds.append(time.time() - t0)
        if trial == 0:
            text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            snippet = text[:120].replace(chr(10), " | ")
            if hasattr(cache, "_total_layer_prefetches"):
                n_prefetches_total = cache._total_layer_prefetches

    mean = statistics.mean(elapseds)
    stdev = statistics.stdev(elapseds) if len(elapseds) > 1 else 0.0
    print("[{}]".format(label))
    print("  trials: {}  mean: {:.3f}s  stdev: {:.3f}s  tok/s: {:.2f}".format(
        n_trials, mean, stdev, MAX_NEW / mean))
    if n_prefetches_total is not None:
        print("  layer-prefetches (trial 1): {}".format(n_prefetches_total))
    print("  text: {}".format(snippet))
    print("")
    return mean, stdev, snippet


def main():
    print("=" * 70)
    print("Cognitive Cache 9-5 smoke (9 modes, N={} trials, gen={} tokens)".format(N_TRIALS, MAX_NEW))
    print("=" * 70)

    t0 = time.time()
    print("[setup] Loading Llama 70B NF4...")
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
        return CognitiveCache(predictor=None, mode="hybrid", config=model.config)
    results["T3_hybrid_none"] = measure(model, tokenizer, f3, "T3 Cognitive hybrid predictor=None")

    def f4():
        return CognitiveCache(predictor=sequential_pred, mode="hybrid", config=model.config)
    results["T4_hybrid_seq"] = measure(model, tokenizer, f4, "T4 Cognitive hybrid sequential")

    def f5():
        return CognitiveCache(predictor=make_top_k_pred(5), mode="hybrid", config=model.config)
    results["T5_hybrid_top5"] = measure(model, tokenizer, f5, "T5 Cognitive hybrid top-5")

    def f6():
        return CognitiveCache(predictor=sequential_pred, mode="replace", config=model.config)
    results["T6_replace_seq"] = measure(model, tokenizer, f6, "T6 Cognitive REPLACE sequential")

    def f7():
        return CognitiveCache(predictor=make_top_k_pred(5), mode="replace", config=model.config)
    results["T7_replace_top5"] = measure(model, tokenizer, f7, "T7 Cognitive REPLACE top-5")

    def f8():
        return CognitiveCache(predictor=skip_pred, mode="replace", config=model.config)
    results["T8_replace_skip"] = measure(model, tokenizer, f8, "T8 Cognitive REPLACE skip (almost no prefetch)")

    def f9():
        return CognitiveCache(predictor=half_pred, mode="replace", config=model.config)
    results["T9_replace_half"] = measure(model, tokenizer, f9, "T9 Cognitive REPLACE half (every other layer)")

    print("=" * 70)
    print("Sanity: text consistency across all 9 modes")
    snippets = [v[2] for v in results.values()]
    all_match = all(s == snippets[0] for s in snippets)
    print("  all texts identical: {}".format(all_match))
    if not all_match:
        for k, (_, _, s) in results.items():
            print("  {}: {}".format(k, s))

    print("")
    print("=" * 70)
    print("Timing summary")
    base = results["T1_vanilla"][0]
    for k, (m, s, _) in results.items():
        rel = m / base
        marker = ""
        if "skip" in k or "replace_seq" in k or "replace_half" in k:
            marker = "  <-- selective"
        print("  {:<22s}  {:.3f}s +-{:.3f}  ({:.2f}x vanilla){}".format(k, m, s, rel, marker))

    print("")
    print("Total elapsed: {:.1f}s".format(time.time() - t0))
    print("Done")


if __name__ == "__main__":
    main()
