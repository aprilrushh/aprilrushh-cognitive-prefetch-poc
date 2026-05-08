"""smoke_long_context.py - 9-6b-epsilon."""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.expanduser("~/cognitive-prefetch-poc"))
sys.path.insert(0, os.path.expanduser("~/cognitive-prefetch-poc/src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.cognitive_cache_v2 import CognitiveCache_v2
from src.attention_patch import (
    apply_cognitive_attention_patch,
    revert_cognitive_attention_patch,
    get_sparse_stats,
)
from predictor import CognitivePredictor


CONTEXT_TARGET = 8192


def make_long_prompt(tokenizer, target_tokens):
    base = (
        "The Modern Hopfield Network, introduced by Ramsauer et al. in 2021, "
        "extends classical Hopfield networks to continuous states by replacing "
        "the dot-product similarity with softmax. The energy function becomes "
        "log-sum-exp, and the retrieval rule is exactly attention. This unifies "
        "associative memory with transformers. The capacity scales exponentially "
        "with the dimension. Sparse Hopfield variants (Hu 2023) use entmax "
        "instead of softmax, achieving exact retrieval at high beta. "
    )
    full = base
    while len(tokenizer(full).input_ids) < target_tokens:
        full = full + base
    ids = tokenizer(full, return_tensors="pt").input_ids[0]
    if ids.shape[0] > target_tokens:
        ids = ids[:target_tokens]
    return ids.unsqueeze(0)


def run_round(model, tokenizer, input_ids, max_new, n_layers, use_sparse, label):
    print("--- round: {} (use_sparse={}) prefill={} ---".format(
        label, use_sparse, int(input_ids.shape[1])))
    cache = CognitiveCache_v2(
        predictor=CognitivePredictor(),
        num_layers=n_layers, device="cuda", enable_async=True,
    )
    apply_cognitive_attention_patch(use_sparse=use_sparse)
    print("  GPU after cache+patch: {:.2f} GB".format(torch.cuda.memory_allocated() / 1e9))
    input_ids_cuda = input_ids.to("cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids_cuda, max_new_tokens=max_new, do_sample=False,
            past_key_values=cache, return_dict_in_generate=True, use_cache=True,
        )
    torch.cuda.synchronize()
    gen_t = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    print("  GPU peak during round: {:.2f} GB".format(peak))
    torch.cuda.reset_peak_memory_stats()
    new_ids = out.sequences[0, input_ids.shape[1]:].tolist()
    union_sizes = [e[1] for e in cache.stats["indices_log"]]
    sparse_stats = dict(get_sparse_stats())
    cache_repr = repr(cache)
    revert_cognitive_attention_patch()
    print("  time: {:.2f}s ({:.2f} tok/s decode)".format(gen_t, max_new / max(gen_t - 0.001, 0.001)))
    print("  new ids:  {}".format(new_ids))
    print("  decoded:  {}".format(repr(tokenizer.decode(new_ids))))
    print("  cache:    {}".format(cache_repr))
    print("  sparse:   {}".format(sparse_stats))
    if union_sizes:
        us_min = min(union_sizes); us_max = max(union_sizes); us_mean = sum(union_sizes) / len(union_sizes)
        print("  union:    n={} min={} max={} mean={:.1f}".format(len(union_sizes), us_min, us_max, us_mean))
    d2h = cache.stats["pcie_d2h_bytes"] / 1e9
    h2d = cache.stats["pcie_h2d_bytes"] / 1e9
    sub = cache.stats["pcie_subset_bytes"] / 1e9
    print("  PCIe (real measured bytes):")
    print("    D2H mirror   : {:.3f} GB (every update -> CPU)".format(d2h))
    print("    H2D K_full   : {:.3f} GB (every hook -> GPU staging, current impl)".format(h2d))
    print("    Subset (K+V) : {:.3f} GB (would-be PCIe in CPU-only mode 9-6d)".format(sub))
    if h2d > 0:
        ideal_save = (1 - sub / h2d) * 100
        print("    ideal save (subset / K_full): {:.2f}%".format(ideal_save))
    return {"new_ids": new_ids, "time": gen_t, "union_sizes": union_sizes, "sparse_stats": sparse_stats}


def main():
    print("=== smoke 9-6b-epsilon: long context (target {} tok) ===".format(CONTEXT_TARGET))
    model_id = "meta-llama/Llama-3.1-70B-Instruct"

    print("[step 1] load Llama 70B NF4 (eager)")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print("  loaded in {:.1f}s, GPU: {:.2f} GB".format(
        time.time() - t0, torch.cuda.memory_allocated() / 1e9))

    print("[step 2] build long prompt")
    input_ids = make_long_prompt(tokenizer, CONTEXT_TARGET)
    print("  prefill_len: {}".format(int(input_ids.shape[1])))

    max_new = 6
    print("[step 3] round FULL")
    r_full = run_round(model, tokenizer, input_ids, max_new, n_layers, False, "FULL")
    print("  GPU after full: {:.2f} GB".format(torch.cuda.memory_allocated() / 1e9))
    torch.cuda.empty_cache()

    print("[step 4] round SPARSE")
    r_sparse = run_round(model, tokenizer, input_ids, max_new, n_layers, True, "SPARSE")

    print()
    print("[step 5] comparison")
    n_match = sum(1 for a, b in zip(r_full["new_ids"], r_sparse["new_ids"]) if a == b)
    n_total = len(r_full["new_ids"])
    match_rate = 100.0 * n_match / max(n_total, 1)
    print("  greedy match: {}/{} ({:.1f}%)".format(n_match, n_total, match_rate))
    print("  full ids:   {}".format(r_full["new_ids"]))
    print("  sparse ids: {}".format(r_sparse["new_ids"]))
    print("  full time:  {:.2f}s / sparse time: {:.2f}s".format(r_full["time"], r_sparse["time"]))
    if r_sparse["union_sizes"]:
        u_mean = sum(r_sparse["union_sizes"]) / len(r_sparse["union_sizes"])
        prefill_len = int(input_ids.shape[1])
        save_pct = (1 - u_mean / prefill_len) * 100
        print("  union mean: {:.1f}".format(u_mean))
        print("  PCIe save estimate: {:.2f}% (subset {} / full prefill {})".format(
            save_pct, int(u_mean), prefill_len))

    print("[step 6] verdict")
    if match_rate >= 98:
        verdict = "A: Gate 4 PASS (>= 98% top-1 match)"
    elif match_rate >= 50:
        verdict = "B: PARTIAL ({:.1f}% match) - beta sweep needed".format(match_rate)
    else:
        verdict = "C: WEAK ({:.1f}% match) - hypothesis revision".format(match_rate)
    print("  verdict: {}".format(verdict))
    print("Done")


if __name__ == "__main__":
    main()
