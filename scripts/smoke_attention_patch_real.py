"""smoke_attention_patch_real.py - 9-6b-beta-2-gamma.

Real Llama 70B NF4 + CognitiveCache_v2 + DummyPredictor + attention_patch
+ short generate. Validates that the post-attention hook fires per-layer
per-decode-token, fills cache.stats counters, and does not break text.
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.expanduser("~/cognitive-prefetch-poc"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.cognitive_cache_v2 import CognitiveCache_v2
from src.attention_patch import (
    apply_cognitive_attention_patch,
    revert_cognitive_attention_patch,
    is_patched,
)


class DummyPredictor:
    """Returns first min(270, seq_len) token indices. Mocks CognitivePredictor."""
    def predict(self, current_q, target_keys, target_values, **kw):
        S = target_keys.shape[2]
        B = target_keys.shape[0]
        K = min(270, S)
        indices = torch.arange(K, device="cpu").unsqueeze(0).expand(B, K).contiguous().long()
        class _R: pass
        r = _R()
        r.predicted_indices = indices
        r.union_size = K
        r.confidence_level = "high"
        r.multi_update_triggered = False
        r.used_full_prefetch_fallback = False
        return r


def main():
    print("=== smoke 9-6b-beta-2-gamma: real LLM hook trigger ===")
    model_id = "meta-llama/Llama-3.1-70B-Instruct"

    print("[step 1] load Llama 70B NF4...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    load_t = time.time() - t0
    n_layers = model.config.num_hidden_layers
    print("  loaded in {:.1f}s, num_layers={}".format(load_t, n_layers))
    gpu_used = torch.cuda.memory_allocated() / 1e9
    print("  GPU used: {:.2f} GB".format(gpu_used))

    print("[step 2] build cache_v2 + DummyPredictor")
    cache = CognitiveCache_v2(
        predictor=DummyPredictor(),
        num_layers=n_layers,
        device="cuda",
        enable_async=True,
    )
    print("  initial:", cache)

    print("[step 3] apply attention patch")
    ok = apply_cognitive_attention_patch()
    print("  apply={} is_patched={}".format(ok, is_patched()))

    print("[step 4] short generate (6 new tokens)")
    prompt = "The Modern Hopfield Network is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    prefill_len = int(inputs.input_ids.shape[1])
    print("  prefill_len:", prefill_len)
    torch.cuda.synchronize()
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=6,
            do_sample=False,
            past_key_values=cache,
            return_dict_in_generate=True,
            use_cache=True,
        )
    torch.cuda.synchronize()
    gen_t = time.time() - t1
    new_tokens = int(out.sequences.shape[1]) - prefill_len
    print("  generate: {:.2f}s, new_tokens={}".format(gen_t, new_tokens))

    print("[step 5] inspect")
    text = tokenizer.decode(out.sequences[0], skip_special_tokens=True)
    print("  text: {}".format(repr(text)))
    print("  cache:", cache)
    n_pred = cache.stats["predicts"]
    n_pref = cache.stats["prefetches"]
    n_fall = cache.stats["fallbacks"]
    n_log = len(cache.stats["indices_log"])
    print("  predicts: {}".format(n_pred))
    print("  prefetches: {}".format(n_pref))
    print("  fallbacks: {}".format(n_fall))
    print("  indices_log entries: {}".format(n_log))
    if n_log > 0:
        print("  indices_log first 3:", cache.stats["indices_log"][:3])
        print("  indices_log last 3:", cache.stats["indices_log"][-3:])

    # expected: each decode token triggers hook on layers 0..n-2 (last skipped)
    # = new_tokens * (n_layers - 1) = 6 * 79 = 474
    expected_pred = new_tokens * (n_layers - 1)
    print("  expected predicts (lower bound): {}".format(expected_pred))
    margin = max(1, expected_pred // 4)
    assert n_pred >= expected_pred - margin, (
        "predicts {} below expected {}".format(n_pred, expected_pred)
    )
    assert n_fall == 0, "expected 0 fallbacks, got {}".format(n_fall)
    print("  ASSERTIONS PASS")

    print("[step 6] inspect a sample gpu_subset")
    sample_layer = 5
    sub = cache.get_layer_subset(sample_layer)
    if sub is not None:
        print("  layer {}: K shape={} V shape={} indices shape={}".format(
            sample_layer,
            tuple(sub["K"].shape),
            tuple(sub["V"].shape),
            tuple(sub["indices"].shape),
        ))

    print("[step 7] revert patch")
    revert_cognitive_attention_patch()
    print("  is_patched:", is_patched())

    print("Done")


if __name__ == "__main__":
    main()
