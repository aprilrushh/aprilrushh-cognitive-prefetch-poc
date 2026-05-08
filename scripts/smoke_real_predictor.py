"""smoke_real_predictor.py - 9-6b-gamma-2.

Replace DummyPredictor with real CognitivePredictor (default config:
similarity=dot, separation=entmax, beta=2.0, alpha=1.5, top_k=256).
Wrap with LoggingCognitivePredictor to capture confidence_level,
multi_update_triggered, fallback per call. First real LLM signal of
the Q_N ~= Q_{N+1} hypothesis (anchor section 11.3).
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.expanduser("~/cognitive-prefetch-poc"))
sys.path.insert(0, os.path.expanduser("~/cognitive-prefetch-poc/src"))

import torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.cognitive_cache_v2 import CognitiveCache_v2
from src.attention_patch import (
    apply_cognitive_attention_patch,
    revert_cognitive_attention_patch,
    is_patched,
)
from predictor import CognitivePredictor


class LoggingCognitivePredictor:
    """Captures PredictionResult metadata per predict() call."""
    def __init__(self, base):
        self.base = base
        self.log = []
        self.errors = 0

    def predict(self, current_q, target_keys, target_values=None, beta_override=None):
        try:
            result = self.base.predict(
                current_q, target_keys, target_values, beta_override=beta_override
            )
            entry = {
                "union_size": int(result.predicted_indices.shape[1]) if hasattr(result, "predicted_indices") else 0,
                "confidence_level": str(getattr(result, "confidence_level", "")),
                "multi_update_triggered": bool(getattr(result, "multi_update_triggered", False)),
                "used_full_prefetch_fallback": bool(getattr(result, "used_full_prefetch_fallback", False)),
                "top1_score": float(getattr(result, "top1_score", 0.0)) if hasattr(result, "top1_score") else None,
            }
            self.log.append(entry)
            return result
        except Exception as e:
            self.errors += 1
            self.log.append({"error": str(e)[:120]})
            raise


def main():
    print("=== smoke 9-6b-gamma-2: real CognitivePredictor ===")
    model_id = "meta-llama/Llama-3.1-70B-Instruct"

    print("[step 1] load Llama 70B NF4...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb,
        dtype=torch.float16, device_map="cuda",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print("  loaded in {:.1f}s, num_layers={}".format(time.time() - t0, n_layers))
    print("  GPU used: {:.2f} GB".format(torch.cuda.memory_allocated() / 1e9))

    print("[step 2] build real CognitivePredictor + Logging wrapper")
    base = CognitivePredictor()
    print("  base config:", base.config)
    log_pred = LoggingCognitivePredictor(base)

    print("[step 3] build cache_v2")
    cache = CognitiveCache_v2(
        predictor=log_pred, num_layers=n_layers,
        device="cuda", enable_async=True,
    )

    print("[step 4] apply patch + short generate")
    apply_cognitive_attention_patch()
    prompt = "The Modern Hopfield Network is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    prefill_len = int(inputs.input_ids.shape[1])
    torch.cuda.synchronize()
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=6, do_sample=False,
            past_key_values=cache, return_dict_in_generate=True, use_cache=True,
        )
    torch.cuda.synchronize()
    gen_t = time.time() - t1
    new_tokens = int(out.sequences.shape[1]) - prefill_len
    text = tokenizer.decode(out.sequences[0], skip_special_tokens=True)
    print("  generate: {:.2f}s ({:.2f} tok/s)".format(gen_t, new_tokens / gen_t))
    print("  text:", repr(text))
    print("  cache:", cache)
    print("  predictor errors:", log_pred.errors)

    print()
    print("[step 5] confidence + fallback distribution analysis")
    log = log_pred.log
    n = len(log)
    print("  total predict calls: {}".format(n))
    if n == 0:
        print("  NO PREDICT CALLS - debug needed")
        return

    valid = [e for e in log if "error" not in e]
    print("  valid: {} / errors: {}".format(len(valid), n - len(valid)))

    conf_counter = Counter(e["confidence_level"] for e in valid)
    print("  confidence_level distribution:")
    for level in ["high", "medium", "low", ""]:
        c = conf_counter.get(level, 0)
        pct = 100.0 * c / max(len(valid), 1)
        print("    {:6s}: {:4d} ({:5.1f}%)".format(level or "(empty)", c, pct))

    multi = sum(1 for e in valid if e["multi_update_triggered"])
    fallback = sum(1 for e in valid if e["used_full_prefetch_fallback"])
    print("  multi_update_triggered: {} ({:.1f}%)".format(multi, 100.0 * multi / max(len(valid), 1)))
    print("  used_full_prefetch_fallback: {} ({:.1f}%)".format(fallback, 100.0 * fallback / max(len(valid), 1)))

    union_sizes = [e["union_size"] for e in valid]
    if union_sizes:
        us_min, us_max = min(union_sizes), max(union_sizes)
        us_mean = sum(union_sizes) / len(union_sizes)
        print("  union_size: min={} max={} mean={:.1f}".format(us_min, us_max, us_mean))

    top1s = [e["top1_score"] for e in valid if e.get("top1_score") is not None]
    if top1s:
        ts_min, ts_max = min(top1s), max(top1s)
        ts_mean = sum(top1s) / len(top1s)
        print("  top1_score: min={:.3f} max={:.3f} mean={:.3f}".format(ts_min, ts_max, ts_mean))
    else:
        print("  top1_score: not exposed on PredictionResult")

    print()
    print("[step 6] hypothesis assessment (anchor 11.3)")
    high_pct = 100.0 * conf_counter.get("high", 0) / max(len(valid), 1)
    if high_pct >= 80:
        verdict = "A: hypothesis STRONG (Q_N approx Q_{N+1} robust)"
    elif high_pct >= 30:
        verdict = "B: hypothesis PARTIAL (layer-dependent)"
    elif multi >= len(valid) * 0.5:
        verdict = "C: hypothesis WEAK but multi_update salvages"
    else:
        verdict = "D: hypothesis BROKEN (random level) - Path 2 fallback"
    print("  verdict: {}".format(verdict))

    revert_cognitive_attention_patch()
    print("  is_patched after revert:", is_patched())
    print("Done")


if __name__ == "__main__":
    main()
