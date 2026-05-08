"""smoke_sparse_attention.py - 9-6b-delta-alpha.

Two-round generate: full attention vs sparse attention with same model,
same prompt. Verify text identity (greedy decode equality).
"""
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
    is_patched,
    get_sparse_stats,
)
from predictor import CognitivePredictor


def run_generate(model, tokenizer, prompt, max_new_tokens, n_layers, use_sparse, label):
    print("--- round: {} (use_sparse={}) ---".format(label, use_sparse))
    cache = CognitiveCache_v2(
        predictor=CognitivePredictor(),
        num_layers=n_layers, device="cuda", enable_async=True,
    )
    apply_cognitive_attention_patch(use_sparse=use_sparse)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    prefill_len = int(inputs.input_ids.shape[1])
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            past_key_values=cache, return_dict_in_generate=True, use_cache=True,
        )
    torch.cuda.synchronize()
    gen_t = time.time() - t0
    text = tokenizer.decode(out.sequences[0], skip_special_tokens=True)
    new_tok_ids = out.sequences[0, prefill_len:].tolist()
    sparse_stats = dict(get_sparse_stats())
    cache_repr = repr(cache)
    revert_cognitive_attention_patch()
    print("  time: {:.2f}s ({:.2f} tok/s)".format(gen_t, max_new_tokens / gen_t))
    print("  text:", repr(text))
    print("  new_token_ids:", new_tok_ids)
    print("  cache:", cache_repr)
    print("  sparse_stats:", sparse_stats)
    return {
        "text": text, "new_tok_ids": new_tok_ids,
        "time": gen_t, "sparse_stats": sparse_stats,
    }


def main():
    print("=== smoke 9-6b-delta-alpha: sparse attention same-output ===")
    model_id = "meta-llama/Llama-3.1-70B-Instruct"
    print("[step 1] load Llama 70B NF4")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb,
        dtype=torch.float16, device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print("  loaded in {:.1f}s, num_layers={}".format(time.time() - t0, n_layers))
    attn_impl = getattr(model.config, "_attn_implementation", "?")
    print("  attn_implementation:", attn_impl)

    prompt = "The Modern Hopfield Network is"
    max_new = 6

    print("[step 2] round FULL")
    r_full = run_generate(model, tokenizer, prompt, max_new, n_layers, False, "FULL")

    print("[step 3] round SPARSE")
    r_sparse = run_generate(model, tokenizer, prompt, max_new, n_layers, True, "SPARSE")

    print("[step 3.5] indices_log diagnosis (last cache from sparse round)")
    # cache is local to run_generate; rerun a small sparse smoke for diagnosis
    diag_cache = CognitiveCache_v2(predictor=CognitivePredictor(), num_layers=n_layers, device="cuda", enable_async=True)
    apply_cognitive_attention_patch(use_sparse=True)
    diag_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        model.generate(**diag_inputs, max_new_tokens=2, do_sample=False, past_key_values=diag_cache, use_cache=True)
    revert_cognitive_attention_patch()
    log = diag_cache.stats["indices_log"]
    print("  total entries: {}".format(len(log)))
    print("  first 3:", log[:3])
    print("  last 3 :", log[-3:])
    if log:
        sample = log[-1]
        print("  last entry tuple ({} elems): {}".format(len(sample), sample))

    print()
    print("[step 4] comparison")
    print("  text identical: {}".format(r_full["text"] == r_sparse["text"]))
    print("  greedy match:   {}".format(r_full["new_tok_ids"] == r_sparse["new_tok_ids"]))
    print("  full new ids:   {}".format(r_full["new_tok_ids"]))
    print("  sparse new ids: {}".format(r_sparse["new_tok_ids"]))
    print("  sparse activations: {} (expected ~ decode_steps * (n_layers-1))".format(
        r_sparse["sparse_stats"]["activations"]
    ))
    print("  sparse skips: {} (subset None at first decode token's layer 0 etc)".format(
        r_sparse["sparse_stats"]["skips"]
    ))
    print("  full activations: {} (expected 0)".format(
        r_full["sparse_stats"]["activations"]
    ))

    if r_full["new_tok_ids"] == r_sparse["new_tok_ids"]:
        print("  VERDICT: sparse mechanism EQUIVALENT to full attention (greedy)")
    else:
        n_match = sum(1 for a, b in zip(r_full["new_tok_ids"], r_sparse["new_tok_ids"]) if a == b)
        print("  VERDICT: sparse DIVERGED at token {} ({}/{} match)".format(
            n_match, n_match, len(r_full["new_tok_ids"])
        ))

    print("Done")


if __name__ == "__main__":
    main()
