"""
NIXL Real Measurement Bench — D-3 production-grade statistical benchmark.

Modes:
  baseline   : transformers DynamicCache (NIXL 미통합, baseline)
  nixl_on    : CognitiveCache_v2 with NIXL hook (in-place 통합, 이미 측정 완료)
  nixl_b80   : NixlCache, hbm_budget_layers=80 (사실상 무제한)
  nixl_b60   : NixlCache, hbm_budget_layers=60 (25% 절감 목표)
  nixl_b40   : NixlCache, hbm_budget_layers=40 (50% 절감 목표)
  nixl_b20   : NixlCache, hbm_budget_layers=20 (75% 절감 목표)
"""
import argparse
import os
import sys
import time
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH      = "meta-llama/Llama-3.1-70B-Instruct"
CONTEXT_LEN     = 1024
DECODE_TOKENS   = 8
RUNS            = 30
WARMUPS         = 5
SEED            = 42


def build_prompt(tokenizer, target_len):
    # 긴 ctx 대응: 반복 횟수 충분히 크게
    repeats = max(300, target_len // 8 + 50)
    base = "The quick brown fox jumps over the lazy dog. " * repeats
    ids = tokenizer(base, return_tensors="pt").input_ids[0, :target_len]
    assert ids.shape[0] == target_len, f"prompt length mismatch: {ids.shape[0]} vs {target_len}"
    return ids.unsqueeze(0).to("cuda")


def load_model_and_cache(mode: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"[load] tokenizer + model (mode={mode})")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=nf4_config,
        device_map="cuda",
        low_cpu_mem_usage=True,
    )
    model.eval()

    if mode == "baseline":
        print("[load] baseline mode — DynamicCache by generate()")
    elif mode == "nixl_on":
        from src.nixl_plugin_manager import NixlSolidigmPlugin
        probe = NixlSolidigmPlugin(num_workers=1)
        print(f"[load] NIXL plugin probe OK: {type(probe).__name__}")
        del probe
    elif mode.startswith("nixl_b"):
        budget = int(mode[6:])
        print(f"[load] NixlCache mode — hbm_budget_layers={budget}")

    return tok, model


def make_cache_factory(mode: str, model_config):
    """매 run마다 새로운 cache 객체를 만드는 factory 반환. None이면 transformers 자동."""
    if mode in ("baseline", "nixl_on"):
        return None
    if mode.startswith("nixl_b"):
        budget = int(mode[6:])
        from src.nixl_cache import NixlCache
        def factory():
            return NixlCache(config=model_config, hbm_budget_layers=budget, nixl_workers=4)
        return factory
    return None


def measure_one_run(model, input_ids, gen_config, cache_factory=None, decode_tokens=None):
    if decode_tokens is None:
        decode_tokens = DECODE_TOKENS
    ev_start   = torch.cuda.Event(enable_timing=True)
    ev_prefill = torch.cuda.Event(enable_timing=True)
    ev_end     = torch.cuda.Event(enable_timing=True)

    past_kv = cache_factory() if cache_factory is not None else None
    if past_kv is not None and hasattr(past_kv, "reset"):
        past_kv.reset()

    # HBM trajectory: prefill 끝/decode 끝 시점 peak 별도 추적
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        ev_start.record()
        out = model.generate(
            input_ids,
            generation_config=gen_config,
            return_dict_in_generate=True,
            past_key_values=past_kv,
        )
        ev_prefill.record()
        torch.cuda.synchronize()
        hbm_after_prefill = torch.cuda.max_memory_allocated() / 1e9

        # Decode peak reset — decode-only peak를 따로 측정
        torch.cuda.reset_peak_memory_stats()

        past = out.past_key_values
        last = out.sequences[:, -1:]
        for _ in range(decode_tokens - 1):
            step = model(last, past_key_values=past, use_cache=True)
            past = step.past_key_values
            last = step.logits[:, -1:, :].argmax(dim=-1)
        ev_end.record()

    torch.cuda.synchronize()
    hbm_decode_peak = torch.cuda.max_memory_allocated() / 1e9
    ttft_ms   = ev_start.elapsed_time(ev_prefill)
    decode_ms = ev_prefill.elapsed_time(ev_end)

    # NixlCache stats 노출 (real_bench에서 evict가 진짜 호출됐는지 확인)
    if past_kv is not None and hasattr(past_kv, "get_nixl_summary"):
        s = past_kv.get_nixl_summary()
        print(f"    [cache] intercept_calls={s['intercept_calls']}, evictions={s['evictions']}, resident_peak={s['hbm_resident_peak']}, budget={s['hbm_budget_layers']}")

    return ttft_ms, decode_ms, hbm_after_prefill, hbm_decode_peak


def run_bench(mode: str, runs: int, warmups: int, reuse_cache: bool = False, ctx_len: int = None, decode_tokens: int = None):
    torch.manual_seed(SEED)
    tok, model = load_model_and_cache(mode)
    effective_ctx = ctx_len if ctx_len is not None else CONTEXT_LEN
    print(f"[ctx] using context length = {effective_ctx}")
    input_ids = build_prompt(tok, effective_ctx)
    cache_factory = make_cache_factory(mode, model.config)
    print(f"[mode] reuse_cache={reuse_cache}")

    # reuse 모드: 첫 cache를 미리 만들고 factory를 None으로 대체
    persistent_cache = None
    if reuse_cache and cache_factory is not None:
        persistent_cache = cache_factory()
        print(f"[cache] persistent {type(persistent_cache).__name__} created, reused across runs")
        cache_factory = lambda: persistent_cache

    from transformers import GenerationConfig
    eos = tok.eos_token_id if isinstance(tok.eos_token_id, int) else tok.eos_token_id[0]
    gen_config = GenerationConfig(
        max_new_tokens=1,
        do_sample=False,
        use_cache=True,
        pad_token_id=eos,
        eos_token_id=eos,
    )
    print(f"[gen_config] pad={eos}, eos={eos}, max_new_tokens=1")

    torch.cuda.reset_peak_memory_stats()
    hbm_model = torch.cuda.memory_allocated() / (1024**3)
    print(f"[hbm] model loaded = {hbm_model:.2f} GB")

    print(f"[warmup] {warmups} runs (excluded)")
    for w in range(warmups):
        measure_one_run(model, input_ids, gen_config, cache_factory, decode_tokens)
        print(f"  warmup {w+1}/{warmups} done")

    torch.cuda.reset_peak_memory_stats()

    print(f"[measure] {runs} runs, decode_tokens={decode_tokens or DECODE_TOKENS}")
    ttfts, decodes = [], []
    prefill_peaks, decode_peaks = [], []
    t0 = time.time()
    for i in range(runs):
        t, d, hpp, hdp = measure_one_run(model, input_ids, gen_config, cache_factory, decode_tokens)
        ttfts.append(t); decodes.append(d)
        prefill_peaks.append(hpp); decode_peaks.append(hdp)
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  run {i+1}/{runs}: ttft={t:6.1f}ms  decode={d:6.1f}ms  prefill_peak={hpp:.2f}GB  decode_peak={hdp:.2f}GB")
    elapsed = time.time() - t0

    hbm_peak = torch.cuda.max_memory_allocated() / (1024**3)
    avg_prefill_peak = float(np.mean(prefill_peaks))
    avg_decode_peak = float(np.mean(decode_peaks))
    print(f"[measure] wall-clock elapsed: {elapsed:.1f}s")
    print(f"[hbm] overall peak = {hbm_peak:.2f} GB (Δ from model = {hbm_peak-hbm_model:+.2f} GB)")
    print(f"[hbm] avg prefill peak = {avg_prefill_peak:.2f} GB,  avg decode-phase peak = {avg_decode_peak:.2f} GB")

    return np.array(ttfts), np.array(decodes), hbm_peak, hbm_model


def report(label, ttfts, decodes, runs, warmups, save_path=None, hbm_peak=None, hbm_model=None):
    totals = ttfts + decodes
    out = {
        "label":        label,
        "n_runs":       int(runs),
        "n_warmup":     int(warmups),
        "context_len":  CONTEXT_LEN,
        "decode_tokens": DECODE_TOKENS,
        "ttft_ms":      {"mean": float(ttfts.mean()),   "std": float(ttfts.std())},
        "decode_ms":    {"mean": float(decodes.mean()), "std": float(decodes.std())},
        "tpot_ms":      {"mean": float(decodes.mean() / DECODE_TOKENS)},
        "total_ms": {
            "mean": float(totals.mean()),
            "std":  float(totals.std()),
            "p50":  float(np.percentile(totals, 50)),
            "p95":  float(np.percentile(totals, 95)),
            "p99":  float(np.percentile(totals, 99)),
            "min":  float(totals.min()),
            "max":  float(totals.max()),
        },
        "raw_totals_ms": totals.tolist(),
        "hbm_peak_gb":  float(hbm_peak) if hbm_peak is not None else None,
        "hbm_model_gb": float(hbm_model) if hbm_model is not None else None,
    }

    print("\n" + "=" * 64)
    print(f"  {label}  (N={runs}, warmup={warmups} excluded)")
    print("=" * 64)
    print(f"  TTFT    : {out['ttft_ms']['mean']:7.1f} ms   (std {out['ttft_ms']['std']:.1f})")
    print(f"  Decode  : {out['decode_ms']['mean']:7.1f} ms   (std {out['decode_ms']['std']:.1f})")
    print(f"  TPOT    : {out['tpot_ms']['mean']:7.1f} ms / token")
    print(f"  Total   : {out['total_ms']['mean']:7.1f} ms   (std {out['total_ms']['std']:.2f})")
    print(f"          p50 = {out['total_ms']['p50']:7.1f} ms")
    print(f"          p95 = {out['total_ms']['p95']:7.1f} ms")
    print(f"          p99 = {out['total_ms']['p99']:7.1f} ms")
    print(f"          min = {out['total_ms']['min']:7.1f}   max = {out['total_ms']['max']:.1f}")
    if hbm_peak is not None:
        print(f"  HBM     : model={hbm_model:.2f} GB  peak={hbm_peak:.2f} GB  Δ={hbm_peak-hbm_model:+.2f} GB")
    print("=" * 64)

    if save_path:
        with open(save_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[save] {save_path}")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline","nixl_on","nixl_b80","nixl_b60","nixl_b40","nixl_b20"], required=True)
    parser.add_argument("--runs",   type=int, default=RUNS)
    parser.add_argument("--warmup", type=int, default=WARMUPS)
    parser.add_argument("--out",    default=None)
    parser.add_argument("--decode", type=int, default=None,
                        help="Decode token 수 override (기본: DECODE_TOKENS=8)")
    parser.add_argument("--ctx", type=int, default=None,
                        help="Context length override (기본: CONTEXT_LEN=1024)")
    parser.add_argument("--reuse-cache", action="store_true",
                        help="첫 run에서만 cache 생성, 이후 재사용 (hyperscaler production 패턴)")
    args = parser.parse_args()

    ttfts, decodes, hbm_peak, hbm_model = run_bench(args.mode, args.runs, args.warmup, reuse_cache=args.reuse_cache, ctx_len=args.ctx, decode_tokens=args.decode)
    report(args.mode.upper(), ttfts, decodes, args.runs, args.warmup, args.out,
           hbm_peak=hbm_peak, hbm_model=hbm_model)
    print("Done")
