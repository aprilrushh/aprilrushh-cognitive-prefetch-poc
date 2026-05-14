"""M1.4 — Concurrent decode + RAM->HBM promote (separate CUDA Stream).

Anchor section 13 anchor 1 *active interference* test:
  M1.2 was passive resident interference (+1.0% verified).
  M1.4 is **active transfer interference** -- RAM->HBM promote happening
  *during* decode, on a separate CUDA Stream.

Hypothesis: torch.cuda.Stream isolation -> decode latency unchanged
  (compute on default stream, transfer on promote stream, PCIe-HBM
   bandwidth not the bottleneck of 78 ms/tok decode).

Setup:
- NF4 Llama 70B (~39.58 GB)
- conv_active: 32K Sherlock prefill (M1.1 reproduce)
- Prefill K/V templates kept on GPU (~10.49 GB)
- 1 idle conv KV bf16 on CPU pinned RAM (~10.49 GB)
- Each round: clone template -> fresh round cache, then decode 32 tokens

Variants:
- A (baseline): pure decode, no concurrent promote
- B (concurrent): decode + concurrent RAM->HBM promote on
                  torch.cuda.Stream, started at decode begin

Timing: torch.cuda.Event (stream-aware, does NOT sync promote stream).

HBM peak (variant B):
  weight 39.58 + template 10.49 + round_cache 10.49 + promote 10.49
  + activation ~5 = ~76 GB (within 80 GB H100)

Compliance:
- v1.6: no CCP attention hook, pure NF4 + bf16 native
- v1.4 section E: dual-evidence path, PDF Lambda 자료 그대로
- v1.4 section D fix 3: DynamicCache .layers[L].keys/.values API
"""
import time, json, gc, statistics
from pathlib import Path
import torch, psutil
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache

t0 = time.time()
proc = psutil.Process()

def log(msg):
    gpu = torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0
    ram = proc.memory_info().rss/1e9
    print(f"[{time.time()-t0:7.2f}s | GPU {gpu:5.1f}GB | RAM {ram:6.1f}GB] {msg}", flush=True)

# Config
MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"
ACTIVE_CTX = 32000
N_DECODE = 32
N_WARMUP = 2
N_MEASURE = 3
SHERLOCK_PATH = "data/corpus/raw/A_Study_in_Scarlet.txt"
M1_1_BASELINE_MS = 78.60
RESULTS_DIR = Path("results/m1_4")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = RESULTS_DIR / f"m1_4_{int(time.time())}.json"

log("Phase 3.5 M1.4 -- Concurrent decode + RAM->HBM promote (separate CUDA Stream)")
log(f"  active prefill = {ACTIVE_CTX} tokens, decode = {N_DECODE} tokens/round")
log(f"  rounds: {N_WARMUP} warmup + {N_MEASURE} measure per variant")
log(f"  variants: A=baseline (no promote), B=concurrent promote (1 conv RAM->HBM)")
log("=" * 80)

# Step 1: load NF4 Llama 70B
log("Step 1: load NF4 Llama 70B")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
load_s = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb,
    device_map={"": 0},
    dtype=torch.bfloat16,
)
model.eval()
log(f"  load done in {time.time()-load_s:.1f}s")

cfg = model.config
n_layers = cfg.num_hidden_layers
n_kv_heads = cfg.num_key_value_heads
head_dim = cfg.hidden_size // cfg.num_attention_heads
bytes_per_conv = 2 * n_layers * n_kv_heads * ACTIVE_CTX * head_dim * 2
log(f"  config: layers={n_layers}, kv_heads={n_kv_heads}, head_dim={head_dim}")
log(f"  KV bytes per conv = {bytes_per_conv/1e9:.2f} GB")

# Step 2: active conv prefill
log("Step 2: active conv prefill from Sherlock")
text = open(SHERLOCK_PATH).read()
ids = tokenizer(text, return_tensors="pt").input_ids
log(f"  raw tokens = {ids.shape[1]}")
if ids.shape[1] >= ACTIVE_CTX:
    ids = ids[:, :ACTIVE_CTX]
else:
    n_repeat = (ACTIVE_CTX // ids.shape[1]) + 1
    ids = ids.repeat(1, n_repeat)[:, :ACTIVE_CTX]
ids = ids.to("cuda")
log(f"  prefill input shape: {ids.shape}")

with torch.no_grad():
    pf_s = time.time()
    out = model(ids, use_cache=True)
    torch.cuda.synchronize()
    pf_e = time.time()
prefill_cache = out.past_key_values
log(f"  prefill done in {pf_e-pf_s:.2f}s, throughput {ACTIVE_CTX/(pf_e-pf_s):.0f} tok/s")
log(f"  GPU after prefill: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# v1.4 section D fix 3: DynamicCache .layers[L].keys/.values
prefill_K_gpu = [prefill_cache.layers[L].keys for L in range(n_layers)]
prefill_V_gpu = [prefill_cache.layers[L].values for L in range(n_layers)]
del out, prefill_cache
torch.cuda.empty_cache()
log(f"  template extracted, GPU now: {torch.cuda.memory_allocated()/1e9:.1f} GB")

seed_id = ids[:, -1:].clone()
del ids
torch.cuda.empty_cache()

# Step 3: idle conv KV on CPU pinned RAM
log("Step 3: allocate 1 idle conv KV on CPU pinned RAM (M1.3 pattern)")
ram_K = [torch.empty((1, n_kv_heads, ACTIVE_CTX, head_dim), dtype=torch.bfloat16, pin_memory=True) for _ in range(n_layers)]
ram_V = [torch.empty((1, n_kv_heads, ACTIVE_CTX, head_dim), dtype=torch.bfloat16, pin_memory=True) for _ in range(n_layers)]
for L in range(n_layers):
    ram_K[L].fill_(0.001)
    ram_V[L].fill_(0.002)
log(f"  RAM KV allocated, {bytes_per_conv/1e9:.2f} GB pinned (synthetic random)")

# Step 4: decode round helper
def decode_one_round(use_concurrent_promote, tag):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    cache = DynamicCache()
    for L in range(n_layers):
        cache.update(prefill_K_gpu[L].clone(), prefill_V_gpu[L].clone(), L)
    torch.cuda.synchronize()

    current_id = seed_id.clone()

    promote_stream = None
    promote_start_evt = None
    promote_end_evt = None
    promoted_K = None
    promoted_V = None

    if use_concurrent_promote:
        promote_stream = torch.cuda.Stream()
        promote_start_evt = torch.cuda.Event(enable_timing=True)
        promote_end_evt = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(promote_stream):
            promote_start_evt.record(promote_stream)
            promoted_K = [k.to("cuda", non_blocking=True) for k in ram_K]
            promoted_V = [v.to("cuda", non_blocking=True) for v in ram_V]
            promote_end_evt.record(promote_stream)

    per_token_ms = []
    with torch.no_grad():
        for tk in range(N_DECODE):
            tk_start = torch.cuda.Event(enable_timing=True)
            tk_end = torch.cuda.Event(enable_timing=True)
            tk_start.record()
            out = model(current_id, past_key_values=cache, use_cache=True)
            tk_end.record()
            tk_end.synchronize()
            per_token_ms.append(tk_start.elapsed_time(tk_end))
            current_id = out.logits[:, -1:].argmax(dim=-1)
            cache = out.past_key_values

    promote_ms = None
    if use_concurrent_promote:
        promote_end_evt.synchronize()
        promote_ms = promote_start_evt.elapsed_time(promote_end_evt)

    gpu_peak = torch.cuda.max_memory_allocated() / 1e9

    del cache, current_id, out
    if use_concurrent_promote:
        del promoted_K, promoted_V, promote_stream
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return per_token_ms, promote_ms, gpu_peak

# Step 5+6: measure
log("=" * 80)
log("Step 5: measure Variant A (baseline)")
results_a = {"per_token_ms_rounds": [], "gpu_peak_gb_rounds": []}
for r in range(N_WARMUP + N_MEASURE):
    tag = "warmup" if r < N_WARMUP else "measure"
    per_tok, _, peak = decode_one_round(False, tag)
    avg = sum(per_tok)/len(per_tok)
    log(f"  Variant A round {r+1}/{N_WARMUP+N_MEASURE} [{tag}]: avg={avg:.2f} ms/tok, peak {peak:.1f} GB")
    if tag == "measure":
        results_a["per_token_ms_rounds"].append(per_tok)
        results_a["gpu_peak_gb_rounds"].append(peak)

log("=" * 80)
log("Step 6: measure Variant B (concurrent RAM->HBM promote)")
results_b = {"per_token_ms_rounds": [], "promote_ms_rounds": [], "gpu_peak_gb_rounds": []}
for r in range(N_WARMUP + N_MEASURE):
    tag = "warmup" if r < N_WARMUP else "measure"
    per_tok, prom_ms, peak = decode_one_round(True, tag)
    avg = sum(per_tok)/len(per_tok)
    log(f"  Variant B round {r+1}/{N_WARMUP+N_MEASURE} [{tag}]: avg={avg:.2f} ms/tok, promote {prom_ms:.1f} ms, peak {peak:.1f} GB")
    if tag == "measure":
        results_b["per_token_ms_rounds"].append(per_tok)
        results_b["promote_ms_rounds"].append(prom_ms)
        results_b["gpu_peak_gb_rounds"].append(peak)

# Step 7: stats
def flat_stats(rounds):
    flat = [t for r in rounds for t in r]
    return {
        "n": len(flat),
        "avg_ms": sum(flat)/len(flat),
        "median_ms": statistics.median(flat),
        "min_ms": min(flat),
        "max_ms": max(flat),
        "stdev_ms": statistics.stdev(flat) if len(flat) > 1 else 0.0,
    }

def token_group_during_after(rounds, promote_ms_per_round):
    during, after = [], []
    for r_idx, per_tok in enumerate(rounds):
        prom = promote_ms_per_round[r_idx]
        cum = 0.0
        for ms in per_tok:
            cum += ms
            if cum <= prom:
                during.append(ms)
            else:
                after.append(ms)
    return {
        "during_promote": {
            "n": len(during),
            "avg_ms": (sum(during)/len(during)) if during else None,
        },
        "after_promote": {
            "n": len(after),
            "avg_ms": (sum(after)/len(after)) if after else None,
        }
    }

a_stats = flat_stats(results_a["per_token_ms_rounds"])
b_stats = flat_stats(results_b["per_token_ms_rounds"])
b_group = token_group_during_after(results_b["per_token_ms_rounds"], results_b["promote_ms_rounds"])

delta_a = (a_stats["avg_ms"] - M1_1_BASELINE_MS) / M1_1_BASELINE_MS * 100
delta_b = (b_stats["avg_ms"] - M1_1_BASELINE_MS) / M1_1_BASELINE_MS * 100
delta_ba = (b_stats["avg_ms"] - a_stats["avg_ms"]) / a_stats["avg_ms"] * 100

log("=" * 80)
log("Step 7: RESULTS")
log(f"  Variant A baseline: avg={a_stats['avg_ms']:.2f} ms/tok, median={a_stats['median_ms']:.2f}, sigma={a_stats['stdev_ms']:.2f}")
log(f"     delta vs M1.1 (78.60): {delta_a:+.2f}%")
log(f"  Variant B concurrent: avg={b_stats['avg_ms']:.2f} ms/tok, median={b_stats['median_ms']:.2f}, sigma={b_stats['stdev_ms']:.2f}")
log(f"     delta vs M1.1 (78.60): {delta_b:+.2f}%")
log(f"     delta vs Variant A: {delta_ba:+.2f}%  <- key metric (stream isolation)")
log(f"  Variant B during-promote tokens: n={b_group['during_promote']['n']}, avg={b_group['during_promote']['avg_ms']}")
log(f"  Variant B after-promote tokens: n={b_group['after_promote']['n']}, avg={b_group['after_promote']['avg_ms']}")
log(f"  Promote durations (ms): {results_b['promote_ms_rounds']}")
log(f"  GPU peak A: {max(results_a['gpu_peak_gb_rounds']):.1f} GB | B: {max(results_b['gpu_peak_gb_rounds']):.1f} GB")

output = {
    "measurement": "M1.4 concurrent decode + RAM->HBM promote (separate CUDA Stream)",
    "phase": "3.5",
    "anchor_version": "v1.6 (CCP excluded, NF4 + bf16 only)",
    "model": MODEL_NAME,
    "active_ctx": ACTIVE_CTX,
    "n_decode": N_DECODE,
    "n_warmup": N_WARMUP,
    "n_measure": N_MEASURE,
    "model_config": {
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "bytes_per_conv_gb": bytes_per_conv/1e9,
    },
    "M1_1_baseline_ms_per_tok": M1_1_BASELINE_MS,
    "variant_A_baseline": {
        **a_stats,
        "delta_vs_M1_1_pct": delta_a,
        "gpu_peak_gb": max(results_a["gpu_peak_gb_rounds"]),
        "per_token_ms_rounds": results_a["per_token_ms_rounds"],
    },
    "variant_B_concurrent": {
        **b_stats,
        "delta_vs_M1_1_pct": delta_b,
        "delta_vs_A_pct": delta_ba,
        "gpu_peak_gb": max(results_b["gpu_peak_gb_rounds"]),
        "per_token_ms_rounds": results_b["per_token_ms_rounds"],
        "promote_ms_rounds": results_b["promote_ms_rounds"],
        "token_group_analysis": b_group,
    },
    "limitations": [
        "single Python process, no multi-tenant interference",
        "cloud VM (virtio-blk), no real NVMe -- but M1.4 doesn't touch SSD anyway",
        "bf16 KV (no V-only quant) -- v1.4 section E dual evidence path",
        "synthetic random idle KV (filled with 0.001/0.002)",
        "promote is 1 conv only (M1.3 method-B pattern); multi-conv concurrent promote not tested here",
        "torch.cuda.Event for per-token timing -- stream-aware (does not sync promote stream)",
    ],
}
with open(OUT_PATH, "w") as f:
    json.dump(output, f, indent=2)
log(f"JSON dumped to {OUT_PATH}")
log("=" * 80)
log("M1.4 done")
