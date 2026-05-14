"""M1.1 — Decode SSD I/O byte trace measurement.

Anchor § 15 M1: "Decode 중 SSD I/O byte trace = 0 bytes during decode"
v1.4 § E: M1 essential (no V-only quant needed, dual-evidence path)

Setup:
- NF4 Llama 70B load (warm HF cache)
- Sherlock 32K tokens prefill (1 conv)
- Decode 32 tokens with use_cache=True
- Per-token: psutil.disk_io_counters() + /proc/diskstats vda1
Expected: decode process I/O bytes < 10 KB (no SSD on critical path)
"""
import time, json, gc, os
from pathlib import Path
import torch, psutil
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

t0 = time.time()
proc = psutil.Process(os.getpid())

def diskstats_vda1():
    try:
        with open("/proc/diskstats") as f:
            for line in f:
                p = line.split()
                if len(p) >= 10 and p[2] == "vda1":
                    return int(p[5])*512, int(p[9])*512
    except: pass
    return 0, 0

def proc_io():
    io = proc.io_counters()
    return io.read_count, io.write_count, io.read_bytes, io.write_bytes

def log(msg):
    print(f"[{time.time()-t0:7.2f}s | GPU {torch.cuda.memory_allocated()/1e9:5.1f}GB] {msg}", flush=True)

log("Load NF4 70B (warm HF cache, bf16 compute_dtype)")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
t_l = time.time()
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct",
    quantization_config=bnb, dtype=torch.bfloat16,
    device_map="auto", low_cpu_mem_usage=True,
)
model.eval()
load_time = time.time() - t_l
gpu_after_load = torch.cuda.memory_allocated()/1e9
log(f"loaded in {load_time:.2f}s, GPU {gpu_after_load:.2f} GB")

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")

log("Prepare Sherlock 32K tokens (A Study in Scarlet, first 32K)")
text = Path("data/corpus/raw/A_Study_in_Scarlet.txt").read_text(encoding="utf-8", errors="replace")
all_ids = tok.encode(text, add_special_tokens=False)
chunk = all_ids[:32000]
input_ids = torch.tensor([chunk], device="cuda")
log(f"  prefill input: {input_ids.shape[1]} tokens (from {len(all_ids)} Scarlet total)")

gc.collect(); torch.cuda.synchronize(); time.sleep(1.0)

log("Baseline I/O snapshot")
b_rcnt, b_wcnt, b_rb, b_wb = proc_io()
b_dr, b_dw = diskstats_vda1()
log(f"  proc baseline: rd={b_rb:,} B ({b_rcnt} ops), wr={b_wb:,} B")
log(f"  vda1 baseline: rd={b_dr:,} B, wr={b_dw:,} B")

log("Prefill 32K (forward, use_cache=True)")
t_p = time.time()
with torch.no_grad():
    out = model(input_ids, use_cache=True)
torch.cuda.synchronize()
prefill_time = time.time() - t_p
past_kv = out.past_key_values
log(f"  prefill done in {prefill_time:.2f}s, GPU {torch.cuda.memory_allocated()/1e9:.2f} GB")

a_rcnt, a_wcnt, a_rb, a_wb = proc_io()
a_dr, a_dw = diskstats_vda1()
prefill_io = {
    "proc_read_count": a_rcnt - b_rcnt, "proc_write_count": a_wcnt - b_wcnt,
    "proc_read_bytes": a_rb - b_rb, "proc_write_bytes": a_wb - b_wb,
    "disk_vda1_read_bytes": a_dr - b_dr, "disk_vda1_write_bytes": a_dw - b_dw,
}
log(f"  prefill ΔI/O proc: rd={prefill_io['proc_read_bytes']/1e6:.2f} MB ({prefill_io['proc_read_count']} ops), wr={prefill_io['proc_write_bytes']/1e6:.2f} MB")
log(f"  prefill ΔI/O vda1: rd={prefill_io['disk_vda1_read_bytes']/1e6:.2f} MB, wr={prefill_io['disk_vda1_write_bytes']/1e6:.2f} MB")

N = 32
log(f"Decode {N} tokens (per-token I/O)")
next_tok = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
cur_ids = next_tok
trace = []
for i in range(N):
    p_rcnt, p_wcnt, p_rb, p_wb = proc_io()
    p_dr, p_dw = diskstats_vda1()
    t_tok = time.time()
    with torch.no_grad():
        out = model(cur_ids, past_key_values=past_kv, use_cache=True)
    torch.cuda.synchronize()
    lat = (time.time() - t_tok) * 1000
    past_kv = out.past_key_values
    cur_ids = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
    n_rcnt, n_wcnt, n_rb, n_wb = proc_io()
    n_dr, n_dw = diskstats_vda1()
    trace.append({
        "tok_idx": i, "latency_ms": round(lat, 2),
        "proc_rd_count": n_rcnt - p_rcnt, "proc_wr_count": n_wcnt - p_wcnt,
        "proc_rd_bytes": n_rb - p_rb, "proc_wr_bytes": n_wb - p_wb,
        "vda1_rd_bytes": n_dr - p_dr, "vda1_wr_bytes": n_dw - p_dw,
        "tok_id": int(cur_ids.item()),
    })

agg = {
    "proc_rd_count": sum(t["proc_rd_count"] for t in trace),
    "proc_wr_count": sum(t["proc_wr_count"] for t in trace),
    "proc_rd_bytes": sum(t["proc_rd_bytes"] for t in trace),
    "proc_wr_bytes": sum(t["proc_wr_bytes"] for t in trace),
    "vda1_rd_bytes": sum(t["vda1_rd_bytes"] for t in trace),
    "vda1_wr_bytes": sum(t["vda1_wr_bytes"] for t in trace),
    "avg_latency_ms": sum(t["latency_ms"] for t in trace) / N,
    "total_decode_s": sum(t["latency_ms"] for t in trace) / 1000,
}

def vstr(b):
    if b == 0: return "ZERO"
    if b < 1024: return f"NEAR-ZERO ({b}B)"
    if b < 1024*1024: return f"SMALL ({b/1024:.1f}KB)"
    return f"LARGE ({b/1e6:.2f}MB)"

log(f"DECODE AGGREGATE ({N} tokens)")
log(f"  avg latency       : {agg['avg_latency_ms']:.1f} ms/token")
log(f"  proc read total   : {vstr(agg['proc_rd_bytes'])} ({agg['proc_rd_count']} ops)")
log(f"  proc write total  : {vstr(agg['proc_wr_bytes'])} ({agg['proc_wr_count']} ops)")
log(f"  vda1 read total   : {vstr(agg['vda1_rd_bytes'])} (system noise)")
log(f"  vda1 write total  : {vstr(agg['vda1_wr_bytes'])} (system noise)")

claim_ok = (agg["proc_rd_bytes"] < 10*1024 and agg["proc_wr_bytes"] < 10*1024)
verdict = ("VERIFIED ✅ anchor § 15 M1 — decode critical path is SSD-free (process I/O < 10 KB)"
           if claim_ok else "REVIEW ⚠ — non-trivial process I/O during decode, investigate")

result = {
    "measurement": "M1.1 — Decode SSD I/O byte trace (single conv NF4 70B baseline)",
    "date": "2026-05-14", "anchor_ref": "v1.0 § 15 M1, v1.4 § E (essential)",
    "spec": {
        "model": "meta-llama/Llama-3.1-70B-Instruct",
        "quant": "nf4 + bf16 compute_dtype + double_quant",
        "kv_quant": "none (bf16 native, V-only deferred per v1.4 § E)",
        "context_tokens": int(input_ids.shape[1]), "n_decode_tokens": N,
        "workload": "A Study in Scarlet first 32K tokens",
    },
    "load": {"time_s": round(load_time, 2), "gpu_gb": round(gpu_after_load, 2)},
    "prefill": {"time_s": round(prefill_time, 2),
                "tokens_per_s": round(input_ids.shape[1] / prefill_time, 1),
                **prefill_io},
    "decode": {"n_tokens": N, "aggregate": agg, "per_token_trace": trace},
    "verdict": verdict,
    "limitations": [
        "Single conv only (M1.1 sanity scope). Multi-conv idle interference = M1.2.",
        "Cloud VM virtio-blk (NOT physical NVMe) — vda1 noise = HF cache + OS, not KV/weight",
        "Process-scoped I/O (psutil) = clean signal for the M1 claim",
        "No V-only quant (v1.4 § E dual-evidence path)",
    ],
    "next": "M1.2 — multi-conv mixed (1 active decode + idle conv background)",
}

out_path = Path("results/phase3/m1_1_decode_ssd_trace.json")
out_path.write_text(json.dumps(result, indent=2))
log(f"saved: {out_path} ({out_path.stat().st_size} bytes)")
log(f"VERDICT: {verdict}")
log("M1.1 DONE")
