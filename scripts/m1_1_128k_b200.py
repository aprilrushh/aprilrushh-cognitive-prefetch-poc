"""M1.1-128K — Decode SSD I/O byte trace at 128K context on B200.

Extension of M1.1 (32K, anchor section 15 M1) to 128K context.
B200 192GB HBM unlocks the H100 80GB ceiling - deck v6.3 section 9
'NixlCache @ 128K - Blocked by H100 80GB' becomes Measured.

Setup:
- NF4 Llama 70B (~39.58 GB weights)
- Sherlock 4 volumes concatenated (need ~128K tokens)
- 32 decode tokens, per-token I/O measurement
- Expected GPU peak: ~80 GB (39.58 weight + ~42 KV at 128K + activation)
- Verdict: process read = 0 bytes (steady state) - same anchor section 15 M1
"""
import time, json, os
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

CTX = 128000
log(f"M1.1-128K -- Decode SSD I/O at {CTX} context on B200")
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
log(f"loaded in {load_time:.2f}s, GPU {torch.cuda.memory_allocated()/1e9:.2f} GB")

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")

log(f"Prepare {CTX} tokens (Sherlock 4 volumes concat)")
books = [
    "data/corpus/raw/A_Study_in_Scarlet.txt",
    "data/corpus/raw/The_Sign_of_the_Four.txt",
    "data/corpus/raw/The_Adventures_of_Sherlock_Holmes.txt",
    "data/corpus/raw/The_Hound_of_the_Baskervilles.txt",
]
full_text = ""
for b in books:
    full_text += Path(b).read_text(encoding="utf-8", errors="replace") + "\n\n"
all_ids = tok.encode(full_text, add_special_tokens=False)
log(f"  total Sherlock corpus tokens: {len(all_ids)}")
if len(all_ids) < CTX:
    log(f"  WARNING: only {len(all_ids)} tokens available, requested {CTX}")
chunk = all_ids[:CTX]
input_ids = torch.tensor([chunk], device="cuda")
log(f"  prefill input: {input_ids.shape[1]} tokens")

import gc
gc.collect(); torch.cuda.synchronize(); time.sleep(1.0)

log("Baseline I/O snapshot")
b_rcnt, b_wcnt, b_rb, b_wb = proc_io()
b_dr, b_dw = diskstats_vda1()
log(f"  proc baseline: rd={b_rb:,} B ({b_rcnt} ops), wr={b_wb:,} B")

log(f"Prefill {CTX} (forward, use_cache=True)")
t_p = time.time()
with torch.no_grad():
    out = model(input_ids, use_cache=True)
torch.cuda.synchronize()
prefill_time = time.time() - t_p
past_kv = out.past_key_values
log(f"  prefill done in {prefill_time:.2f}s ({CTX/prefill_time:.0f} tok/s)")
log(f"  GPU after prefill: {torch.cuda.memory_allocated()/1e9:.2f} GB")
log(f"  GPU peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

a_rcnt, a_wcnt, a_rb, a_wb = proc_io()
prefill_io = {
    "proc_read_bytes": a_rb - b_rb, "proc_read_count": a_rcnt - b_rcnt,
    "proc_write_bytes": a_wb - b_wb, "proc_write_count": a_wcnt - b_wcnt,
}
log(f"  prefill ΔI/O proc: rd={prefill_io['proc_read_bytes']/1e6:.2f} MB ({prefill_io['proc_read_count']} ops)")

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
}
peak_gpu = torch.cuda.max_memory_allocated()/1e9

def vstr(b):
    if b == 0: return "ZERO"
    if b < 1024: return f"NEAR-ZERO ({b}B)"
    if b < 1024*1024: return f"SMALL ({b/1024:.1f}KB)"
    return f"LARGE ({b/1e6:.2f}MB)"

log(f"DECODE AGGREGATE ({N} tokens) at {CTX} context")
log(f"  avg latency: {agg['avg_latency_ms']:.1f} ms/tok")
log(f"  proc read: {vstr(agg['proc_rd_bytes'])} ({agg['proc_rd_count']} ops)")
log(f"  proc write: {vstr(agg['proc_wr_bytes'])} ({agg['proc_wr_count']} ops)")
log(f"  vda1 read: {vstr(agg['vda1_rd_bytes'])}")
log(f"  GPU peak: {peak_gpu:.2f} GB")

# Verdict: tokens 1-31 (skip first-token init noise)
steady = trace[1:]
steady_rd = sum(t["proc_rd_bytes"] for t in steady)
steady_wr = sum(t["proc_wr_bytes"] for t in steady)
claim_ok = (steady_rd < 10*1024 and steady_wr < 10*1024)
verdict = ("VERIFIED ✅ anchor section 15 M1 at 128K -- decode critical path is SSD-free"
           if claim_ok else "REVIEW ⚠ -- non-trivial process I/O during decode steady state")
log(verdict)

result = {
    "measurement": f"M1.1-128K -- Decode SSD I/O at {CTX} context on B200",
    "date": "2026-05-27",
    "platform": "B200 SXM6 192GB sm_100",
    "anchor_ref": "v1.0 section 15 M1, deck v6.3 section 9 (128K Blocked-by-H100 unblock)",
    "spec": {
        "model": "meta-llama/Llama-3.1-70B-Instruct",
        "quant": "nf4 + bf16 compute_dtype + double_quant",
        "context_tokens": CTX,
        "n_decode_tokens": N,
        "workload": "Sherlock 4 volumes concatenated, first 128K tokens"
    },
    "load": {"time_s": round(load_time, 2), "gpu_gb": round(torch.cuda.memory_allocated()/1e9, 2)},
    "prefill": {
        "time_s": round(prefill_time, 2),
        "tokens_per_s": round(CTX/prefill_time, 1),
        **prefill_io,
    },
    "decode": {
        "n_tokens": N,
        "aggregate": agg,
        "gpu_peak_gb": round(peak_gpu, 2),
        "verdict": verdict,
        "per_token_trace": trace,
    },
}
out_file = f"results/phase3/m1_1_decode_128k_b200.json"
Path("results/phase3").mkdir(parents=True, exist_ok=True)
Path(out_file).write_text(json.dumps(result, indent=2))
log(f"saved: {out_file} ({Path(out_file).stat().st_size} bytes)")
log("M1.1-128K DONE")
