"""M5 — 256K+ SSD-offload projection (measurement-grounded).

NOT real inference. Projects SSD-tier KV access latency on top of
Day 2 measured facts, using the xhbm_emulator latency model.

Grounding facts (measured, Day 1-2 B200 sm_100):
- 128K M1.1: GPU peak 116.49 GB, decode 114.17 ms/tok, SSD I/O 0
- 256K M1.1: OOM (LM head 65 GB single tensor, NOT KV)
- per-token KV (theory, matches M1.2 32K=10.49GB exactly):
  80 layers x 2(K+V) x 8 kv_heads x 128 head_dim x 2 bytes = 327680 B/tok

Scenario: single-conv long context. KV that exceeds HBM budget is
offloaded to SSD tier. Each decode step reads the full KV once
(attention over all past tokens). We project the SSD read latency
contribution per token.

Honest disclosure: emulator models throughput + PCIe + write-amp only.
NOT modeled: queueing, thermal, GC, FTL. Real measurement = PoC stage
with Solidigm hardware.
"""
from src.xhbm_emulator import LatencyModel, load_defaults

# --- Grounding constants (measured + theory) ---
PER_TOK_KV_BYTES = 80 * 2 * 8 * 128 * 2   # 327680 B/tok, verified vs M1.2
WEIGHTS_GB = 39.58                          # NF4 70B, Day 1 measured
B200_USABLE_GB = 178.0                      # Day 2 OOM log: 178.35 usable
WORKSPACE_GB = 10.0                         # activation + CUDA workspace
DECODE_BASE_MS = 114.17                     # 128K measured decode latency
GB = 1e9

def kv_gb(ctx):
    return ctx * PER_TOK_KV_BYTES / GB

def hbm_kv_budget_gb():
    return B200_USABLE_GB - WEIGHTS_GB - WORKSPACE_GB

def project(ctx, lm):
    total_kv = kv_gb(ctx)
    budget = hbm_kv_budget_gb()
    ssd_kv_gb = max(0.0, total_kv - budget)
    ssd_bytes = ssd_kv_gb * GB
    # Each decode token: attention reads all past KV. SSD-resident
    # portion must be streamed in. Sequential read (KV blocks contiguous).
    if ssd_bytes > 0:
        r = lm.seq_read_latency_us(int(ssd_bytes))
        ssd_read_ms = r.total_us / 1000.0
    else:
        ssd_read_ms = 0.0
    return {
        "ctx": ctx,
        "total_kv_gb": round(total_kv, 2),
        "hbm_budget_gb": round(budget, 2),
        "ssd_kv_gb": round(ssd_kv_gb, 2),
        "ssd_read_ms_per_tok": round(ssd_read_ms, 2),
        "projected_decode_ms": round(DECODE_BASE_MS + ssd_read_ms, 2),
        "fits_hbm_alone": ssd_kv_gb == 0.0,
    }

def main():
    cfgs = load_defaults()
    lm_p5520 = LatencyModel(cfgs["p5520"])   # Active TLC
    lm_p5316 = LatencyModel(cfgs["p5316"])   # Archive QLC

    print("=" * 78)
    print("M5 — 256K+ SSD-offload projection (measurement-grounded, NOT real inference)")
    print("=" * 78)
    print(f"per-token KV: {PER_TOK_KV_BYTES} B ({PER_TOK_KV_BYTES/1e6:.3f} MB)")
    print(f"HBM KV budget: {hbm_kv_budget_gb():.1f} GB (178 usable - {WEIGHTS_GB} weights - {WORKSPACE_GB} workspace)")
    print(f"decode base latency (128K measured): {DECODE_BASE_MS} ms/tok")
    print()

    for ctx in [128000, 256000, 512000, 1000000]:
        print(f"--- {ctx//1000}K context ---")
        for name, lm in [("D7-P5520 (Active TLC)", lm_p5520),
                         ("D5-P5316 (Archive QLC)", lm_p5316)]:
            p = project(ctx, lm)
            tag = "HBM-alone fits" if p["fits_hbm_alone"] else f"SSD offload {p['ssd_kv_gb']} GB"
            print(f"  {name:24s}: KV={p['total_kv_gb']:.1f}GB | {tag}")
            print(f"  {'':24s}  decode {DECODE_BASE_MS} -> {p['projected_decode_ms']} ms/tok (+{p['ssd_read_ms_per_tok']} ms SSD)")
        print()

    print("HONEST DISCLOSURE:")
    print("- 256K KV alone (83.9 GB) + weights (39.6 GB) = 123.5 GB < 178 GB usable.")
    print("  256K KV FITS in HBM. The 256K OOM was the LM head (65 GB single")
    print("  tensor), fixable with num_logits_to_keep=1 -- NOT a KV problem.")
    print("- SSD offload value appears at 512K+ where KV alone exceeds HBM budget.")
    print("- Emulator: throughput + PCIe + write-amp only. NOT queue/thermal/GC/FTL.")
    print("- Real measurement = PoC stage with Solidigm hardware.")

if __name__ == "__main__":
    main()
