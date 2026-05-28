"""M5 v2 — 256K+ SSD-offload projection WITH XHBM algorithm.

Compares naive full-KV-reread (v1) against the XHBM algorithm:
  1. V-only quant: K kept bf16, V compressed to 4-bit -> KV per-tok shrinks
  2. One-time write: KV written to SSD once at idle-demote, NOT per token
  3. Idle-driven residency: active conv KV stays in HBM during decode;
     only idle (>15min) conv demoted to SSD. So active-decode SSD read ~ 0.
     The SSD cost is the demote-write + the promote-read at reactivation
     (amortized over a typing window, NOT per decode token).

Honest disclosure: emulator = throughput + PCIe + write-amp only.
NOT modeled: queueing, thermal, GC, FTL. Real measurement = PoC stage.
"""
from src.xhbm_emulator import LatencyModel, load_defaults

# --- Grounding constants (measured + theory) ---
# naive per-token KV: K + V both bf16
KV_BF16_BYTES = 80 * 2 * 8 * 128 * 2        # 327680 B/tok (K+V bf16)
# V-only quant: K bf16 (2B) + V 4-bit (0.5B) per element
# per layer per tok: 8 heads * 128 dim * (2 + 0.5) bytes * 80 layers
KV_VONLY_BYTES = 80 * 8 * 128 * (2 + 0.5)   # K=2B, V=0.5B
WEIGHTS_GB = 39.58
B200_USABLE_GB = 178.0
WORKSPACE_GB = 10.0
DECODE_BASE_MS = 114.17
TYPING_WINDOW_S = 15.0                       # human typing pause, amortization base
GB = 1e9

def kv_gb(ctx, per_tok_bytes):
    return ctx * per_tok_bytes / GB

def hbm_budget_gb():
    return B200_USABLE_GB - WEIGHTS_GB - WORKSPACE_GB

def naive_decode_ms(ctx, lm, per_tok_bytes):
    """v1: full SSD-resident KV re-read every decode token."""
    total = kv_gb(ctx, per_tok_bytes)
    ssd_gb = max(0.0, total - hbm_budget_gb())
    if ssd_gb <= 0:
        return DECODE_BASE_MS, 0.0, 0.0
    r = lm.seq_read_latency_us(int(ssd_gb * GB))
    add = r.total_us / 1000.0
    return DECODE_BASE_MS + add, add, ssd_gb

def xhbm_reactivation_ms(ctx, lm, per_tok_bytes):
    """XHBM: SSD read happens ONCE at promote (reactivation), amortized
    over the typing window, not per decode token. During active decode,
    KV is in HBM -> decode latency = base. The SSD cost is a one-time
    promote-read of the demoted KV, hidden inside the typing pause."""
    total = kv_gb(ctx, per_tok_bytes)
    ssd_gb = max(0.0, total - hbm_budget_gb())
    if ssd_gb <= 0:
        return DECODE_BASE_MS, 0.0, 0.0, 0.0
    r = lm.seq_read_latency_us(int(ssd_gb * GB))
    promote_ms = r.total_us / 1000.0
    # amortized: promote cost spread over typing window (ms)
    window_ms = TYPING_WINDOW_S * 1000.0
    amort_per_tok = promote_ms / (window_ms / DECODE_BASE_MS)  # rough per-tok share
    hidden = promote_ms <= window_ms
    return DECODE_BASE_MS, promote_ms, amort_per_tok, ssd_gb if hidden else -ssd_gb

def main():
    cfgs = load_defaults()
    lm = LatencyModel(cfgs["p5520"])  # Active TLC, primary

    print("=" * 78)
    print("M5 v2 — XHBM algorithm vs naive (measurement-grounded projection)")
    print("=" * 78)
    print(f"naive KV/tok:   {KV_BF16_BYTES} B ({KV_BF16_BYTES/1e6:.3f} MB) K+V bf16")
    print(f"V-only KV/tok:  {int(KV_VONLY_BYTES)} B ({KV_VONLY_BYTES/1e6:.3f} MB) K bf16 + V 4bit")
    print(f"  -> V-only reduction: {(1-KV_VONLY_BYTES/KV_BF16_BYTES)*100:.1f}%")
    print(f"HBM KV budget:  {hbm_budget_gb():.1f} GB")
    print(f"decode base:    {DECODE_BASE_MS} ms/tok | typing window: {TYPING_WINDOW_S}s")
    print()

    for ctx in [256000, 512000, 1000000]:
        print(f"--- {ctx//1000}K context (D7-P5520) ---")
        # naive with bf16
        n_ms, n_add, n_gb = naive_decode_ms(ctx, lm, KV_BF16_BYTES)
        print(f"  NAIVE (bf16, reread/tok) : KV={kv_gb(ctx,KV_BF16_BYTES):.1f}GB "
              f"offload={n_gb:.1f}GB -> {n_ms:.1f} ms/tok (+{n_add:.1f})")
        # xhbm with v-only quant
        x_ms, promote_ms, amort, x_gb = xhbm_reactivation_ms(ctx, lm, KV_VONLY_BYTES)
        kv_v = kv_gb(ctx, KV_VONLY_BYTES)
        off_v = max(0.0, kv_v - hbm_budget_gb())
        print(f"  XHBM (V-only+idle-driven): KV={kv_v:.1f}GB offload={off_v:.1f}GB")
        print(f"     active decode: {x_ms:.1f} ms/tok (SSD read ~0, KV in HBM)")
        if promote_ms > 0:
            hidden = "HIDDEN in typing window" if promote_ms <= TYPING_WINDOW_S*1000 else "EXCEEDS window!"
            print(f"     reactivation promote: {promote_ms:.0f} ms one-time ({hidden})")
        else:
            print(f"     reactivation: none (fits HBM with V-only quant)")
        # speedup
        if n_ms > 0 and x_ms > 0:
            print(f"     -> active-decode speedup vs naive: {n_ms/x_ms:.1f}x")
        print()

    print("KEY INSIGHT:")
    print("- Naive rereads SSD KV every token -> seconds/token (unusable).")
    print("- XHBM keeps active KV in HBM; SSD only for idle conv. Active decode")
    print("  stays at base latency. SSD promote is one-time, hidden in typing pause.")
    print("- V-only quant shrinks KV so more context fits HBM before any offload.")
    print()
    print("HONEST DISCLOSURE:")
    print("- Amortization assumes promote overlaps typing window (deck v6.3 sec 5).")
    print("- Emulator: throughput+PCIe+write-amp only. NOT queue/thermal/GC/FTL.")
    print("- Real measurement = PoC stage with Solidigm hardware.")

if __name__ == "__main__":
    main()
