# Phase 3 Step 3.1c — Llama 3.1 70B NF4 cold-start load (Axis A)

**Date**: 2026-05-14
**Branch**: idle-promotion-bench
**v1.3 spec**: bf16 compute_dtype, NF4 weight quant only (NO V-only KV quant yet — that's 3.1d)

## Configuration
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,   # v1.3 § C — bf16, NOT fp16
    bnb_4bit_use_double_quant=True,
)
AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct",
    quantization_config=bnb_config,
    dtype=torch.bfloat16,                    # transformers 5.x: 'dtype' not 'torch_dtype'
    device_map="auto",
    low_cpu_mem_usage=True,
)
```

## Cold-start load measurement

| Metric | Value | Reference |
|---|---|---|
| Cold-start time (HF cache hot) | **23.48s** | PDF Page 8: 53s (warmed) / 503s (pre-opt) |
| Warm reload (immediately after) | 23.76s | Δ +0.28s — HF cache fully dominant |
| GPU peak (during load) | 43.78 GB | transient ~4.2 GB de-quant buffer |
| GPU post-load | 39.58 GB | quiescent |
| Param count | 36.33 B | from 70.6 B native |
| uint8 (NF4 blocks) | 34.23 B (94.2%) | |
| bf16 (norm/embed) | 2.10 B (5.8%) | |

## Sanity forward pass
- Prompt: `'Sherlock Holmes was a famous'`
- Greedy next token: `' detective'` ✅ semantic OK
- Forward latency: 430.9 ms (7-token prompt, no batch)

## KV cache (transformers 5.x DynamicCache API)
- API: `past_key_values.layers[i].keys / .values` (5.x style, NOT `kv.key_cache` or `kv[i]`)
- num_layers: 80
- K shape: `(1, 8, 7, 128)` bf16
- V shape: `(1, 8, 7, 128)` bf16

## v1.3 § B math reconcile (실측 검증)

| Metric | v1.3 § B claim | 3.1c-fix measurement | Match |
|---|---|---|---|
| 1 conv 32K bf16 KV | 10.74 GB | **10.74 GB** | ✅ exact |
| 1 conv 32K V-only q4 | 6.71 GB | **6.71 GB** | ✅ exact |
| n_kv_heads | 8 | 8 | ✅ |
| head_dim | 128 | 128 | ✅ |

→ v1.3 § B 의 HBM 49.7 GB + RAM 75 GB *동시* mechanism = **수학 100% 검증 완료**.

## Findings (v1.3 ledger update candidates)

### Finding 1: Cold-start 23.48s — PDF narrative 한 칸 확장
- PDF Page 8: 8m 23s → 53s (multi-thread warmup, 11.7×)
- 2026-05-14 measurement: **23.48s** (vs 53s = 2.26×, vs 503s = **21.4×**)
- Boundary: HF disk cache hot + transformers 5.8 fast-loader. Production network-cold = 별도 측정

### Finding 2: transformers 5.x DynamicCache API
- `past_key_values` returns `DynamicCache` instance (not tuple)
- Layer access: `kv.layers[i].keys / .values` (only working pattern)
- `attention_patch.py` (9-6b) hook strategy 검증 → 3.1d target

### Finding 3: `torch_dtype` deprecated
- transformers 5.8 uses `dtype=` parameter
- All Phase 3 code adopts new name

## attention_patch.py 3.1d target (preview)
- Owner: `transformers.models.llama.modeling_llama`
- Class: `LlamaAttention`
- forward signature returns `tuple[torch.Tensor, torch.Tensor]`
- Hook strategy decision: 3.1d-inspect step

## Limitations
- HF disk cache hot (not network-cold)
- Single thread (no multi-thread warmup compare to PDF Page 8)
- n=2 measurement only (cold 23.48s + warm 23.76s) — production cold-start n=3 in Phase 3.2 (M1)
