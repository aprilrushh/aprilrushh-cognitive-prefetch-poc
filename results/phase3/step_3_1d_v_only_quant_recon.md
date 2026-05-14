# Phase 3 Step 3.1d — V-only Quant Origin Reconcile + Plan Redefinition

**Date**: 2026-05-14
**Branch**: idle-promotion-bench
**Notion ledger**: https://www.notion.so/360c78cb12ce81ef8461cc0481270d83 (anchor v1.4)

## What we discovered (2 critical findings)

### Finding 1: `attention_patch.py` is NOT V-only quant hook
9-6b clean state of `attention_patch.py` (130 LOC) + `cognitive_cache_v2.py` (217 LOC)
are the **CCP (Cognitive Cued Prefetching) Hopfield-guided sparse attention**
assets, NOT the V-only asymmetric quantization assets.

Anchor v1.0 § 11 description "(V-only quant hook)" was an echo chamber.

### Finding 2: V-only quant code is in NO repo
Verified across 5 repos under aprilrushh GitHub account:
- cognitive-prefetch-poc: 0 V-only quant files
- aprilrushh-cognitive-prefetch-poc: 0
- xhbm-bench (SSD emulator): 0
- xhbm (empty placeholder repo): 0
- brain-poc (Wikipedia external memory): 0

PDF Page 4 "V-only quant + 70B + cos_sim 1.000000" origin =
**Lambda Labs GH200 + llama.cpp ad-hoc measurement (2026-04-15)**.
Cost: ~$3.50, 1.5h, instance terminated. Code not asset-ized.

## Plan redefinition — Phase 3 = dual evidence (NOT Lambda reproduce)

Original Phase 3 plan (anchor v1.0 § 15):
> M1-M4 real measurement on V-only quant + NF4 + Llama 70B + Sherlock

V1.4 reality-checked plan:
> Phase 3 = idle-driven mechanism + NF4 baseline on H100 + transformers 5.8.
> Lambda V-only quant measurement stays as historical Blue Intelligence asset.
> Two together = dual evidence (theory + code, past + present).

### M1-M4 new scope

| M | Scope | V-only needed? | Status |
|---|---|---|---|
| **M1** | Decode SSD I/O byte trace (NF4 + idle scheduler + Sherlock) | No | essential |
| **M2** | Idle Gantt 4 conv x 32K (NF4 baseline, V-only not needed for scope-reduced) | Partial | partial (4 conv) |
| **M3** | Typing prefetch latency (1 conv NF4 + UI signal) | No | essential |
| **M4** | Capacity 49 GB peak reproduce | Yes | DEFERRED (3 sub-paths, see v1.4 §E) |

## v1.4 key facts captured

1. PDF Page 4 = Lambda GH200 + llama.cpp Q4_K_M, 2026-04-15
2. Company brand = UmpaRumpa (Blue Intelligence as historical origin)
3. attention_patch.py 9-6b = CCP hook, NOT V-only quant hook
4. transformers 5.x DynamicCache API: `kv.layers[i].keys/.values`
5. `torch_dtype` deprecated, use `dtype=` in transformers 5.8
6. Security incident: GitHub PAT exposure + revoke + replace + 5 repos cleaned (resolved)

## Echo chamber ledger update

v1.2 (4) + v1.3 (5) + v1.4 (5) = total **14 fact-based corrections** before Phase 3.2.
Plus 1 security incident (PAT) fully resolved.

## Next: Step 3.2 (M1 - Decode SSD I/O byte trace)

- Workload: NF4 Llama 70B + idle scheduler + 1 conv x 32K Sherlock chunk
- Instrumentation: blktrace or iostat or psutil io_counters during decode
- Expected: 0 bytes during decode phase (SSD inactive on critical path)
- This is the "SSD does NOT block" proof — anchor § 13 anchor 1 evidence
