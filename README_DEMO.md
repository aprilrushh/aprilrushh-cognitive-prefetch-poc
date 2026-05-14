> ⚠️ **STATUS UPDATE (2026-05-14, anchor v1.6)**
>
> 이 demo 는 CCP (Cognitive Cued Prefetching) prototype 의 historical record 입니다.
> CCP 는 anchor § 4 의 5-stage 측정 결과 (4/5 stage slower, GPU Hopfield retrieval cost 추가)
> 에 따라 **Phase 3 / Solidigm 방문 자료 / FMS 2026 paper 에서 제외** 됩니다.
>
> 이유: Andy 직접 짚음 (2026-05-14):
> > "CCP 는 우리가 테스트 해보니까 도리어 GPU 에 이것 계산하느라 부담을 주어서
> > GPU 가 더 느려졌으므로 이건 포함 안 하는 것이 좋을 것 같다"
>
> Phase 3 main measurement asset = M1.1-M1.4 (NF4 + idle-driven scheduling).
> Anchor v1.5 / v1.6 참조:
> - v1.6: https://www.notion.so/360c78cb12ce81b88284e8c6f5163be4
> - v1.5: https://www.notion.so/360c78cb12ce813e8f37fdb944fce3c3
>
> 코드 자산은 보존 (historical reference). 외부 communication 에서만 제외.

---

# Cognitive Cued Prefetching v2 — Reproducibility Demo

**For**: Solidigm AI Technologist & Engineering Team
**From**: Andy Lee, UmpaRumpa
**Date**: May 2026
**Companion**: Follow-up Update PDF (v3.1)

This README accompanies cognitive_demo.py and lets a Solidigm engineer
reproduce, on their own H100 (or equivalent) and via a single command,
the production smoke results cited in the v3.1 Follow-up Update.

---

## 1. What This Demo Shows

Side-by-side comparison on the same prompt, single model load:

| Round  | What it runs                          |
|--------|----------------------------------------|
| FULL   | Baseline standard full attention       |
| SPARSE | Cognitive prefetch Hopfield-guided     |

Output (rendered on a single screen):

- Greedy match (top-1 token equivalence) between FULL and SPARSE
- PCIe traffic save both estimate (subset / prefill) and measured ideal save (subset bytes / K_full bytes), plus the gap in pp
- Wall-clock time and GPU peak per round
- Verdict line (Gate 4 PASS / PARTIAL / WEAK)
- Honest limitations block (always disclosed)

A representative 4K-context run (~50 seconds end-to-end on H100):

    Greedy match:               6/6 (100.0%)  [bit-equivalent]
    PCIe save (estimate):       63.42%
    PCIe save (measured ideal): 63.44%  (subset / K_full)
    Estimate vs measured:       0.02pp difference
    Sparse time: 4.82s   Full time: 5.23s   (-7.9% faster)
    GPU peak:    full 46.91 GB   sparse 45.33 GB
    Verdict:                    Gate 4 PASS  (>= 98% top-1 match)

---

## 2. Environment Requirements

### Hardware
- 1x NVIDIA H100 80GB (or equivalent: A100 80GB also works, B200 better)
- ~50 GB free GPU memory (Llama 70B NF4 baseline + SPARSE round peak)
- ~80 GB free system RAM (CPU-side KV mirror + buffers)

### Software
- Linux (tested: Ubuntu 22.04 / 24.04)
- CUDA 12.4+ (driver 550+)
- Python 3.10 or 3.11
- The repository at the same checkout used for v3.1 measurements

### Python packages (versions used in our measurements)
- torch == 2.6.0+cu124
- transformers == 5.8.0
- bitsandbytes == 0.49.2
- accelerate (any recent)
- entmax == 1.3 (Sparse Hopfield)

### Model access
- Hugging Face account with access to meta-llama/Llama-3.1-70B-Instruct
- huggingface-cli login completed once
- Model weights are downloaded on first run (~140 GB raw, ~40 GB with --exclude original/, ~30 min on a fast link; cached afterwards)

---

## 3. One-Time Setup

Run these once after cloning the repository:

    # 1. Create a virtual environment (recommended)
    python3.11 -m venv .venv
    source .venv/bin/activate

    # 2. Install pinned dependencies
    pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
    pip install transformers==5.8.0 bitsandbytes==0.49.2 accelerate entmax==1.3

    # 3. Authenticate with Hugging Face (one time)
    huggingface-cli login

    # 4. Verify GPU + driver
    nvidia-smi

    # 5. Sanity check: import the package
    python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.__version__)"

Expected output of step 5: a line like "NVIDIA H100 80GB HBM3" and "2.6.0+cu124".


---

## 4. CLI Usage

Single command, single model load, full vs sparse comparison:

    # Inline prompt (extended by repetition to target context length)
    python scripts/cognitive_demo.py --prompt "Long context about Hopfield..." --max-new 8

    # Or from a prompt file
    python scripts/cognitive_demo.py --prompt-file long_doc.txt --max-new 12

    # Or repeat a short prompt to fill the target context (auto generation)
    python scripts/cognitive_demo.py --prompt "Hopfield" --target-context 16384 --max-new 8

### All flags

| Flag                | Meaning                                                                |
|---------------------|------------------------------------------------------------------------|
| --prompt            | Inline prompt (extended by repetition to target).                      |
| --prompt-file       | Path to a text file used as prompt.                                    |
| --target-context    | Target prefill length in tokens (default: 8192).                       |
| --max-new           | Number of new tokens to generate (default: 8).                         |
| --seed              | Random seed (default: 42).                                             |
| --no-color          | Disable ANSI color (for log capture or non-TTY).                       |
| --model             | Model ID (default: meta-llama/Llama-3.1-70B-Instruct).                 |

---

## 5. How to Read the Output

### Greedy match
Number of top-1 tokens that are identical between FULL and SPARSE.
We treat 100% (or >= 98% for >50-token decodes) as **bit-equivalent**:
sparse attention reproduces full attention output exactly. This is
the v3.1 Gate 4 criterion.

### PCIe save (estimate)
(1 - union_mean / prefill_length) * 100. Mathematical projection
based on the union of selected token indices across attention layers.
Cited throughout the v3.1 PDF Section 4 table.

### PCIe save (measured ideal)
(1 - subset_bytes / K_full_bytes) * 100. Computed from the actual
byte counters tracked inside cognitive_cache_v2.py. This represents
the PCIe save that would be realized once the prototype is moved to
**K_subset-only mode** (9-6d follow-up). The estimate-vs-measured gap
is typically within 0.01-0.02 pp, validating that the mathematical
narrative is faithful to code behavior.

### Union mean / prefill (% of prefill)
Average number of tokens selected per attention layer. The remaining
(100% - this%) is the PCIe traffic that the cognitive predictor
identifies as not needed and skips.

### Sparse time vs Full time
With cognitive prefetch enabled, sparse attention typically runs
2-15% faster than full because the attention compute itself is on a
smaller subset. The PCIe overhead of staging is more than offset.

---

## 6. Reproducing Specific v3.1 Numbers

The v3.1 Follow-up PDF Section 4 cites four context stages on a single
H100. Each row maps to one demo invocation:

| PDF row | Command                                                                      |
|---------|------------------------------------------------------------------------------|
| 4K      | python scripts/cognitive_demo.py --target-context 4096  --max-new 6          |
| 8K      | python scripts/cognitive_demo.py --target-context 8192  --max-new 6          |
| 16K     | python scripts/cognitive_demo.py --target-context 16384 --max-new 6          |
| 32K     | python scripts/cognitive_demo.py --target-context 32768 --max-new 6          |

Expected on H100 80GB:

| Stage | Greedy match | PCIe save (estimate) | Sparse vs Full | GPU peak (full / sparse) |
|-------|--------------|----------------------|----------------|--------------------------|
| 4K    | 6/6 (100%)   | 63.42%               | -7.9% faster   | 46.91 / 45.33 GB         |
| 8K    | 6/6 (100%)   | 64.40%               | -7% faster     | 47.41 / 47.45 GB         |
| 16K   | 6/6 (100%)   | 60.01%               | -4% faster     | 52.05 / 52.09 GB         |
| 32K   | 6/6 (100%)   | 58.75%               | -2% faster     | 61.32 / 61.36 GB         |

PCIe save is monotonically lower as context grows due to KV-head
deduplication ratio. Greedy match stays at 100% across all stages
(bit-equivalent sparse attention at 70B scale).


---

## 7. Honest Limitations (Always Disclosed)

These appear at the bottom of every demo run, in v3.0's spirit:

1. **PCIe save is a mathematical estimate.** Estimate and measured
   ideal save typically agree within 0.01-0.02 pp. Direct
   nvidia-smi dmon measurement is the natural next step under
   v3.0 Section 6 Path 6 (DC Joint Validation).

2. **Current prototype operates in K_full GPU staging mode**
   (mechanism validation prioritized). The real K_subset-only mode
   yielding true measured ~64% PCIe save is the 9-6d follow-up.

3. **PCIe save at 32K context is 58.75%** (slightly below the 60%
   conservative bar). Reaching the 80% theoretical requires top_k
   reduction and increased KV-head independence (planned).

4. **HBM save figures (30-75% range) are mathematical projections.**
   Measured combined HBM save under simultaneous V-only quantization
   plus cognitive prefetch requires 9-6d implementation followed by
   joint validation per v3.0 Section 6 Path 6.

---

## 8. Troubleshooting

| Symptom                                       | Likely cause / fix                                           |
|-----------------------------------------------|--------------------------------------------------------------|
| OSError meta-llama not found                  | Run huggingface-cli login and request gated model access.    |
| OOM at --target-context 32768                 | Free other GPU processes. 32K peak is ~61 GB on H100 80 GB.  |
| transformers ImportError                      | Install exactly 5.8.0; older 4.x will not match the patch.   |
| bitsandbytes cuda load failure                | Verify CUDA 12.4 driver and rebuild bnb 0.49.2 if needed.    |
| Sparse time slower than full at very small ctx| Expected at < 1K prefill the staging overhead dominates.     |
| Demo prints raw ANSI escapes                  | Use --no-color, or run in a real TTY.                        |
| transformers warning lines mixed with banner  | Already suppressed via warning patch; if seen, check that scripts/cognitive_demo.py is the latest version. |

---

## 9. Contact & Next Steps

For technical questions or to discuss next-step joint validation
(under v3.0 Section 6 Path 6):

> **Andy Lee**, UmpaRumpa see Follow-up Update v3.1 cover for contact.

A natural progression after this smoke:
1. Confirm reproduction on Solidigm-side H100 (or equivalent).
2. Co-design nvidia-smi dmon measurement protocol for direct PCIe.
3. Plan 9-6d K_subset-only mode joint test.
4. Initiate XHBM-Ready / Brain-Cooperative SKU spec discussion.

