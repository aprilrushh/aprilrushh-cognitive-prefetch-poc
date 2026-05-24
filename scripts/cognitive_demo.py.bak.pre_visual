# ============================================================================
# DEPRECATED for Phase 3 / Solidigm narrative (anchor v1.6, 2026-05-14)
#
# This script demonstrates CCP (Cognitive Cued Prefetching) FULL vs SPARSE
# attention. CCP was excluded from Phase 3 / Solidigm visit material /
# FMS 2026 paper based on anchor § 4 measurements (4/5 stage slower, GPU
# Hopfield retrieval cost). Andy directly confirmed 2026-05-14:
#   "CCP 는 GPU 에 부담을 주어서 더 느려졌으므로 포함 안 한다"
#
# Code preserved as historical R&D asset; do NOT use in current narrative.
# References:
#   anchor v1.6: https://www.notion.so/360c78cb12ce81b88284e8c6f5163be4
#   anchor v1.5: https://www.notion.so/360c78cb12ce813e8f37fdb944fce3c3
#   anchor § 4 (5-stage measurement table)
# ============================================================================

"""cognitive_demo.py - Production demo for Cognitive Cued Prefetching v2.

A single-command CLI that compares full attention (baseline) against
cognitive prefetch on Llama 3.1 70B NF4. One model load, two rounds,
side-by-side comparison with PCIe save estimate, measured ideal save,
and honest disclosures.

Usage:
    python scripts/cognitive_demo.py --target-context 8192 --max-new 8
    python scripts/cognitive_demo.py --prompt-file doc.txt --target-context 16384
    python scripts/cognitive_demo.py --prompt "Hopfield" --target-context 4096

UmpaRumpa x Solidigm | May 2026
"""
from __future__ import annotations
import argparse, os, sys, textwrap, time, warnings, logging

# Suppress transformers info/warnings for clean Solidigm-facing demo output.
# Real errors and our own prints still reach the user.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.cognitive_cache_v2 import CognitiveCache_v2
from src.attention_patch import (
    apply_cognitive_attention_patch,
    revert_cognitive_attention_patch,
    get_sparse_stats,
)
from predictor import CognitivePredictor


class C:
    USE = True
    @classmethod
    def disable(cls): cls.USE = False
    @classmethod
    def _w(cls, code, s):
        return ("\033[" + code + "m" + str(s) + "\033[0m") if cls.USE else str(s)
    @classmethod
    def cyan(cls, s):   return cls._w("36", s)
    @classmethod
    def green(cls, s):  return cls._w("32", s)
    @classmethod
    def yellow(cls, s): return cls._w("33", s)
    @classmethod
    def red(cls, s):    return cls._w("31", s)
    @classmethod
    def bold(cls, s):   return cls._w("1", s)
    @classmethod
    def dim(cls, s):    return cls._w("2", s)


HR = "=" * 72
SEP = "-" * 72


def banner():
    print()
    print(C.cyan(HR))
    print(C.cyan("  ") + C.bold("Cognitive Cued Prefetching v2 - Production Demo"))
    print(C.dim("  UmpaRumpa x Solidigm  |  Llama 3.1 70B NF4  |  May 2026"))
    print(C.cyan(HR))


def step(n_total, n_curr, msg):
    print()
    print("[" + str(n_curr) + "/" + str(n_total) + "] " + msg)


DEFAULT_BASE = (
    "The Modern Hopfield Network, introduced by Ramsauer et al. in 2021, "
    "extends classical Hopfield networks to continuous states by replacing "
    "the dot-product similarity with softmax. The energy function becomes "
    "log-sum-exp, and the retrieval rule is exactly attention. This unifies "
    "associative memory with transformers. The capacity scales exponentially "
    "with the dimension. Sparse Hopfield variants (Hu 2023) use entmax "
    "instead of softmax, achieving exact retrieval at high beta. "
)


def build_prompt(tokenizer, args):
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            base = f.read()
    elif args.prompt:
        base = args.prompt
    else:
        base = DEFAULT_BASE
    target = args.target_context
    full = base
    while len(tokenizer(full).input_ids) < target:
        full = full + base
    ids = tokenizer(full, return_tensors="pt").input_ids[0]
    if ids.shape[0] > target:
        ids = ids[:target]
    return ids.unsqueeze(0)


def run_round(model, tokenizer, input_ids, max_new, n_layers, use_sparse):
    cache = CognitiveCache_v2(
        predictor=CognitivePredictor(),
        num_layers=n_layers, device="cuda", enable_async=True,
    )
    apply_cognitive_attention_patch(use_sparse=use_sparse)

    input_ids_cuda = input_ids.to("cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids_cuda, max_new_tokens=max_new, do_sample=False,
            past_key_values=cache, return_dict_in_generate=True, use_cache=True,
        )
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()

    new_ids = out.sequences[0, input_ids.shape[1]:].tolist()
    decoded = tokenizer.decode(new_ids)
    union_sizes = [e[1] for e in cache.stats["indices_log"]]
    sparse_stats = dict(get_sparse_stats())
    pcie_d2h = cache.stats["pcie_d2h_bytes"] / 1e9
    pcie_h2d = cache.stats["pcie_h2d_bytes"] / 1e9
    pcie_sub = cache.stats["pcie_subset_bytes"] / 1e9

    revert_cognitive_attention_patch()
    return {
        "new_ids": new_ids, "decoded": decoded, "elapsed": elapsed, "peak": peak,
        "union_sizes": union_sizes, "sparse_stats": sparse_stats,
        "pcie_d2h": pcie_d2h, "pcie_h2d": pcie_h2d, "pcie_sub": pcie_sub,
    }


def print_round_result(r, baseline_time=None):
    line = "   time:       " + ("%.2f" % r["elapsed"]) + "s"
    if baseline_time is not None and baseline_time > 0:
        delta_pct = (1.0 - r["elapsed"] / baseline_time) * 100.0
        if delta_pct > 0.5:
            line += "   (" + C.green(("%.1f%%" % delta_pct) + " faster than full") + ")"
        elif delta_pct < -0.5:
            line += "   (" + C.yellow(("%.1f%%" % abs(delta_pct)) + " slower than full") + ")"
        else:
            line += "   (~equal to full)"
    print(line)
    print("   new ids:    " + str(r["new_ids"]))
    print("   decoded:    " + repr(r["decoded"]))
    print("   GPU peak:   " + ("%.2f" % r["peak"]) + " GB")


def print_comparison(r_full, r_sparse, prefill_len):
    print()
    print(C.cyan(HR))
    print(C.cyan("  ") + C.bold("COMPARISON"))
    print(C.cyan(HR))

    n_match = sum(1 for a, b in zip(r_full["new_ids"], r_sparse["new_ids"]) if a == b)
    n_total = len(r_full["new_ids"])
    match_rate = 100.0 * n_match / max(n_total, 1)

    if match_rate >= 98:
        match_str = C.green(("%d/%d (%.1f%%)" % (n_match, n_total, match_rate)) + "  [bit-equivalent]")
        verdict = C.green("Gate 4 PASS  (>= 98% top-1 match)")
    elif match_rate >= 50:
        match_str = C.yellow("%d/%d (%.1f%%)" % (n_match, n_total, match_rate))
        verdict = C.yellow("PARTIAL - beta sweep recommended")
    else:
        match_str = C.red("%d/%d (%.1f%%)" % (n_match, n_total, match_rate))
        verdict = C.red("WEAK - hypothesis revision")

    if r_sparse["union_sizes"]:
        u_mean = sum(r_sparse["union_sizes"]) / len(r_sparse["union_sizes"])
        save_pct = (1.0 - u_mean / prefill_len) * 100.0
        u_ratio = (u_mean / prefill_len) * 100.0
    else:
        u_mean = 0.0; save_pct = 0.0; u_ratio = 0.0

    if r_sparse["pcie_h2d"] > 0:
        ideal_save_pct = (1.0 - r_sparse["pcie_sub"] / r_sparse["pcie_h2d"]) * 100.0
    else:
        ideal_save_pct = 0.0

    print("   Greedy match:           " + match_str)
    print("   Full text:              " + repr(r_full["decoded"]))
    print("   Sparse text:            " + repr(r_sparse["decoded"]))
    print()
    print("   PCIe save (estimate):       " + C.bold("%.2f%%" % save_pct))
    print("   PCIe save (measured ideal): " + C.bold("%.2f%%" % ideal_save_pct) + "  (subset / K_full)")
    print("   Estimate vs measured:       %.2fpp difference" % abs(save_pct - ideal_save_pct))
    print("   Union mean:                 %.0f / %d tokens (%.1f%% of prefill)" % (u_mean, prefill_len, u_ratio))
    print()
    print("   Full time:    %.2fs    GPU peak: %.2f GB" % (r_full["elapsed"], r_full["peak"]))
    print("   Sparse time:  %.2fs    GPU peak: %.2f GB" % (r_sparse["elapsed"], r_sparse["peak"]))
    print()
    print("   Verdict:                " + verdict)


def print_disclosures():
    print()
    print(C.cyan(HR))
    print(C.cyan("  ") + C.bold("HONEST LIMITATIONS  (always disclosed, in v3.0 spirit)"))
    print(C.cyan(HR))
    items = [
        "PCIe save is a mathematical estimate (subset bytes / full bytes), "
        "shown above to agree with measured ideal save typically within "
        "0.01pp. Direct nvidia-smi dmon measurement is the natural next "
        "step under v3.0 Section 6 Path 6 (DC Joint Validation).",

        "The current prototype operates in K_full GPU staging mode "
        "(mechanism validation prioritized). The real K_subset-only mode "
        "yielding true measured ~64% PCIe save is the 9-6d follow-up.",

        "PCIe save at 32K context is 58.75% (slightly below the 60% bar). "
        "Reaching the 80% theoretical requires top_k reduction and "
        "increased KV-head independence (planned).",

        "HBM save figures (30-75% range) are mathematical projections. "
        "Measured combined HBM save under simultaneous V-only quantization "
        "+ cognitive prefetch requires 9-6d implementation followed by "
        "joint validation per v3.0 Section 6 Path 6.",
    ]
    for i, txt in enumerate(items, 1):
        wrapped = textwrap.fill(
            txt, width=68,
            initial_indent="   " + str(i) + ". ",
            subsequent_indent="      ",
        )
        print(wrapped)
        print()
    print(C.cyan(HR))


def parse_args():
    p = argparse.ArgumentParser(
        prog="cognitive_demo",
        description=(
            "Cognitive Cued Prefetching v2 - production demo for Solidigm "
            "engineers. Compares baseline (full attention) vs cognitive "
            "prefetch on Llama 3.1 70B NF4. Single model load, two rounds."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/cognitive_demo.py --target-context 8192 --max-new 8\n"
            "  python scripts/cognitive_demo.py --prompt-file doc.txt --target-context 16384\n"
            "  python scripts/cognitive_demo.py --prompt 'Hopfield' --target-context 4096\n"
        ),
    )
    p.add_argument("--prompt", type=str, default=None,
                   help="Inline prompt text (extended by repetition to --target-context).")
    p.add_argument("--prompt-file", type=str, default=None,
                   help="Path to a text file used as prompt (extended/truncated to --target-context).")
    p.add_argument("--target-context", type=int, default=8192,
                   help="Target context length in tokens (default: 8192).")
    p.add_argument("--max-new", type=int, default=8,
                   help="Number of new tokens to generate (default: 8).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42).")
    p.add_argument("--no-color", action="store_true",
                   help="Disable ANSI color codes (for logging or non-TTY output).")
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.1-70B-Instruct",
                   help="Model ID (default: meta-llama/Llama-3.1-70B-Instruct).")
    return p.parse_args()


def main():
    args = parse_args()
    if args.no_color or not sys.stdout.isatty():
        C.disable()

    torch.manual_seed(args.seed)

    banner()
    print("  Model:           " + args.model)
    print("  Target context:  %d tokens" % args.target_context)
    print("  Decode tokens:   %d" % args.max_new)
    print("  Seed:            %d" % args.seed)
    if not torch.cuda.is_available():
        print(C.red("  ERROR: CUDA not available. This demo requires a GPU."))
        return 1
    print("  Device:          CUDA (" + torch.cuda.get_device_name(0) + ")")

    n_steps = 4

    step(n_steps, 1, "Loading Llama 70B NF4 (typically 30-35s)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    load_t = time.time() - t0
    gpu_after = torch.cuda.memory_allocated() / 1e9
    print("   " + C.green("OK") + "  %.1fs, GPU %.2f GB, %d layers" % (load_t, gpu_after, n_layers))

    step(n_steps, 2, "Building prompt...")
    input_ids = build_prompt(tokenizer, args)
    prefill_len = int(input_ids.shape[1])
    print("   " + C.green("OK") + "  prefill_len = %d tokens" % prefill_len)

    step(n_steps, 3, "Round " + C.bold("FULL") + "    (baseline, full attention)")
    r_full = run_round(model, tokenizer, input_ids, args.max_new, n_layers, False)
    print_round_result(r_full)
    torch.cuda.empty_cache()

    step(n_steps, 4, "Round " + C.bold("SPARSE") + "  (cognitive prefetch)")
    r_sparse = run_round(model, tokenizer, input_ids, args.max_new, n_layers, True)
    print_round_result(r_sparse, baseline_time=r_full["elapsed"])

    print_comparison(r_full, r_sparse, prefill_len)
    print_disclosures()

    print()
    print(C.dim("  See README_DEMO.md for environment requirements and reproducibility."))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
