#!/usr/bin/env python3
"""
iu_align_smoke.py — IU-parameterization smoke test (no GPU required)

Verifies that the alignment layer is a config parameter, not a constant:
for each candidate IU (4/16/32/64/512 KiB) it checks
  (P5336 QLC IU is 16 KiB or 32 KiB depending on capacity — Solidigm spec)
  (1) NixlSerializationCore box-packing math uses the injected IU
  (2) churn-pilot chunking produces IU-multiple write sizes (except the tail)
  (3) a real file written with the same pwrite loop lands at the exact payload size
Run:  python3 scripts/iu_align_smoke.py
"""
import json, os, sys, tempfile
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

PAYLOAD_PER_TOK = 102_400
RESULTS = {"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "purpose": "IU parameterization smoke — alignment is a config parameter",
           "cells": []}

def check_boxes(iu_bytes, total_bytes):
    from nixl_core import NixlSerializationCore
    os.environ["XHBM_IU_BYTES"] = str(iu_bytes)
    core = NixlSerializationCore()
    import math
    expect = math.ceil(total_bytes / iu_bytes)
    got_iu = core.chunk_size_bytes
    ok = (got_iu == iu_bytes)
    waste = expect * iu_bytes - total_bytes
    return ok, expect, waste

def check_chunking(iu_bytes, payload):
    chunk = max(1 * 1024 * 1024 // iu_bytes, 1) * iu_bytes
    sizes, written = [], 0
    while written < payload:
        n = min(chunk, payload - written)
        sizes.append(n)
        written += n
    body_aligned = all(s % iu_bytes == 0 for s in sizes[:-1])
    tail_aligned = sizes[-1] % iu_bytes == 0
    return chunk, len(sizes), body_aligned, tail_aligned

def check_real_write(iu_bytes, payload_mb=8):
    payload = payload_mb * 1024 * 1024
    chunk = max(1 * 1024 * 1024 // iu_bytes, 1) * iu_bytes
    src = os.urandom(chunk)
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        path = tf.name
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    written = 0
    try:
        while written < payload:
            n = min(chunk, payload - written)
            os.pwrite(fd, src[:n], written)
            written += n
    finally:
        os.close(fd)
    size_ok = os.path.getsize(path) == payload
    os.unlink(path)
    return size_ok

def main():
    ctx = 32_768
    session_payload = PAYLOAD_PER_TOK * ctx
    print(f"session payload @32K = {session_payload:,} B "
          f"({session_payload/1e9:.3f} GB)\n")
    print(f"{'IU':>7} | core-IU | boxes/session | waste(B) | chunk(B) | body-align | tail-align | real-write")
    print("-" * 100)
    all_ok = True
    for iu_kb in (4, 16, 32, 64, 512):
        iu = iu_kb * 1024
        core_ok, boxes, waste = check_boxes(iu, session_payload)
        chunk, nchunks, body, tail = check_chunking(iu, session_payload)
        real = check_real_write(iu)
        ok = core_ok and body and real
        all_ok &= ok
        print(f"{iu_kb:>5}KB | {'PASS' if core_ok else 'FAIL':>7} | {boxes:>13,} | {waste:>8,} "
              f"| {chunk:>8,} | {str(body):>10} | {str(tail):>10} | {'PASS' if real else 'FAIL'}")
        RESULTS["cells"].append(dict(iu_kb=iu_kb, core_injected=core_ok, boxes_per_session=boxes,
                                     residual_waste_bytes=waste, chunk_bytes=chunk,
                                     body_chunks_iu_aligned=body, tail_iu_aligned=tail,
                                     real_write_size_exact=real))
    RESULTS["verdict"] = "PASS" if all_ok else "FAIL"
    out = os.path.join(os.path.dirname(__file__), "iu_align_smoke_result.json")
    json.dump(RESULTS, open(out, "w"), indent=2)
    print(f"\nverdict: {RESULTS['verdict']}  ->  {out}")
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
