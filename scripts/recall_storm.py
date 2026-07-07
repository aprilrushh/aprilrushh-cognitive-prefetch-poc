#!/usr/bin/env python3
"""
recall_storm.py — concurrent-recall serialization measurement

Question a fleet operator asks first: when k users return simultaneously,
how does recall latency serialize? Two phases:

  Phase A (disk -> RAM): cold reads of per-session payload files, k in {1,2,4,8}.
           Uses sudo drop_caches between trials for true cold reads.
  Phase B (RAM -> HBM, optional, needs torch+CUDA): pinned-host to GPU copies on
           separate CUDA streams, k in {1,2,4,8} -> PCIe serialization factor.

The serialization FACTOR (p95_k / p95_1) transfers across hardware even where
absolute latency is cloud-storage-specific.

Usage:
  python3 recall_storm.py --context 32768 --trials 3 --data-dir ~/churn_data
  (reuses churn_pilot session files if present; creates them otherwise)
"""
import argparse, json, os, statistics, subprocess, threading, time, sys
from datetime import datetime

PAYLOAD_PER_TOK = 102_400
CHUNK = 4 * 1024 * 1024

def make_file(path, size):
    if os.path.exists(path) and os.path.getsize(path) >= size:
        return
    src = os.urandom(64 * 1024 * 1024)
    with open(path, "wb") as f:
        w = 0
        while w < size:
            n = min(len(src), size - w)
            f.write(src[:n]); w += n

def drop_caches():
    try:
        subprocess.run(["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
                       check=True, capture_output=True)
        return True
    except Exception as e:
        print(f"  (warn) drop_caches failed: {e} -> reads may be warm", flush=True)
        return False

def timed_read(path, out):
    t0 = time.monotonic()
    n = 0
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(CHUNK)
            if not b: break
            n += len(b)
    out.append((time.monotonic() - t0, n))

def phase_a(files, ks, trials):
    res = {}
    for k in ks:
        per_thread = []
        for t in range(trials):
            drop_caches()
            outs = [[] for _ in range(k)]
            ths = [threading.Thread(target=timed_read, args=(files[i], outs[i])) for i in range(k)]
            t0 = time.monotonic()
            for th in ths: th.start()
            for th in ths: th.join()
            wall = time.monotonic() - t0
            lat = [o[0][0] for o in outs]
            per_thread.extend(lat)
            gb = sum(o[0][1] for o in outs) / 1e9
            print(f"  A k={k} trial{t+1}: wall={wall:.2f}s per-recall p95={pct(lat,95):.2f}s "
                  f"agg_read={gb/wall:.2f} GB/s", flush=True)
        res[k] = {"p50_s": round(pct(per_thread,50),3), "p95_s": round(pct(per_thread,95),3),
                  "max_s": round(max(per_thread),3), "n": len(per_thread)}
    base = res[ks[0]]["p95_s"]
    for k in ks:
        res[k]["serialization_factor_p95"] = round(res[k]["p95_s"]/base, 2)
    return res

def pct(v, p):
    v = sorted(v); i = max(0, min(len(v)-1, int(round(p/100*(len(v)-1)))))
    return v[i]

def phase_b(payload, ks, trials):
    try:
        import torch
        assert torch.cuda.is_available()
    except Exception as e:
        print(f"Phase B skipped (torch/CUDA unavailable: {e})")
        return None
    import torch
    dev = torch.device("cuda:0")
    res = {}
    n_el = payload  # uint8
    max_k = max(ks)
    print(f"  allocating {max_k} pinned host buffers x {payload/1e9:.2f} GB ...", flush=True)
    host = [torch.empty(n_el, dtype=torch.uint8, pin_memory=True) for _ in range(max_k)]
    gpu  = [torch.empty(n_el, dtype=torch.uint8, device=dev) for _ in range(max_k)]
    streams = [torch.cuda.Stream() for _ in range(max_k)]
    for k in ks:
        lats = []
        for t in range(trials):
            evs = []
            torch.cuda.synchronize()
            t0 = time.monotonic()
            for i in range(k):
                with torch.cuda.stream(streams[i]):
                    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
                    e0.record(streams[i])
                    gpu[i].copy_(host[i], non_blocking=True)
                    e1.record(streams[i])
                    evs.append((e0, e1))
            torch.cuda.synchronize()
            wall = time.monotonic() - t0
            per = [e0.elapsed_time(e1)/1000.0 for (e0, e1) in evs]
            lats.extend(per)
            print(f"  B k={k} trial{t+1}: wall={wall:.2f}s per-copy p95={pct(per,95):.2f}s "
                  f"agg_h2d={k*payload/1e9/wall:.1f} GB/s", flush=True)
        res[k] = {"p50_s": round(pct(lats,50),3), "p95_s": round(pct(lats,95),3), "n": len(lats)}
    base = res[ks[0]]["p95_s"]
    for k in ks:
        res[k]["serialization_factor_p95"] = round(res[k]["p95_s"]/base, 2)
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--data-dir", default=os.path.expanduser("~/churn_data"))
    ap.add_argument("--ks", default="1,2,4,8")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--skip-b", action="store_true")
    args = ap.parse_args()

    payload = PAYLOAD_PER_TOK * args.context
    ks = [int(x) for x in args.ks.split(",")]
    os.makedirs(args.data_dir, exist_ok=True)
    files = [os.path.join(args.data_dir, f"session_{i:02d}.kv") for i in range(max(ks))]
    print(f"=== RECALL STORM: payload/session {payload/1e9:.3f} GB (context {args.context}) ===")
    for f in files: make_file(f, payload)

    print("--- Phase A: disk -> RAM (cold) ---")
    a = phase_a(files, ks, args.trials)
    b = None
    if not args.skip_b:
        print("--- Phase B: RAM -> HBM (pinned, per-stream) ---")
        b = phase_b(payload, ks, args.trials)

    report = {"context": args.context, "payload_GB": round(payload/1e9,3),
              "phase_a_disk_to_ram": a, "phase_b_ram_to_hbm": b,
              "notes": [
                "Phase A absolute values are cloud virtio-blk specific; factors transfer",
                "true end-to-end recall ~ max(disk, h2d) when pipelined; sum when naive",
                "P5336 direct-attached numbers measured in PoC Week 1",
              ]}
    out = f"recall_storm_{datetime.now().strftime('%m%d_%H%M')}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"=== done; report -> {out} ===")

if __name__ == "__main__":
    main()
