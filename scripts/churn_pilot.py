#!/usr/bin/env python3
"""
churn_pilot.py — Q3 write-volume formula verification pilot (synthetic session payloads)

Scenario-A 1/16-scale pilot: N sessions cycle idle->demote(write)->recall(read)
with Poisson-ish (exponential) dwell times. Verifies:
  (1) bytes written per demote == formula payload (102,400 B/token x context)
  (2) sustained write bandwidth, 64KB-aligned sequential
  (3) realized cadence vs assumption
Labels: I/O characterization with synthetic session payloads (device-independent
write accounting). Not a correctness test; correctness cells run separately.

Usage:
  python3 churn_pilot.py --sessions 16 --context 32768 --duration 3600 \
      --mean-active 120 --mean-idle 120 --data-dir ~/churn_data --demo

Outputs: results JSON + demo log lines (meeting-ready format).
"""
import argparse, json, os, random, threading, time, queue, statistics, sys
from datetime import datetime, timezone

PAYLOAD_PER_TOK = 102_400          # B/token: Cold-40 layers, K bf16 + V 4-bit (transport codec)
CHUNK = 1 * 1024 * 1024            # 1 MiB write chunks (64KB-aligned multiple)
ALIGN = 64 * 1024

def ts():
    return datetime.now().strftime("%H:%M:%S")

def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def read_diskstats(dev="vda"):
    """Return (sectors_read, sectors_written) for device from /proc/diskstats."""
    try:
        with open("/proc/diskstats") as f:
            for line in f:
                p = line.split()
                if len(p) >= 10 and p[2] == dev:
                    return int(p[5]), int(p[9])
    except Exception:
        pass
    return None, None

class Session(threading.Thread):
    def __init__(self, sid, payload_bytes, data_dir, stop_at, mean_active, mean_idle,
                 src_buf, stats, lock, demo, log):
        super().__init__(daemon=True)
        self.sid = sid
        self.payload = payload_bytes
        self.path = os.path.join(data_dir, f"session_{sid:02d}.kv")
        self.stop_at = stop_at
        self.mean_active = mean_active
        self.mean_idle = mean_idle
        self.src = src_buf
        self.stats = stats
        self.lock = lock
        self.demo = demo
        self.log = log

    def emit(self, msg):
        line = f"[{ts()}] {msg}"
        if self.demo:
            print(line, flush=True)
        self.log.append(line)

    def demote(self):
        t0 = time.monotonic()
        written = 0
        # overwrite same file each cycle -> bounded disk usage, cumulative writes still counted
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            src_len = len(self.src)
            off = 0
            while written < self.payload:
                n = min(CHUNK, self.payload - written)
                # 64KB-aligned chunk sizes except possibly the tail
                buf = self.src[off % src_len: off % src_len + n]
                if len(buf) < n:
                    buf = (self.src * 2)[:n]
                os.pwrite(fd, buf, written)
                written += n
                off += n
            # fsync-free by design; durability handled by end-of-run sync for accounting
        finally:
            os.close(fd)
        dt = time.monotonic() - t0
        with self.lock:
            self.stats["app_bytes_written"] += written
            self.stats["demotes"] += 1
            self.stats["demote_secs"].append(dt)
        self.emit(f"session_{self.sid:02d} idle detected \u2192 demoting "
                  f"{written/1e9:.2f} GB in single op (64KB-aligned, fsync-free) "
                  f"\u2192 {written/1e9/dt:.2f} GB/s \u2192 HBM seat freed "
                  f"[cum {self.stats['app_bytes_written']/1e12:.3f} TB]")
        return dt

    def recall(self):
        t0 = time.monotonic()
        got = 0
        try:
            with open(self.path, "rb", buffering=0) as f:
                while True:
                    b = f.read(CHUNK)
                    if not b:
                        break
                    got += len(b)
        except FileNotFoundError:
            return 0.0
        dt = time.monotonic() - t0
        with self.lock:
            self.stats["app_bytes_read"] += got
            self.stats["recalls"] += 1
            self.stats["recall_secs"].append(dt)
        self.emit(f"session_{self.sid:02d} cognitive cue \u2192 recall "
                  f"{got/1e9:.2f} GB in {dt:.2f}s (page-cache-warm; cold-read numbers in recall_storm.py) "
                  f"\u2192 decode resumed")
        return dt

    def run(self):
        # stagger starts a bit
        time.sleep(random.uniform(0, 5))
        cycle_marks = []
        while time.monotonic() < self.stop_at:
            # ACTIVE dwell
            time.sleep(min(random.expovariate(1.0 / self.mean_active),
                           max(0.0, self.stop_at - time.monotonic())))
            if time.monotonic() >= self.stop_at:
                break
            c0 = time.monotonic()
            self.demote()
            # IDLE dwell
            time.sleep(min(random.expovariate(1.0 / self.mean_idle),
                           max(0.0, self.stop_at - time.monotonic())))
            if time.monotonic() >= self.stop_at:
                break
            self.recall()
            cycle_marks.append(time.monotonic() - c0)
        with self.lock:
            self.stats["cycle_secs"].extend(cycle_marks)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=16)
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--duration", type=int, default=3600, help="seconds")
    ap.add_argument("--mean-active", type=float, default=120.0)
    ap.add_argument("--mean-idle", type=float, default=120.0)
    ap.add_argument("--data-dir", default=os.path.expanduser("~/churn_data"))
    ap.add_argument("--device", default="vda", help="block device for /proc/diskstats")
    ap.add_argument("--demo", action="store_true", help="print live demo log lines")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    payload = PAYLOAD_PER_TOK * args.context
    os.makedirs(args.data_dir, exist_ok=True)
    st = os.statvfs(args.data_dir)
    free = st.f_bavail * st.f_frsize
    need = payload * args.sessions
    if free < need * 1.2:
        sys.exit(f"insufficient disk: need ~{need/1e9:.0f} GB resident, free {free/1e9:.0f} GB")

    print(f"=== CHURN PILOT start {utcnow()} ===")
    print(f"sessions={args.sessions} context={args.context} payload/session={payload/1e9:.3f} GB "
          f"(formula: {PAYLOAD_PER_TOK} B/tok x {args.context} tok)")
    print(f"expected bytes per full-fleet cycle = {args.sessions * payload / 1e9:.1f} GB")
    print(f"duration={args.duration}s mean_active={args.mean_active}s mean_idle={args.mean_idle}s")

    src = os.urandom(64 * 1024 * 1024)  # 64 MiB random source, reused (incompressible-ish)
    stats = {"app_bytes_written": 0, "app_bytes_read": 0, "demotes": 0, "recalls": 0,
             "demote_secs": [], "recall_secs": [], "cycle_secs": []}
    lock = threading.Lock()
    log = []

    sr0, sw0 = read_diskstats(args.device)
    t_start = time.monotonic()
    stop_at = t_start + args.duration

    threads = [Session(i, payload, args.data_dir, stop_at, args.mean_active,
                       args.mean_idle, src, stats, lock, args.demo, log)
               for i in range(args.sessions)]
    for t in threads: t.start()
    for t in threads: t.join()

    os.sync()  # flush page cache so device counters reflect all writes
    elapsed = time.monotonic() - t_start
    sr1, sw1 = read_diskstats(args.device)

    dev_written = (sw1 - sw0) * 512 if (sw0 is not None and sw1 is not None) else None
    app_w = stats["app_bytes_written"]

    report = {
        "start_utc": utcnow(), "elapsed_s": round(elapsed, 1),
        "config": vars(args) | {"payload_per_session_bytes": payload},
        "demotes": stats["demotes"], "recalls": stats["recalls"],
        "app_bytes_written": app_w,
        "app_TB_written": round(app_w / 1e12, 4),
        "expected_bytes(demotes x payload)": stats["demotes"] * payload,
        "formula_delta_pct": round(100.0 * (app_w - stats["demotes"] * payload) /
                                   max(1, stats["demotes"] * payload), 3),
        "device_bytes_written(diskstats)": dev_written,
        "device_vs_app_ratio": round(dev_written / app_w, 3) if (dev_written and app_w) else None,
        "sustained_write_GBps(app)": round(app_w / 1e9 / elapsed, 3),
        "demote_secs_p50": round(statistics.median(stats["demote_secs"]), 2) if stats["demote_secs"] else None,
        "recall_secs_p50(warm)": round(statistics.median(stats["recall_secs"]), 2) if stats["recall_secs"] else None,
        "realized_mean_cycle_s": round(statistics.mean(stats["cycle_secs"]), 1) if stats["cycle_secs"] else None,
        "notes": [
            "synthetic session payloads; device-independent write accounting",
            "device counter on cloud virtio-blk; NAND-level WAF requires physical drive (PoC Week 1)",
            "recall here is page-cache-warm; cold-read latency measured by recall_storm.py",
        ],
    }
    out = args.out or f"churn_pilot_{datetime.now().strftime('%m%d_%H%M')}.json"
    with open(out, "w") as f:
        json.dump({"report": report, "demo_log_tail": log[-40:]}, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"=== CHURN PILOT end; report -> {out} ===")

if __name__ == "__main__":
    main()
