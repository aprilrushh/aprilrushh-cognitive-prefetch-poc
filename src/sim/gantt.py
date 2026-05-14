"""Phase 2.3a — Gantt-style trace output from SimResult.

Converts SimulationLoop result into:
- "intervals": each block's (tier, t_start, t_end) sequence
  (e.g. conv0 was in HBM from 0..30s, then RAM from 30..900s, ...)
- "events": each transition as a point-in-time event
- Aggregate stats: per-tier dwell time, per-block tier-residence breakdown

Limitation notes propagated to output JSON 'limitations' field so any
consumer of the data sees what's modeled / not modeled.

Library analogy: Gantt = 매일 아침 사서가 작성한 *책 위치 기록표*
(어느 책이 몇 시부터 몇 시까지 어디 있었는지)
"""
from dataclasses import dataclass, asdict
from typing import Dict, List
from pathlib import Path
import json
from src.xhbm_emulator import Tier
from src.sim.loop import SimResult


@dataclass
class TierInterval:
    block_id: str
    tier: str           # Tier.name
    t_start: float
    t_end: float

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


@dataclass
class TransitionEvent:
    t: float
    block_id: str
    from_tier: str
    to_tier: str
    move_latency_us: float
    is_prefetch: bool
    n_hops: int


def build_intervals(result: SimResult) -> List[TierInterval]:
    """Convert per-tick snapshots into (block, tier, t_start, t_end) intervals."""
    intervals: List[TierInterval] = []
    if not result.snapshots:
        return intervals
    # Track current tier per block, opened at start_t
    current: Dict[str, tuple] = {}  # block_id -> (tier_name, start_t)
    for bid, tier in result.snapshots[0].locations.items():
        current[bid] = (tier.name, result.snapshots[0].tick_t)
    for snap in result.snapshots[1:]:
        for bid, tier in snap.locations.items():
            prev_tier, prev_start = current.get(bid, (tier.name, snap.tick_t))
            if tier.name != prev_tier:
                intervals.append(TierInterval(bid, prev_tier, prev_start, snap.tick_t))
                current[bid] = (tier.name, snap.tick_t)
    # Close final intervals at last snapshot time
    end_t = result.snapshots[-1].tick_t
    for bid, (tier_name, start_t) in current.items():
        if start_t < end_t:
            intervals.append(TierInterval(bid, tier_name, start_t, end_t))
    return intervals


def build_events(result: SimResult) -> List[TransitionEvent]:
    return [TransitionEvent(
        t=tr.tick_t, block_id=tr.block_id,
        from_tier=tr.from_tier.name, to_tier=tr.to_tier.name,
        move_latency_us=tr.move_latency_us, is_prefetch=tr.is_prefetch,
        n_hops=tr.n_hops,
    ) for tr in result.transitions]


def per_tier_dwell(intervals: List[TierInterval]) -> Dict[str, float]:
    """Total time blocks spent in each tier (sum across blocks)."""
    out: Dict[str, float] = {t.name: 0.0 for t in Tier}
    for iv in intervals:
        out[iv.tier] += iv.duration
    return out


def per_block_summary(intervals: List[TierInterval]) -> Dict[str, Dict[str, float]]:
    """For each block: time spent in each tier."""
    out: Dict[str, Dict[str, float]] = {}
    for iv in intervals:
        out.setdefault(iv.block_id, {t.name: 0.0 for t in Tier})
        out[iv.block_id][iv.tier] += iv.duration
    return out


def to_dict(result: SimResult, intervals: List[TierInterval],
            events: List[TransitionEvent]) -> dict:
    """Serializable bundle ready for JSON dump."""
    return {
        "scenario_meta": result.scenario_meta,
        "tick_seconds": result.tick_seconds,
        "duration_seconds": result.duration_seconds,
        "n_blocks": len(result.snapshots[-1].locations) if result.snapshots else 0,
        "n_transitions": len(result.transitions),
        "n_prefetch": sum(1 for tr in result.transitions if tr.is_prefetch),
        "tier_visit_counts": {t.name: c for t, c in result.tier_visit_counts().items()},
        "per_tier_dwell_seconds": per_tier_dwell(intervals),
        "per_block_summary_seconds": per_block_summary(intervals),
        "intervals": [asdict(iv) for iv in intervals],
        "events": [asdict(ev) for ev in events],
        "limitations": [
            "Phase 2.2 Loop1: move_latency_us recorded but does NOT advance "
            "simulation clock - PCIe/SSD queueing NOT modeled",
            "Each transition treated as instantaneous at tick boundary "
            "(max <30s jitter from tick=30s)",
            "Block size = 6.4 GB (1 conv at 32K tokens, V-only NF4). "
            "PDF Page 7 / 49 GB peak match",
            "Synthetic uniform pattern (Phase 2a) - WildChat-sampled "
            "distribution = Phase 2b upgrade",
        ],
        "data_provenance": {
            "ssd_active_profile": "Solidigm D7-P5520 (verified 2026-05-13)",
            "ssd_archive_profile": "Solidigm D5-P5316 (verified 2026-05-13)",
            "threshold_evidence_base": "WildChat-1M 200K conversations "
                                        "(measured 2026-05-13, P50 inter-conv "
                                        "idle = 15min 37s)",
        },
    }


def save_json(result: SimResult, out_path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    intervals = build_intervals(result)
    events = build_events(result)
    payload = to_dict(result, intervals, events)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return out_path


if __name__ == "__main__":
    from src.sim.event_gen import generate_for_scenario
    from src.sim.loop import SimulationLoop
    from src.xhbm_emulator import LatencyModel, TierManager, load_defaults
    from src.idle_tier_scheduler import IdleTierScheduler

    timelines, meta = generate_for_scenario("default_2_5h")
    cfgs = load_defaults()
    mgr = TierManager(LatencyModel(cfgs["p5520"]), LatencyModel(cfgs["p5316"]))
    sch = IdleTierScheduler(cfgs["thresholds"], mgr)
    loop = SimulationLoop(timelines, sch, tick_seconds=30)
    result = loop.run(meta["duration_seconds"], scenario_meta=meta)

    intervals = build_intervals(result)
    events = build_events(result)
    print(f"Intervals: {len(intervals)}")
    print(f"Events:    {len(events)}")
    print(f"\n--- Sample intervals (first 10) ---")
    for iv in intervals[:10]:
        print(f"  {iv.block_id} in {iv.tier:12s} from t={iv.t_start:.0f} to t={iv.t_end:.0f} ({iv.duration:.0f}s)")

    dwell = per_tier_dwell(intervals)
    print(f"\n--- Total tier-dwell (sum across 8 blocks) ---")
    total = sum(dwell.values())
    for tier_name, secs in dwell.items():
        pct = 100 * secs / total if total else 0
        print(f"  {tier_name:12s}: {secs:7.0f}s ({pct:5.1f}%)")

    per_block = per_block_summary(intervals)
    print(f"\n--- Per-block tier residence (seconds) ---")
    print(f"  {'block':<8} {'HBM':>6} {'RAM':>6} {'ACTIVE':>7} {'ARCHIVE':>8}")
    for bid in sorted(per_block.keys()):
        b = per_block[bid]
        print(f"  {bid:<8} {b.get('HBM',0):6.0f} {b.get('RAM',0):6.0f} "
              f"{b.get('SSD_ACTIVE',0):7.0f} {b.get('SSD_ARCHIVE',0):8.0f}")
