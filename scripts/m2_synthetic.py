"""M2 Gantt asset — synthetic scenario (default_2_5h).

Phase 2.3a entry point. Produces:
- results/m2/m2_synthetic_default_2_5h.json (full Gantt data)
- stdout: summary + per-block transitions for human review

Phase 2.3b (TBD) will add Loop2 latency contention + companion JSON.
Phase 2.4-2.5 will add WildChat-sampled scenario for comparison.
"""
from pathlib import Path
from src.sim.event_gen import generate_for_scenario
from src.sim.loop import SimulationLoop
from src.sim.gantt import save_json, build_intervals
from src.xhbm_emulator import Tier, LatencyModel, TierManager, load_defaults
from src.idle_tier_scheduler import IdleTierScheduler


def main():
    scenario = "default_2_5h"
    timelines, meta = generate_for_scenario(scenario)
    cfgs = load_defaults()
    mgr = TierManager(LatencyModel(cfgs["p5520"]), LatencyModel(cfgs["p5316"]))
    sch = IdleTierScheduler(cfgs["thresholds"], mgr)
    loop = SimulationLoop(timelines, sch, tick_seconds=30)

    print(f"=== M2 synthetic Gantt — scenario {scenario} ===")
    print(f"  n_conversations: {meta['n_conversations']}")
    print(f"  duration:        {meta['duration_seconds']}s ({meta['duration_seconds']/3600:.1f}h)")
    print(f"  thresholds:      {cfgs['thresholds']}")

    result = loop.run(meta["duration_seconds"], scenario_meta=meta)

    out_path = Path("results/m2") / f"m2_synthetic_{scenario}.json"
    saved = save_json(result, out_path)
    print(f"\nGantt JSON: {saved}")
    print(f"  size: {saved.stat().st_size} bytes")
    print(f"  transitions: {len(result.transitions)}")
    print(f"  prefetches:  {sum(1 for tr in result.transitions if tr.is_prefetch)}")

    intervals = build_intervals(result)
    print(f"  intervals:   {len(intervals)}")
    print(f"  tier visit:  {dict((t.name, c) for t,c in result.tier_visit_counts().items())}")

    print(f"\nFor visualization, the JSON includes:")
    print(f"  - intervals: [(block_id, tier, t_start, t_end), ...]")
    print(f"  - events: [(t, block_id, from->to, latency_us, is_prefetch), ...]")
    print(f"  - per_block_summary_seconds: time each block spent in each tier")
    print(f"  - limitations: explicit list of what's NOT modeled in Phase 2.2")


if __name__ == "__main__":
    main()
