"""Phase 2.2 — Simulation loop (Loop1: clock + scheduler + transition log).

Tick granularity: 30s (Andy 2026-05-13 decision).
Aligns with smallest threshold (30s HBM->RAM). Max jitter <30s.

LIMITATION (intentional Phase 2.2 scope):
- Each transition records its latency_us in TransitionLog, but the latency
  does NOT advance the simulation clock. Multiple transitions can fire at
  the same tick with no PCIe/SSD queueing modeled.
- Step 2.3 will add Loop2: PCIe busy tracking + queue + contention.

Library analogy:
- Loop1 = 사서가 매 30초 시계 보며 책 옮길 사람 결정 + 옮긴 시간 기록
- Loop2 (next) = 복도/창고가 동시에 하나씩만 다룰 수 있는 제약 반영
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from src.xhbm_emulator import (Tier, KVBlock, MoveResult, TierManager,
                                 LatencyModel, load_defaults)
from src.idle_tier_scheduler import IdleTierScheduler
from src.sim.event_gen import ConvTimeline


@dataclass
class TransitionRecord:
    tick_t: float            # simulation second of decision
    block_id: str
    from_tier: Tier
    to_tier: Tier
    bytes_moved: int
    move_latency_us: float
    is_prefetch: bool        # session-aware activated this move
    n_hops: int

    def __repr__(self):
        return (f"t={self.tick_t:5.0f}s {self.block_id} "
                f"{self.from_tier.name}->{self.to_tier.name} "
                f"({self.move_latency_us/1e6:.3f}s"
                f"{', prefetch' if self.is_prefetch else ''})")


@dataclass
class TierSnapshot:
    tick_t: float
    locations: Dict[str, Tier]   # block_id -> current tier
    bytes_by_tier: Dict[Tier, int]

    def n_in_tier(self, tier: Tier) -> int:
        return sum(1 for t in self.locations.values() if t == tier)


@dataclass
class SimResult:
    duration_seconds: int
    tick_seconds: int
    transitions: List[TransitionRecord]
    snapshots: List[TierSnapshot]      # one per tick
    scenario_meta: dict

    def transitions_by_block(self) -> Dict[str, List[TransitionRecord]]:
        out: Dict[str, List[TransitionRecord]] = {}
        for tr in self.transitions:
            out.setdefault(tr.block_id, []).append(tr)
        return out

    def tier_visit_counts(self) -> Dict[Tier, int]:
        """How many distinct blocks ever visited each tier (anywhere in timeline)."""
        visits = {t: set() for t in Tier}
        for snap in self.snapshots:
            for bid, tier in snap.locations.items():
                visits[tier].add(bid)
        return {t: len(s) for t, s in visits.items()}


class SimulationLoop:
    """Loop1 — tick-based clock + scheduler invocation.

    NOT modeled (Phase 2.2 scope):
    - PCIe/SSD queueing
    - Move latency advancing simulation clock
    - Block size effects on transition duration
    """

    def __init__(self,
                 timelines: List[ConvTimeline],
                 scheduler: IdleTierScheduler,
                 tick_seconds: int = 30,
                 session_window_seconds: float = 30.0):
        self.timelines = timelines
        self.scheduler = scheduler
        self.tick_seconds = tick_seconds
        self.session_window = session_window_seconds
        self.manager = scheduler.manager

    def _ensure_blocks(self):
        """Add KVBlock for each timeline to the TierManager (idempotent)."""
        for tl in self.timelines:
            if tl.conv_id not in self.manager.blocks:
                # last_accessed_unix = first turn timestamp
                first_turn = min(tl.turn_unix_timestamps)
                self.manager.add_block(KVBlock(
                    block_id=tl.conv_id,
                    size_bytes=tl.block_size_bytes,
                    location=Tier.HBM,
                    last_accessed_unix=first_turn,
                ))

    def _active_session_ids(self, t: float) -> Set[str]:
        return {tl.conv_id for tl in self.timelines
                if tl.is_active_at(t, self.session_window)}

    def _update_last_accessed(self, t: float):
        """If any turn fires at tick t, update block.last_accessed_unix."""
        for tl in self.timelines:
            for turn_ts in tl.turn_unix_timestamps:
                # If a turn happens within (t - tick, t], mark last_accessed
                if t - self.tick_seconds < turn_ts <= t:
                    block = self.manager.get_block(tl.conv_id)
                    block.last_accessed_unix = turn_ts

    def run(self, duration_seconds: int,
            scenario_meta: Optional[dict] = None) -> SimResult:
        self._ensure_blocks()
        transitions: List[TransitionRecord] = []
        snapshots: List[TierSnapshot] = []

        # Initial snapshot at t=0 (before any tick logic)
        snapshots.append(self._snapshot(0.0))

        t = self.tick_seconds
        while t <= duration_seconds:
            # 1. New user turns this tick -> update last_accessed
            self._update_last_accessed(t)
            # 2. Determine active sessions at this tick
            active = self._active_session_ids(t)
            # 3. Scheduler decides + applies moves (Phase 1 logic)
            tick_moves = self.scheduler.tick(t, active)
            # 4. Record transitions
            for bid, move_result in tick_moves:
                is_prefetch = bid in active and move_result.final_tier == Tier.HBM
                transitions.append(TransitionRecord(
                    tick_t=t,
                    block_id=bid,
                    from_tier=move_result.initial_tier,
                    to_tier=move_result.final_tier,
                    bytes_moved=move_result.bytes_moved,
                    move_latency_us=move_result.total_latency_us,
                    is_prefetch=is_prefetch,
                    n_hops=len(move_result.hops),
                ))
            # 5. Snapshot
            snapshots.append(self._snapshot(t))
            t += self.tick_seconds

        return SimResult(
            duration_seconds=duration_seconds,
            tick_seconds=self.tick_seconds,
            transitions=transitions,
            snapshots=snapshots,
            scenario_meta=scenario_meta or {},
        )

    def _snapshot(self, t: float) -> TierSnapshot:
        locs = {bid: b.location for bid, b in self.manager.blocks.items()}
        util = self.manager.get_tier_utilization()
        return TierSnapshot(tick_t=t, locations=locs, bytes_by_tier=util)


if __name__ == "__main__":
    from src.sim.event_gen import generate_for_scenario

    timelines, meta = generate_for_scenario("default_2_5h")
    cfgs = load_defaults()
    mgr = TierManager(LatencyModel(cfgs["p5520"]), LatencyModel(cfgs["p5316"]))
    sch = IdleTierScheduler(cfgs["thresholds"], mgr)
    loop = SimulationLoop(timelines, sch, tick_seconds=30)

    print(f"Running simulation: {meta['scenario']} "
          f"({meta['duration_seconds']}s, tick=30s)")
    result = loop.run(meta["duration_seconds"], scenario_meta=meta)

    print(f"\nTransitions: {len(result.transitions)}")
    print(f"Snapshots:   {len(result.snapshots)} (one per tick + initial)")

    print(f"\n--- Transition timeline (first 20 + last 5) ---")
    for tr in result.transitions[:20]:
        print(f"  {tr}")
    if len(result.transitions) > 25:
        print(f"  ... ({len(result.transitions) - 25} more) ...")
    for tr in result.transitions[-5:]:
        print(f"  {tr}")

    print(f"\n--- Tier visit counts (distinct blocks that touched each tier) ---")
    visits = result.tier_visit_counts()
    for tier in Tier:
        print(f"  {tier.name:12s}: {visits[tier]} / 8 blocks")

    print(f"\n--- Per-block transition history ---")
    by_block = result.transitions_by_block()
    for bid in sorted(by_block.keys()):
        chain = [tr.to_tier.name for tr in by_block[bid]]
        initial = by_block[bid][0].from_tier.name if by_block[bid] else "HBM"
        print(f"  {bid}: {initial} -> {' -> '.join(chain)}")

    print(f"\n--- Final tier utilization (t={meta['duration_seconds']}) ---")
    last = result.snapshots[-1]
    for tier in Tier:
        print(f"  {tier.name:12s}: {last.bytes_by_tier[tier]/1e9:5.2f} GB "
              f"({last.n_in_tier(tier)} blocks)")
