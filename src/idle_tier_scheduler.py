"""Idle-driven KV cache tier scheduler (Phase 1 skeleton).

Decision logic: pure function `decide(block, current_unix, active_session_ids)`.
Phase 2 will add full event-driven simulation loop on top of this skeleton.

Library analogy:
- Skeleton 단계 = 사서가 시계 보며 책 자동 정리하는 *판단 로직*
- Phase 2 = 시계 자체 + Gantt timeline + ablation knobs

Session-aware boundary (critical):
- If block's conversation is in an active session -> ALWAYS prefetch to HBM
- Inter-conv idle only triggers SSD demotion (no within-conv punishment)
- This is the most important invariant - per anchor § 16 D session_aware spec
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Set, List, Tuple, Dict
from src.xhbm_emulator import (Tier, KVBlock, MoveResult,
                                TierManager, ThresholdConfig)


@dataclass
class SchedulerStats:
    transitions: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    prefetch_triggers: int = 0    # session-activation -> HBM moves
    ssd_to_archive_count: int = 0
    total_ticks: int = 0

    def record(self, result: MoveResult, is_prefetch: bool):
        key = f"{result.initial_tier.name}->{result.final_tier.name}"
        self.transitions[key] += 1
        if is_prefetch: self.prefetch_triggers += 1
        if result.final_tier == Tier.SSD_ARCHIVE:
            self.ssd_to_archive_count += 1

    def summary(self) -> str:
        lines = [f"  ticks: {self.total_ticks}",
                 f"  prefetch_triggers (active session detected): {self.prefetch_triggers}",
                 f"  total demotes to SSD_ARCHIVE: {self.ssd_to_archive_count}",
                 "  transition counts:"]
        for k, v in sorted(self.transitions.items()):
            lines.append(f"    {k}: {v}")
        return "\n".join(lines)


class IdleTierScheduler:
    """Phase 1 skeleton: threshold ladder + session-aware decide."""

    def __init__(self, thresholds: ThresholdConfig, manager: TierManager):
        self.thresholds = thresholds
        self.manager = manager
        self.stats = SchedulerStats()

    def _target_tier_from_idle(self, idle_seconds: float) -> Tier:
        """Pure ladder lookup."""
        if idle_seconds >= self.thresholds.ssd_to_archive_seconds:
            return Tier.SSD_ARCHIVE
        if idle_seconds >= self.thresholds.ram_to_ssd_seconds:
            return Tier.SSD_ACTIVE
        if idle_seconds >= self.thresholds.hbm_to_ram_seconds:
            return Tier.RAM
        return Tier.HBM

    def decide(self, block: KVBlock, current_unix: float,
               active_session_ids: Set[str]) -> Optional[Tier]:
        """Returns target tier if transition needed, else None.

        Session-aware: if block.block_id in active_session_ids,
        always return HBM (or None if already there).
        """
        if not self.thresholds.session_aware:
            # Defensive: scheduler enforces session_aware unconditionally
            pass
        if block.block_id in active_session_ids:
            return Tier.HBM if block.location != Tier.HBM else None
        idle = current_unix - block.last_accessed_unix
        target = self._target_tier_from_idle(idle)
        return target if target != block.location else None

    def tick(self, current_unix: float,
             active_session_ids: Optional[Set[str]] = None) -> List[Tuple[str, MoveResult]]:
        """Apply scheduler to all tracked blocks at this timestamp.

        Returns list of (block_id, MoveResult) for transitions that fired.
        Each move is treated as instantaneous at this tick (latency recorded
        in MoveResult but doesn't advance scheduler time - Phase 2 fixes this).
        """
        active = active_session_ids or set()
        transitions = []
        for block in list(self.manager.blocks.values()):
            target = self.decide(block, current_unix, active)
            if target is None:
                continue
            is_prefetch = (block.block_id in active and
                           block.location in (Tier.SSD_ACTIVE, Tier.SSD_ARCHIVE,
                                              Tier.RAM))
            aligned = True   # default; Phase 2 will add per-block knob
            result = self.manager.move(block.block_id, target, aligned=aligned)
            self.stats.record(result, is_prefetch)
            transitions.append((block.block_id, result))
        self.stats.total_ticks += 1
        return transitions


if __name__ == "__main__":
    from src.xhbm_emulator import LatencyModel, load_defaults
    cfgs = load_defaults()
    mgr = TierManager(LatencyModel(cfgs["p5520"]), LatencyModel(cfgs["p5316"]))
    sch = IdleTierScheduler(cfgs["thresholds"], mgr)

    # --- Scenario 1: Single conversation idle progression ---
    print("=" * 70)
    print("Scenario 1: 1 conv, idle progression (t=0..t=8000)")
    print(f"Thresholds: {cfgs['thresholds']}")
    print("=" * 70)
    mgr.add_block(KVBlock("conv0", int(6.4e9), Tier.HBM, last_accessed_unix=0))
    checkpoints = [10, 40, 300, 900, 1500, 3600, 7300, 8000]
    activate_at = {8000: {"conv0"}}   # user types at t=8000
    for t in checkpoints:
        active = activate_at.get(t, set())
        trans = sch.tick(t, active)
        loc = mgr.get_block("conv0").location.name
        idle = t - mgr.get_block("conv0").last_accessed_unix if "conv0" not in active else 0
        sess = " [ACTIVE]" if "conv0" in active else ""
        if trans:
            for _, r in trans:
                print(f"  t={t:5d}s (idle~{idle:5.0f}s){sess}: {r}")
        else:
            print(f"  t={t:5d}s (idle~{idle:5.0f}s){sess}: no transition, location={loc}")
        # Reset last_accessed if user activated this turn
        if "conv0" in active:
            mgr.get_block("conv0").last_accessed_unix = t

    print(f"\n--- Scenario 1 stats ---\n{sch.stats.summary()}")

    # --- Scenario 2: 4 conv with staggered idle ---
    print("\n" + "=" * 70)
    print("Scenario 2: 4 conv staggered idle, evaluated at t=2000s")
    print("=" * 70)
    mgr2 = TierManager(LatencyModel(cfgs["p5520"]), LatencyModel(cfgs["p5316"]))
    sch2 = IdleTierScheduler(cfgs["thresholds"], mgr2)
    convs = [("c0", 0),       # idle 2000s = 33min -> SSD_ACTIVE
             ("c1", 60),      # idle 1940s = 32min -> SSD_ACTIVE
             ("c2", 1700),    # idle 300s = 5min -> RAM
             ("c3", 1990)]    # idle 10s -> HBM
    for cid, last in convs:
        mgr2.add_block(KVBlock(cid, int(6.4e9), Tier.HBM, last_accessed_unix=last))
    trans = sch2.tick(2000, active_session_ids=set())
    for _, r in trans:
        print(f"  {r}")
    print("\nFinal locations:")
    for cid, _ in convs:
        b = mgr2.get_block(cid)
        idle_s = 2000 - b.last_accessed_unix
        print(f"  {cid}: idle={idle_s}s -> {b.location.name}")
    util = mgr2.get_tier_utilization()
    print("\nTier utilization:")
    for t in Tier:
        n = len(mgr2.get_blocks_by_tier(t))
        print(f"  {t.name:12s}: {util[t]/1e9:5.2f} GB ({n} blocks)")
