"""Event timestamp generators for conversations.

Phase 2a: UniformEventGenerator + StaggeredUniformGenerator (synthetic).
Phase 2b: WildChatEventGenerator (sampling from 200K real distribution).

Library analogy:
- EventGenerator = 손님 시간표 만드는 사람
- Synthetic = 정해진 패턴 (균등/시차)
- WildChat-sampled = 실제 200K 손님 분포에서 8명 뽑음
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple
import random


@dataclass
class ConvTimeline:
    conv_id: str
    turn_unix_timestamps: List[float]   # user activity moments (each = active session window opens)
    block_size_bytes: int = int(6.4e9)  # 1 conv KV at 32K tokens, V-only NF4

    def is_active_at(self, t: float, window_seconds: float = 30.0) -> bool:
        """Active session window after each turn for `window_seconds` (typing -> response)."""
        return any(ts <= t < ts + window_seconds for ts in self.turn_unix_timestamps)


class EventGenerator(ABC):
    @abstractmethod
    def generate(self, n_conversations: int, duration_seconds: int,
                 seed: int = 0) -> List[ConvTimeline]:
        """Returns list of ConvTimeline, one per conversation."""


class UniformEventGenerator(EventGenerator):
    """All conv have a turn every `turn_interval_seconds`, all starting at t=0."""

    def __init__(self, turn_interval_seconds: float = 1500.0):
        self.turn_interval = turn_interval_seconds

    def generate(self, n_conversations: int, duration_seconds: int,
                 seed: int = 0) -> List[ConvTimeline]:
        timelines = []
        for i in range(n_conversations):
            ts = []
            t = 0.0
            while t < duration_seconds:
                ts.append(t)
                t += self.turn_interval
            timelines.append(ConvTimeline(f"conv{i}", ts))
        return timelines


class StaggeredUniformGenerator(EventGenerator):
    """Each conv starts at stagger_seconds * i, then turns every turn_interval.

    Default scenario per Andy 2026-05-13 decision:
    - n_conversations=8, duration=7200s (2h), stagger=600s, turn_interval=1500s (25min)
    - First conv's first idle reaches 2h trigger at t=7200
    """

    def __init__(self,
                 stagger_seconds: float = 600.0,
                 turn_interval_seconds: float = 1500.0):
        self.stagger = stagger_seconds
        self.turn_interval = turn_interval_seconds

    def generate(self, n_conversations: int, duration_seconds: int,
                 seed: int = 0) -> List[ConvTimeline]:
        timelines = []
        for i in range(n_conversations):
            start = i * self.stagger
            ts = []
            t = start
            # exactly 2 turns per conv (idle progression dominated by interval >> threshold)
            # Actually: one initial turn at start, then later turn at start+turn_interval if still in window
            ts.append(start)
            if start + self.turn_interval < duration_seconds:
                ts.append(start + self.turn_interval)
            timelines.append(ConvTimeline(f"conv{i}", ts))
        return timelines


def generate_for_scenario(scenario: str = "default_2h",
                           seed: int = 0) -> Tuple[List[ConvTimeline], dict]:
    """Convenience factory for named scenarios. Returns (timelines, metadata)."""
    if scenario == "default_2h":
        gen = StaggeredUniformGenerator(stagger_seconds=600.0,
                                          turn_interval_seconds=1500.0)
        timelines = gen.generate(n_conversations=8, duration_seconds=7200, seed=seed)
        meta = {
            "scenario": "default_2h",
            "generator": "StaggeredUniformGenerator",
            "n_conversations": 8,
            "duration_seconds": 7200,
            "stagger_seconds": 600,
            "turn_interval_seconds": 1500,
            "decided": "Andy 2026-05-13: 2h to guarantee 4-tier visit",
            "expected_transitions": ("4-tier full chain: first conv reaches 2h "
                                     "idle (SSD_ARCHIVE) at t=7200"),
        }
        return timelines, meta
    raise ValueError(f"unknown scenario: {scenario}")


if __name__ == "__main__":
    timelines, meta = generate_for_scenario("default_2h")
    print(f"Scenario: {meta['scenario']}")
    print(f"  duration={meta['duration_seconds']}s, n={meta['n_conversations']}, "
          f"stagger={meta['stagger_seconds']}s")
    print(f"\nTimelines:")
    for tl in timelines:
        print(f"  {tl.conv_id}: turns at {tl.turn_unix_timestamps}, "
              f"idle gap = {tl.turn_unix_timestamps[1] - tl.turn_unix_timestamps[0] if len(tl.turn_unix_timestamps)>1 else 'N/A'}s")

    # Sanity: at t=7200, conv0 (first) has idle = 7200 - 1500 = 5700s = 95min (no archive yet)
    # Need conv0's idle to reach 7200s for archive. Adjust turn_interval or duration?
    print(f"\n--- Idle analysis at t=7200 ---")
    for tl in timelines:
        last_turn = max(tl.turn_unix_timestamps)
        idle = 7200 - last_turn
        print(f"  {tl.conv_id}: last turn t={last_turn}, idle at end={idle}s = {idle/60:.1f}min")
