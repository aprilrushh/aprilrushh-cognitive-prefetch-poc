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
    turn_unix_timestamps: List[float]   # user activity moments
    block_size_bytes: int = int(6.4e9)  # 1 conv KV at 32K tokens, V-only NF4

    def is_active_at(self, t: float, window_seconds: float = 30.0) -> bool:
        """Active session window after each turn for `window_seconds`."""
        return any(ts <= t < ts + window_seconds for ts in self.turn_unix_timestamps)


class EventGenerator(ABC):
    @abstractmethod
    def generate(self, n_conversations: int, duration_seconds: int,
                 seed: int = 0) -> List[ConvTimeline]:
        """Returns list of ConvTimeline, one per conversation."""


class UniformEventGenerator(EventGenerator):
    """All conv have turn every `turn_interval_seconds`, starting at t=0."""

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
    """Each conv starts at stagger_seconds * i, with one follow-up turn after turn_interval.

    Default scenario per Andy 2026-05-13 decisions:
    - n=8, duration=9000s (2.5h), stagger=600s, turn_interval=1500s (25min)
    - conv0 last turn at t=1500, idle at end = 9000-1500 = 7500s >= 7200s (SSD_ARCHIVE threshold)
    - 4-tier visit guaranteed
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
            ts = [start]
            if start + self.turn_interval < duration_seconds:
                ts.append(start + self.turn_interval)
            timelines.append(ConvTimeline(f"conv{i}", ts))
        return timelines


def generate_for_scenario(scenario: str = "default_2_5h",
                           seed: int = 0) -> Tuple[List[ConvTimeline], dict]:
    """Convenience factory for named scenarios. Returns (timelines, metadata)."""
    if scenario == "default_2_5h":
        gen = StaggeredUniformGenerator(stagger_seconds=600.0,
                                          turn_interval_seconds=1500.0)
        timelines = gen.generate(n_conversations=8, duration_seconds=9000, seed=seed)
        meta = {
            "scenario": "default_2_5h",
            "generator": "StaggeredUniformGenerator",
            "n_conversations": 8,
            "duration_seconds": 9000,
            "stagger_seconds": 600,
            "turn_interval_seconds": 1500,
            "decided": ("Andy 2026-05-13: 2.5h to guarantee 4-tier visit "
                        "(2h had conv0 max idle=95min < 2h archive threshold)"),
            "expected_transitions": ("4-tier full chain: conv0 (first) reaches 2h "
                                     "idle at t=1500+7200=8700 < 9000"),
        }
        return timelines, meta
    raise ValueError(f"unknown scenario: {scenario}")


if __name__ == "__main__":
    timelines, meta = generate_for_scenario("default_2_5h")
    print(f"Scenario: {meta['scenario']}")
    print(f"  duration={meta['duration_seconds']}s, n={meta['n_conversations']}, "
          f"stagger={meta['stagger_seconds']}s, turn_interval={meta['turn_interval_seconds']}s")
    print(f"\nTimelines:")
    for tl in timelines:
        print(f"  {tl.conv_id}: turns at {tl.turn_unix_timestamps}")

    print(f"\n--- Idle analysis at t={meta['duration_seconds']} ---")
    for tl in timelines:
        last_turn = max(tl.turn_unix_timestamps)
        idle = meta['duration_seconds'] - last_turn
        print(f"  {tl.conv_id}: last turn t={last_turn}, idle at end={idle}s = {idle/60:.1f}min")
