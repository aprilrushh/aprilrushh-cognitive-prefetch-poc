"""Phase 2: event-driven simulation layer on top of xhbm_emulator.

Modules:
- event_gen: generate conversation turn timestamps (synthetic or sampled)
- loop:      run scheduler over the timeline, record transitions
- gantt:     produce Gantt-style trace (block × time -> tier)
"""
from .event_gen import (
    EventGenerator, UniformEventGenerator, StaggeredUniformGenerator,
    generate_for_scenario,
)
__all__ = ["EventGenerator", "UniformEventGenerator",
           "StaggeredUniformGenerator", "generate_for_scenario"]
