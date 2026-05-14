"""XHBM-Bench v2: Idle-driven KV cache tier emulator + scheduler.

Built on verified Solidigm spec sheet profiles (D7-P5520 active, D5-P5316 archive)
and WildChat-1M 200K measurement-based threshold config.
"""
from .profiles import (
    SSDProfile, ThresholdConfig,
    load_profile, load_thresholds, load_defaults,
)
__all__ = ["SSDProfile", "ThresholdConfig",
           "load_profile", "load_thresholds", "load_defaults"]
