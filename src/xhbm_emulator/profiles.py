"""SSD profile + threshold config loader for XHBM-Bench v2.

Loads JSON spec sheet profiles (D7-P5520, D5-P5316) and measurement-based
threshold config from configs/. All values verifiable against external sources
(Solidigm Product Brief, StorageReview, WildChat-1M 200K conversations).

Library analogy:
- HBM = desk (GPU memory, 80GB on H100)
- RAM = bookshelf (DDR5, 221GB on this box)
- SSD-Active (D7-P5520) = warehouse (close, frequent access, IU=4KB)
- SSD-Archive (D5-P5316) = deep warehouse (far, rare access, IU=64KB!)
"""
from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Optional, Dict, Any

DEFAULT_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


@dataclass
class SSDProfile:
    """A single SSD product profile derived from public spec sheet.

    Indirection unit matters: 4KB (TLC P5520) vs 64KB (QLC P5316).
    Writes to QLC must be 64K-aligned or write amplification is 2-4x.
    """
    name: str
    tier_role: str           # "active" or "archive"
    tier_label: str          # "warehouse" or "deep_storage"
    interface: str           # "PCIe Gen4 x4 NVMe"
    nand_type: str           # "TLC" or "QLC"
    seq_read_MBps: float
    seq_write_MBps: float
    rand_read_4k_IOPS: float
    rand_write_4k_IOPS: float
    indirection_unit_KB: int
    qd1_read_latency_us: float
    dwpd: float
    verified_on: str
    sources: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @property
    def is_qlc(self) -> bool:
        return self.nand_type.upper() == "QLC"

    @property
    def needs_64k_alignment(self) -> bool:
        """QLC w/ IU=64KB: writes must align to 64K or 2-4x amplification."""
        return self.indirection_unit_KB >= 64

    def __repr__(self):
        return (f"SSDProfile({self.name}, tier={self.tier_role}, "
                f"R={self.seq_read_MBps}MB/s, IU={self.indirection_unit_KB}KB)")


@dataclass
class ThresholdConfig:
    """Idle-driven tier promotion thresholds (WildChat-1M evidence base)."""
    hbm_to_ram_seconds: int
    ram_to_ssd_seconds: int
    ssd_to_archive_seconds: int
    session_aware: bool
    configurable: bool
    evidence_dataset: str
    evidence_sample_size: int
    decided_on: str
    raw: dict = field(default_factory=dict)

    def __repr__(self):
        return (f"ThresholdConfig(HBM->RAM={self.hbm_to_ram_seconds}s, "
                f"RAM->SSD={self.ram_to_ssd_seconds//60}min, "
                f"SSD->Archive={self.ssd_to_archive_seconds//3600}h, "
                f"session_aware={self.session_aware})")


def load_profile(path) -> SSDProfile:
    path = Path(path)
    with open(path) as f:
        d = json.load(f)
    seq = d["sequential"]; rnd = d["random_4k"]; lat = d["latency_us"]
    return SSDProfile(
        name=d["name"],
        tier_role=d["tier_role"],
        tier_label=d["tier_label"],
        interface=d["interface"],
        nand_type=d["nand"]["type"],
        seq_read_MBps=seq["read_MBps"],
        seq_write_MBps=(seq.get("write_MBps")
                        or seq.get("write_MBps_30TB")
                        or seq.get("write_MBps_15TB")),
        rand_read_4k_IOPS=rnd["read_IOPS"],
        rand_write_4k_IOPS=(rnd.get("write_IOPS_spec")
                            or rnd.get("write_IOPS_steady_misaligned")),
        indirection_unit_KB=d["indirection_unit_KB"],
        qd1_read_latency_us=(lat.get("qd1_read_4nines")
                             or lat.get("qd1_read_99_999")),
        dwpd=d["endurance"]["dwpd"],
        verified_on=d["verified_on"],
        sources=d.get("sources", []),
        raw=d,
    )


def load_thresholds(path) -> ThresholdConfig:
    path = Path(path)
    with open(path) as f:
        d = json.load(f)
    t = d["thresholds"]; e = d["evidence_base"]
    return ThresholdConfig(
        hbm_to_ram_seconds=t["hbm_to_ram"]["seconds"],
        ram_to_ssd_seconds=t["ram_to_ssd_active"]["seconds"],
        ssd_to_archive_seconds=t["ssd_active_to_archive"]["seconds"],
        session_aware=d["session_aware"],
        configurable=d["configurable"],
        evidence_dataset=e["dataset"],
        evidence_sample_size=e["sample_size_conversations"],
        decided_on=d["decided_on"],
        raw=d,
    )


def load_defaults(configs_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Single entry point: returns {'p5520', 'p5316', 'thresholds'}."""
    base = Path(configs_dir) if configs_dir else DEFAULT_CONFIGS_DIR
    return {
        "p5520": load_profile(base / "profiles" / "d7_p5520.json"),
        "p5316": load_profile(base / "profiles" / "d5_p5316.json"),
        "thresholds": load_thresholds(base / "scheduler" / "default_thresholds.json"),
    }


if __name__ == "__main__":
    cfgs = load_defaults()
    print("--- SSD profiles ---")
    for label, p in [("Active", cfgs["p5520"]), ("Archive", cfgs["p5316"])]:
        print(f"  {label}: {p}")
        print(f"    is_qlc={p.is_qlc}, needs_64k_alignment={p.needs_64k_alignment}")
        print(f"    verified_on={p.verified_on}, sources={len(p.sources)}")
    print("\n--- Thresholds ---")
    th = cfgs["thresholds"]
    print(f"  {th}")
    print(f"  evidence: {th.evidence_dataset} (n={th.evidence_sample_size})")
    print(f"  decided_on: {th.decided_on}")
