"""Latency model: spec sheet -> microsecond latency for SSD ops.

Fidelity level: L2 (throughput-based + PCIe contention + write amplification).
Andy decision 2026-05-13: PCIe + write amp BOTH ON, narrative-aligned.

Library analogy:
- Spec sheet seq R/W MB/s = "복도 포함 책 한 권 가져오는 시간 (single device)"
- PCIe Gen4 x4 ceiling = "복도 자체의 폭" (multi-device contention 시 별도)
- P5316 IU=64KB = "창고에 책 64KB 묶음으로만 받음" (misaligned -> amp)

What we DO model:
- Sequential read/write: size / spec throughput
- Random 4K read/write: 1 / spec IOPS per op
- D5-P5316 (QLC) write amplification: 64K-aligned (510 MB/s) vs misaligned
  (17K-19K IOPS - spec already reflects amplification)
- PCIe Gen4 x4 contention (multi-device): theoretical 7.88 GB/s,
  practical 7.25 GB/s (web-verified 2026-05-13)

What we DO NOT model (limitations - external communication 시 disclose 필수):
- Queueing dynamics (Little's law NOT used)
- Thermal throttling, NAND wear, GC scheduling
- DRAM cache, FTL internals (CSAL etc.)
- Multi-tenant fairness / QoS knobs
- Burst vs steady-state distinction beyond spec
"""
from dataclasses import dataclass
from .profiles import SSDProfile


# PCIe Gen4 x4 effective bandwidth (verified web 2026-05-13)
# Source: simcentric.com, xda-developers.com, passmark.com
PCIE_GEN4_X4_THEORETICAL_MBPS = 7880   # 128b/130b encoded
PCIE_GEN4_X4_PRACTICAL_MBPS   = 7250   # after TLP/DLLP overhead
PCIE_GEN5_X4_THEORETICAL_MBPS = 15760  # 2x Gen4


@dataclass
class LatencyResult:
    """Result of a latency lookup, with breakdown for transparency."""
    op_type: str
    size_bytes: int
    total_us: float
    spec_us: float
    pcie_us: float
    aligned: bool
    note: str = ""

    def __repr__(self):
        return (f"LatencyResult({self.op_type} {self.size_bytes}B "
                f"= {self.total_us:.1f}μs, spec={self.spec_us:.1f}, "
                f"pcie={self.pcie_us:.1f}, aligned={self.aligned})")


class LatencyModel:
    """Latency estimator from an SSD profile + PCIe environment."""

    def __init__(self,
                 profile: SSDProfile,
                 pcie_effective_MBps: float = PCIE_GEN4_X4_PRACTICAL_MBPS,
                 n_concurrent_devices: int = 1):
        """
        Args:
            profile: SSDProfile (D7-P5520 or D5-P5316).
            pcie_effective_MBps: practical PCIe BW. Default Gen4 x4 practical.
            n_concurrent_devices: SSDs sharing one PCIe root.
                1 = single-device (spec already includes PCIe).
                >1 = each gets 1/n of effective bandwidth.
        """
        self.profile = profile
        self.pcie_effective_MBps = pcie_effective_MBps
        self.n_concurrent_devices = n_concurrent_devices

    @property
    def pcie_share_MBps(self) -> float:
        return self.pcie_effective_MBps / self.n_concurrent_devices

    def _aggregate(self, spec_us: float, size_bytes: int):
        """Combine spec time with PCIe contention. Returns (total, pcie_extra)."""
        if self.n_concurrent_devices <= 1:
            return spec_us, 0.0
        pcie_us = size_bytes / (self.pcie_share_MBps * 1e6) * 1e6
        return max(spec_us, pcie_us), pcie_us

    # --- public latency methods ---

    def seq_read_latency_us(self, size_bytes: int) -> LatencyResult:
        spec = size_bytes / (self.profile.seq_read_MBps * 1e6) * 1e6
        total, pcie = self._aggregate(spec, size_bytes)
        return LatencyResult("seq_read", size_bytes, total, spec, pcie, True)

    def seq_write_latency_us(self, size_bytes: int, aligned: bool = True) -> LatencyResult:
        note = ""
        if self.profile.needs_64k_alignment and not aligned:
            # QLC misaligned: fall back to 4K IOPS path (spec covers amp)
            n_4k = max(1, (size_bytes + 4095) // 4096)
            spec = n_4k * 1e6 / self.profile.rand_write_4k_IOPS
            note = "QLC misaligned -> 4K IOPS path (amplification included)"
        elif self.profile.needs_64k_alignment and aligned:
            aligned_MBps = self.profile.raw.get("random_4k", {}).get("write_MBps_64K_aligned")
            if aligned_MBps:
                spec = size_bytes / aligned_MBps  # = size_bytes / (MBps*1e6)*1e6
                note = f"QLC 64K-aligned ({aligned_MBps} MB/s)"
            else:
                spec = size_bytes / (self.profile.seq_write_MBps * 1e6) * 1e6
        else:
            spec = size_bytes / (self.profile.seq_write_MBps * 1e6) * 1e6
        total, pcie = self._aggregate(spec, size_bytes)
        return LatencyResult("seq_write", size_bytes, total, spec, pcie, aligned, note)

    def random_read_latency_us(self, size_bytes: int) -> LatencyResult:
        n_4k = max(1, (size_bytes + 4095) // 4096)
        spec = n_4k * 1e6 / self.profile.rand_read_4k_IOPS
        total, pcie = self._aggregate(spec, size_bytes)
        return LatencyResult("rand_read", size_bytes, total, spec, pcie, True)

    def random_write_latency_us(self, size_bytes: int,
                                 aligned: bool = True) -> LatencyResult:
        note = ""
        if self.profile.needs_64k_alignment and not aligned:
            n_4k = max(1, (size_bytes + 4095) // 4096)
            spec = n_4k * 1e6 / self.profile.rand_write_4k_IOPS
            note = "QLC misaligned (steady-state IOPS, amp included)"
        elif self.profile.needs_64k_alignment and aligned:
            aligned_MBps = self.profile.raw.get("random_4k", {}).get("write_MBps_64K_aligned")
            if aligned_MBps:
                spec = size_bytes / aligned_MBps
                note = f"QLC 64K-aligned ({aligned_MBps} MB/s)"
            else:
                n_4k = max(1, (size_bytes + 4095) // 4096)
                spec = n_4k * 1e6 / self.profile.rand_write_4k_IOPS
        else:
            n_4k = max(1, (size_bytes + 4095) // 4096)
            spec = n_4k * 1e6 / self.profile.rand_write_4k_IOPS
        total, pcie = self._aggregate(spec, size_bytes)
        return LatencyResult("rand_write", size_bytes, total, spec, pcie, aligned, note)

    def qd1_read_latency_us(self) -> float:
        return self.profile.qd1_read_latency_us


if __name__ == "__main__":
    from .profiles import load_defaults
    cfgs = load_defaults()
    p5520 = cfgs["p5520"]; p5316 = cfgs["p5316"]

    print("=" * 70)
    print("Single-device latency model (PCIe overhead already in spec)")
    print("=" * 70)
    for label, prof in [("D7-P5520 (Active TLC)", p5520),
                         ("D5-P5316 (Archive QLC)", p5316)]:
        m = LatencyModel(prof)
        print(f"\n--- {label} ---")
        print(f"  1 GB seq read:  {m.seq_read_latency_us(2**30)}")
        print(f"  6.4 GB conv KV: {m.seq_read_latency_us(int(6.4e9))}")
        print(f"  64 KB write (aligned):    {m.random_write_latency_us(65536, True)}")
        print(f"  64 KB write (misaligned): {m.random_write_latency_us(65536, False)}")
        print(f"  qd1 read floor: {m.qd1_read_latency_us():.0f} μs")

    print("\n" + "=" * 70)
    print("M3 prefetch viability (1 conv KV from P5520, typing window 10-30s)")
    print("=" * 70)
    m = LatencyModel(p5520)
    r = m.seq_read_latency_us(int(6.4e9))
    print(f"  6.4 GB conv KV from P5520: {r.total_us/1e6:.3f} s")
    print(f"  {'✅' if r.total_us/1e6 < 10 else '⚠'} {'fits' if r.total_us/1e6<10 else 'exceeds'} typing window")

    print("\n--- PCIe contention: 8 SSDs sharing Gen4 x4 root ---")
    m8 = LatencyModel(p5520, n_concurrent_devices=8)
    r8 = m8.seq_read_latency_us(int(1e9))
    print(f"  1 GB read with 8-way share: {r8}")
    print(f"  per-device pcie_share = {m8.pcie_share_MBps:.0f} MB/s")
