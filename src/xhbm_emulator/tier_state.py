"""Tier state machine: KV block location tracking + move latency.

Library analogy (회사 narrative anchor):
- HBM        = 책상 (GPU memory)
- RAM        = 책장 (DDR5)
- SSD_ACTIVE = 창고 1 (D7-P5520)
- SSD_ARCHIVE= 깊은 창고 (D5-P5316)

KV block unit: 1 block = 1 conversation (~6.4 GB at 32K tokens, V-only NF4).
Move routing: SSD↔SSD must go via RAM (no peer-to-peer SSD DMA in our model).

PCIe Gen5 x16 for HBM↔RAM (H100 host link):
- Theoretical: 63 GB/s (128b/130b encoded)
- Practical: ~50 GB/s estimate (TLP/DLLP overhead, deployment-specific)
NOTE: Practical value is conservative estimate. Verify per-deployment with
      `lspci -s <gpu> -vvv | grep LnkSta` for actual link width/speed.
"""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional
from .latency_model import LatencyModel


PCIE_GEN5_X16_THEORETICAL_MBPS = 63040   # 32 GT/s × 16 / 8 × (128/130)
PCIE_GEN5_X16_PRACTICAL_MBPS   = 50000   # H100 host-link conservative estimate


class Tier(IntEnum):
    HBM         = 0   # GPU memory
    RAM         = 1   # DDR5
    SSD_ACTIVE  = 2   # D7-P5520
    SSD_ARCHIVE = 3   # D5-P5316


@dataclass
class KVBlock:
    """A conversation's KV cache as a movable unit."""
    block_id: str               # conversation_id (session)
    size_bytes: int             # total KV bytes (sum over layers, V-only NF4)
    location: Tier
    last_accessed_unix: float   # last user activity (turn timestamp)
    n_layers: int = 80
    metadata: dict = field(default_factory=dict)

    @property
    def size_GB(self) -> float:
        return self.size_bytes / 1e9


@dataclass
class MoveHop:
    from_tier: Tier
    to_tier: Tier
    latency_us: float
    note: str = ""

    def __repr__(self):
        return f"{self.from_tier.name}->{self.to_tier.name}={self.latency_us/1000:.2f}ms"


@dataclass
class MoveResult:
    block_id: str
    bytes_moved: int
    initial_tier: Tier
    final_tier: Tier
    total_latency_us: float
    hops: List[MoveHop] = field(default_factory=list)

    def __repr__(self):
        return (f"Move(block={self.block_id}, {self.initial_tier.name}->"
                f"{self.final_tier.name}, {self.bytes_moved/1e9:.2f}GB, "
                f"total={self.total_latency_us/1e6:.3f}s, hops={len(self.hops)})")


class TierManager:
    """Tracks KV block locations and computes move latency.

    No actual KV tensor data here - this is the *measurement framework* layer.
    Real KV ops happen in Phase 2 scheduler (vLLM/transformers integration).
    """

    def __init__(self,
                 p5520_model: LatencyModel,
                 p5316_model: LatencyModel,
                 hbm_ram_pcie_MBps: float = PCIE_GEN5_X16_PRACTICAL_MBPS):
        self.p5520 = p5520_model
        self.p5316 = p5316_model
        self.hbm_ram_bw_MBps = hbm_ram_pcie_MBps
        self.blocks: Dict[str, KVBlock] = {}

    def add_block(self, block: KVBlock):
        if block.block_id in self.blocks:
            raise ValueError(f"block_id {block.block_id} already tracked")
        self.blocks[block.block_id] = block

    def get_block(self, block_id: str) -> KVBlock:
        return self.blocks[block_id]

    def get_blocks_by_tier(self, tier: Tier) -> List[KVBlock]:
        return [b for b in self.blocks.values() if b.location == tier]

    def get_tier_utilization(self) -> Dict[Tier, int]:
        """Total bytes in each tier."""
        util = {t: 0 for t in Tier}
        for b in self.blocks.values():
            util[b.location] += b.size_bytes
        return util

    def _hop_latency_us(self, src: Tier, dst: Tier, size_bytes: int,
                        aligned: bool = True) -> tuple:
        """Single-hop latency between adjacent tiers."""
        pair = (src, dst)
        if pair in [(Tier.HBM, Tier.RAM), (Tier.RAM, Tier.HBM)]:
            return size_bytes / self.hbm_ram_bw_MBps, "PCIe Gen5 x16"
        elif pair == (Tier.RAM, Tier.SSD_ACTIVE):
            r = self.p5520.seq_write_latency_us(size_bytes, aligned=True)
            return r.total_us, "P5520 seq_write"
        elif pair == (Tier.SSD_ACTIVE, Tier.RAM):
            r = self.p5520.seq_read_latency_us(size_bytes)
            return r.total_us, "P5520 seq_read"
        elif pair == (Tier.RAM, Tier.SSD_ARCHIVE):
            r = self.p5316.seq_write_latency_us(size_bytes, aligned=aligned)
            return r.total_us, f"P5316 seq_write {'aligned' if aligned else 'misaligned'}"
        elif pair == (Tier.SSD_ARCHIVE, Tier.RAM):
            r = self.p5316.seq_read_latency_us(size_bytes)
            return r.total_us, "P5316 seq_read"
        else:
            raise ValueError(f"No direct hop {src.name} -> {dst.name} (use compute_path)")

    def _compute_path(self, src: Tier, dst: Tier) -> List[Tier]:
        """Route between any two tiers. SSD<->SSD goes via RAM."""
        if src == dst:
            return [src]
        direct = {
            (Tier.HBM, Tier.RAM), (Tier.RAM, Tier.HBM),
            (Tier.RAM, Tier.SSD_ACTIVE), (Tier.SSD_ACTIVE, Tier.RAM),
            (Tier.RAM, Tier.SSD_ARCHIVE), (Tier.SSD_ARCHIVE, Tier.RAM),
        }
        if (src, dst) in direct:
            return [src, dst]
        return [src, Tier.RAM, dst]

    def move(self, block_id: str, target: Tier,
             aligned: bool = True) -> MoveResult:
        """Move block to target tier. Auto-routes via RAM if needed."""
        block = self.blocks[block_id]
        path = self._compute_path(block.location, target)
        total_us = 0.0
        hops = []
        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            lat, note = self._hop_latency_us(src, dst, block.size_bytes, aligned)
            hops.append(MoveHop(src, dst, lat, note))
            total_us += lat
        initial = block.location
        block.location = target
        return MoveResult(block_id, block.size_bytes, initial, target, total_us, hops)


if __name__ == "__main__":
    from .profiles import load_defaults
    cfgs = load_defaults()
    p5520, p5316 = cfgs["p5520"], cfgs["p5316"]
    tm = TierManager(LatencyModel(p5520), LatencyModel(p5316))

    # 8 conversations × 6.4 GB each (matching PDF Page 7 / 49 GB peak)
    conv_size = int(6.4e9)
    for i in range(8):
        tm.add_block(KVBlock(f"conv{i}", conv_size, Tier.HBM, last_accessed_unix=0))

    print("=" * 70)
    print("Initial state: 8 conv × 6.4 GB all in HBM")
    print("=" * 70)
    for t, b in tm.get_tier_utilization().items():
        print(f"  {t.name:12s}: {b/1e9:.2f} GB")

    print("\n--- conv0: HBM → RAM (30s idle) ---")
    r = tm.move("conv0", Tier.RAM)
    print(f"  {r}")
    for h in r.hops: print(f"    {h}: {h.note}")

    print("\n--- conv0: RAM → SSD_ACTIVE (15min idle) ---")
    r = tm.move("conv0", Tier.SSD_ACTIVE)
    print(f"  {r}")
    for h in r.hops: print(f"    {h}: {h.note}")

    print("\n--- conv0: SSD_ACTIVE → SSD_ARCHIVE (2h idle, aligned=True) ---")
    r = tm.move("conv0", Tier.SSD_ARCHIVE, aligned=True)
    print(f"  {r}")
    for h in r.hops: print(f"    {h}: {h.note}")

    print("\n--- conv1: same path with aligned=False (mis-batched demote) ---")
    tm.move("conv1", Tier.RAM)
    tm.move("conv1", Tier.SSD_ACTIVE)
    r = tm.move("conv1", Tier.SSD_ARCHIVE, aligned=False)
    print(f"  {r}")
    for h in r.hops: print(f"    {h}: {h.note}")

    print("\n--- M3 prefetch: conv0 SSD_ARCHIVE → HBM (typing prefetch) ---")
    r = tm.move("conv0", Tier.HBM)
    print(f"  {r}")
    for h in r.hops: print(f"    {h}: {h.note}")
    print(f"  ✅ within 10-30s typing window" if r.total_latency_us/1e6 < 10 else "  ⚠ exceeds")

    print("\n=== Final tier utilization ===")
    for t, b in tm.get_tier_utilization().items():
        n = len(tm.get_blocks_by_tier(t))
        print(f"  {t.name:12s}: {b/1e9:5.2f} GB ({n} blocks)")
