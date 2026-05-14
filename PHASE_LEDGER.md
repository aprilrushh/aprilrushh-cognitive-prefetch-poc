# Phase 1-2 Implementation Ledger

H100 Idle-Driven Benchmark, 2026-05-13 1일 작업.

- **상세 ledger (발표자료 base)**: <https://www.notion.so/360c78cb12ce81808adfe8ada4e5cdc7>
- **의사결정 ledger (echo chamber 정정 + threshold revision)**: <https://www.notion.so/35fc78cb12ce8163a036ede051c591a5>

## Overview

- Branch: `idle-promotion-bench`
- 9 commits + 1 fixup + 2 tags (`phase-1-complete`, `phase-2-implementation-complete`)
- ~1,373 LOC + 16 KB Gantt JSON 자산
- 54 invariant assertions all PASS (32 Phase 1 + 22 Phase 2)

회사 narrative § 13 의 두 정직성 anchor:
- *"우리는 bandwidth 를 크게 한 거지, latency 를 개선한 게 아니다"*
- *"우리는 5분이라고 추측한 게 아니라, 200K 실측 분포 위에서 결정했다"*

두 anchor 가 *코드 자체에 박힌 measurement framework* 의 base.

## Phase 1: Emulator Framework (tag `phase-1-complete`)

| Step | Commit | Module | LOC | Invariants |
|---|---|---|---|---|
| 1.1 | cb1192a | configs (D7-P5520 + D5-P5316 + threshold v1.2) | (JSON) | (config) |
| 1.2 | 597d08b | src/xhbm_emulator/profiles.py | 145 | 9 |
| 1.3 | 73485cb | src/xhbm_emulator/latency_model.py | 171 | 6 |
| 1.4 | d51b86c | src/xhbm_emulator/tier_state.py | 209 | 8 |
| 1.5 | 5379f93 | src/idle_tier_scheduler.py (skeleton) | 160 | 9 |

**Verified findings**:
- M3 prefetch 6.4 GB SSD_ARCHIVE → HBM = **1.042 s** (typing window 10-30 s 의 1/10~1/30 여유) → "사람은 기다린 적 없음" 의 *코드 수준 증명*
- P5316 misaligned/aligned demote = **6.9× ratio** (941 μs vs 128 μs) → *vendor-cooperation insight visible signature*
- Session-aware boundary 강제 (within-conv idle 은 SSD trigger 안 함)
- 4-tier full chain (HBM → RAM → SSD_ACTIVE → SSD_ARCHIVE → HBM prefetch) 검증

## Phase 2: Simulation + Gantt (tag `phase-2-implementation-complete`)

| Step | Commit | Module | Note |
|---|---|---|---|
| 2.1 | bffc9fa | src/sim/event_gen.py v1 | 135 LOC, 7 invariants. default_2h |
| 2.1b | b43f3b9 | src/sim/event_gen.py fix | 4-tier visit 보장: 2h → 2.5h |
| 2.2 | 0f93c78 | src/sim/loop.py (Loop1) | 209 LOC, 8 invariants. tick=30s, latency 기록만 |
| 2.3a | 605d0ff | src/sim/gantt.py + scripts/m2_synthetic.py | 984 LOC inc 16KB JSON, 7 invariants |

**결과 자산**:
- `results/m2/m2_synthetic_default_2_5h.json` (Gantt data, 발표자료 측정 base)
- Gantt visualization (chat artifact)

**Key findings** (Gantt 가 드러낸 fact):
- 41 transitions / 8 prefetches / 4-tier visit (HBM=8, RAM=8, SSD_ACTIVE=8, **SSD_ARCHIVE=1/8**)
- 모든 conv 동일한 6-stage cycle: HBM → RAM (870s) → SSD_ACTIVE (600s) → ● prefetch → HBM (30s) → RAM (870s) → SSD_ACTIVE → ...
- **conv0 만 SSD_ARCHIVE** 도달 (1/8) — *예외적 사건* visible 증명
- 매 600s 마다 **P5520 concurrent read + write** (prefetch + demote) — *vendor-cooperation insight, Wayne Gao 접촉 시 공동 R&D 영역*

## Limitations (정직성 boundary, 발표자료 anchor)

**Phase 1**:
- Queueing dynamics 미모델 (Little's law 미사용)
- Thermal throttling, NAND wear, GC scheduling 미모델
- DRAM cache, FTL internals (CSAL 등) 미모델
- Multi-tenant fairness / QoS 미모델

**Phase 2** (Loop1 의 의도적 scope):
- Move latency 기록만, simulation clock advance 안 함 → *PCIe/SSD queue contention 미모델*
- 각 transition = tick boundary 의 instantaneous event (max <30s jitter)
- Synthetic uniform pattern (Phase 2a) — *real distribution 아님*. Phase 2b WildChat sampling 보류
- Block size 추정 = 6.4 GB per conv (Llama 70B, 32K tokens, V-only NF4) — *측정값 아닌 spec-derived*

## Next Steps

- **Phase 3 (다음 작업)**: M1-M4 실측 (실제 Llama 70B + Sherlock workload)
- Phase 2.3b (선택): Loop2 queue contention modeling — narrative 안 바꿈, 정확도 layer
- Phase 2.4-5 (선택): WildChat-1M sampling — real distribution proxy
- Phase 4 (Phase 3 후): visualization + Solidigm 방문 자료 (1-page PDF + dashboard)

## Git history (idle-promotion-bench branch)

    605d0ff feat(phase2.3a): Gantt JSON output + M2 entry point
    0f93c78 feat(phase2.2): src/sim/loop.py - Loop1 (clock + scheduler + log)
    b43f3b9 fix(phase2.1): rewrite event_gen.py with default_2_5h scenario
    bffc9fa feat(phase2.1): src/sim/event_gen - timeline generators (2a synthetic)
    5379f93 feat(phase1): idle_tier_scheduler skeleton - Phase 1 완결  <- tag phase-1-complete
    d51b86c feat(phase1): xhbm_emulator.tier_state - block location + move
    73485cb feat(phase1): xhbm_emulator.latency_model - L2 fidelity
    597d08b feat(phase1): xhbm_emulator.profiles - JSON loader + dataclasses
    cb1192a feat(phase1): SSD profiles + measurement-based threshold config

Total: 9 commits, 1 fixup, 2 tags. ~1,373 LOC + JSON.
