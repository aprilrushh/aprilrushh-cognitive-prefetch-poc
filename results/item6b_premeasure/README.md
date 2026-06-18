# Item 6b 사전측정 — XHBM vs KVBM vs LMCache (2026-06-17~18, North Texas H100)

Notion 기록: 페이지 9 "Item 6b 사전측정" §11, §14–§21
환경: Lambda H100 80GB, Llama-3.1-70B-Instruct NF4(bitsandbytes), vLLM 0.19.1,
      venv-anchor(XHBM 측정: torch 2.6/transformers 5.8), venv-compare(vLLM 서버)
GPU KV 예산 = 94,688 토큰 (3자 공정 비교 기준)

## 제1원칙
모든 문제에는 해결책이 있다. 끝까지 대안을 동원해 최고 상태로 달성하되,
결과는 성공/부분달성/한계 무엇이든 있는 그대로 정직하게 기록한다.
부풀린 성공보다 정직한 진전이 PoC에서 무너지지 않는다.

## 핵심 결과 요약
| 항목 | 결과 | Notion |
|---|---|---|
| 고유 capacity (3자, all-active) | 모두 1.0× (GPU-bound, 동시성12 천장) | §14 |
| idle capacity (3자, 192K 과부하) | 모두 1.0× (큐잉 지배, preempt 0) | §18 |
| LMCache 디스크 쓰기 | write-through 무압축 (2pass 101GB, idle +31GB) | §14,§18 |
| KVBM 디스크 쓰기 | 0 (Host-only, freq≥2 필터) | §14,§18 |
| XHBM V-only 압축 (텐서) | 총 KV −35.9%, cos_sim 0.99+ | §15 |
| XHBM active-decode 무영향 | 유휴적재 ±0.25% / promote중 +0.45% | §16 |
| XHBM idle-demote capacity (구현+실측) | N=64 SSD보유, correctness 100% | §20 |
| XHBM V-only demote | SSD쓰기 −28.3%(파일기준), 무손상 N=64 | §21 |

## 정직한 공백 (PoC 또는 추가개발)
- 실제 동시 디코딩 처리량 배수: 미측정 (보유≠동시디코딩, vLLM과 축 다름)
- NAND WAF/endurance/GC tail: Lambda virtio라 측정불가 → Solidigm PoC 전용
- "4.57×"는 보유 기준 — "2× 달성"으로 단정 금지, "메커니즘 N=64까지 무손상 작동"으로 표현

## 스크립트 → 측정 매핑
| 스크립트 | 측정 | Notion |
|---|---|---|
| vonly_measure.py | XHBM V-only 압축률+cos_sim (70B 32K) | §15 |
| idle_sweep.py | 유휴 0→7 적재 시 active decode 무영향 | §16 |
| idle_pressure.py | 192K 과부하 idle capacity (바닐라/KVBM/LMCache 공용) | §18 |
| idle_concurrent.py, idle_session_bench.py | idle 측정 초기 시도(압박 실패→교훈) | §18.1 |
| xhbm_step1_correctness.py | 단일세션 demote→promote→정답확인 40/40 | §20.2 |
| xhbm_step2_capacity.py | 16세션 RAM 보유 + correctness | §20.2 |
| xhbm_ssd_sweep.py | SSD tier N=24~64 raw demote sweep | §20.3 |
| xhbm_vonly_sweep.py | SSD tier N=24~64 V-only demote sweep | §21 |
| kvbm_correctness.py | KVBM #5068 재현시도(미발현) | §14 |

## 재현 방법
cd ~/xhbm-persistent/work/cognitive-prefetch-poc
PYTHONPATH=. venv-anchor/bin/python results/item6b_premeasure/scripts/<script>.py
(SSD sweep은 SWEEP=24,32,48,64 환경변수, idle은 vLLM 서버 기동 후 실행)

## 한계 명시
모든 XHBM capacity 측정은 "보유 후 라운드로빈"이며 동시 디코딩이 아님.
SSD tier는 Lambda virtio(쓰기 1.6GB/s) — 기능·correctness·쓰기량만 유효,
실제 NAND 성능/마모는 Solidigm PoC에서 측정.
