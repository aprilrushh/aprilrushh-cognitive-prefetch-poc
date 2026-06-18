import urllib.request, json, time, statistics
BASE="http://localhost:8000/v1/completions"; M="meta-llama/Llama-3.1-70B-Instruct"
N_SESS=14; CTX_REPEAT=520; ROUNDS=3   # 세션당 ~8K, N*8K=112K > GPU 94.7K
def call(prompt, mx=24):
    body=json.dumps({"model":M,"prompt":prompt,"max_tokens":mx,"temperature":0}).encode()
    r=urllib.request.Request(BASE,data=body,headers={"Content-Type":"application/json"})
    t=time.time()
    with urllib.request.urlopen(r,timeout=600) as resp: d=json.loads(resp.read())
    return d["choices"][0]["text"], time.time()-t

# 각 세션 고유 컨텍스트 (서로 다른 내용 → 세션 구분)
sessions=[]
for s in range(N_SESS):
    ctx=f"Session {s} dossier. "+(f"Case file {s}: the investigator noted detail number {s} repeatedly. "*CTX_REPEAT)
    sessions.append(ctx+f"\n\nQ1: Summarize session {s}.\nA1:")

print(f"=== idle-session bench: N={N_SESS} sessions, {ROUNDS} rounds round-robin ===", flush=True)
# Round 0: 모든 세션 1턴씩 (cold prefill, 누적 시작)
print("[round 0] prime all sessions (cold)...", flush=True)
t0=time.time()
for s in range(N_SESS):
    o,dt=call(sessions[s]); sessions[s]+=o+f"\n\nQ: Continue.\nA:"
print(f"  primed {N_SESS} sessions in {time.time()-t0:.1f}s", flush=True)

# Round 1..ROUNDS: 라운드로빈 재방문 → 재활성 TTFT 측정
reactivation=[]
for rnd in range(1, ROUNDS+1):
    lat=[]
    for s in range(N_SESS):
        o,dt=call(sessions[s]); sessions[s]+=o+f"\n\nQ: Continue.\nA:"
        lat.append(dt); reactivation.append(dt)
    print(f"[round {rnd}] reactivation dt: avg={statistics.mean(lat):.2f}s med={statistics.median(lat):.2f}s min={min(lat):.2f} max={max(lat):.2f}", flush=True)

print(f"\n=== REACTIVATION TTFT (all rounds, n={len(reactivation)}) ===")
print(f"avg={statistics.mean(reactivation):.2f}s  median={statistics.median(reactivation):.2f}s  p90={sorted(reactivation)[int(len(reactivation)*0.9)]:.2f}s  max={max(reactivation):.2f}s")
