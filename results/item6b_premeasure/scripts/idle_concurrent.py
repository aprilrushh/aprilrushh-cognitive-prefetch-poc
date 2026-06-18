import urllib.request, json, time, statistics, concurrent.futures
BASE="http://localhost:8000/v1/completions"; M="meta-llama/Llama-3.1-70B-Instruct"
N=14; OUT=256
def call(sid, prompt):
    body=json.dumps({"model":M,"prompt":prompt,"max_tokens":OUT,"temperature":0,"ignore_eos":True}).encode()
    r=urllib.request.Request(BASE,data=body,headers={"Content-Type":"application/json"})
    t=time.time()
    with urllib.request.urlopen(r,timeout=900) as resp: d=json.loads(resp.read())
    return sid, d["choices"][0]["text"], time.time()-t

# 공유 프리픽스 절반 + 세션 고유 절반 (재방문 시 prefix 재사용 여지)
shared="Company knowledge base. "+("Policy clause: all agents must verify identity before disclosure. "*250)  # ~4K 공유
def make(sid, suffix):
    uniq=f" Session {sid} private notes: detail {sid} repeated. "*250  # ~4K 고유 → 세션당 ~8K
    return shared+uniq+suffix

print(f"=== concurrent idle-session: N={N}, out={OUT}, 세션당 ~8K (합 {N*8}K vs GPU 94.7K) ===", flush=True)
# Pass A (cold): 14세션 동시
t0=time.time(); 
with concurrent.futures.ThreadPoolExecutor(max_workers=N) as ex:
    futs=[ex.submit(call, s, make(s, f"\nQ1: summarize.\nA1:")) for s in range(N)]
    resA=[f.result() for f in concurrent.futures.as_completed(futs)]
dtA=[r[2] for r in resA]; wallA=time.time()-t0
print(f"[PASS A cold] wall={wallA:.1f}s  TTFT/req: avg={statistics.mean(dtA):.2f} med={statistics.median(dtA):.2f} max={max(dtA):.2f}", flush=True)
time.sleep(3)
# Pass B (warm): 같은 14세션 재방문 (동일 프롬프트 → prefix 재사용 가능하면 빨라야)
t0=time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=N) as ex:
    futs=[ex.submit(call, s, make(s, f"\nQ1: summarize.\nA1:")) for s in range(N)]
    resB=[f.result() for f in concurrent.futures.as_completed(futs)]
dtB=[r[2] for r in resB]; wallB=time.time()-t0
print(f"[PASS B warm] wall={wallB:.1f}s  TTFT/req: avg={statistics.mean(dtB):.2f} med={statistics.median(dtB):.2f} max={max(dtB):.2f}", flush=True)
print(f"\nwarm/cold wall ratio = {wallB/wallA:.2f}  (커넥터가 KV 보관하면 warm이 빨라짐; 바닐라는 재계산이라 비슷)")
