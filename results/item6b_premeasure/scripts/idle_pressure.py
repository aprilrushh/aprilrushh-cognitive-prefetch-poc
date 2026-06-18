import urllib.request, json, time, statistics, concurrent.futures
from transformers import AutoTokenizer
BASE="http://localhost:8000/v1/completions"; M="meta-llama/Llama-3.1-70B-Instruct"
N=16; OUT=200; TGT=12000   # 16 x 12K = 192K >> GPU 94.7K
tok=AutoTokenizer.from_pretrained(M)

def pad_to(text, n):  # 토크나이저로 정확히 n토큰 맞춤
    ids=tok(text).input_ids
    if len(ids)>=n: return tok.decode(ids[:n])
    unit=tok(" detail").input_ids[-1]
    return tok.decode(ids + [unit]*(n-len(ids)))

shared_raw="Shared corporate knowledge base. "+("Compliance clause: verify identity before any disclosure. "*600)
SHARED=pad_to(shared_raw, 6000)   # 정확히 6K 공유
print(f"[prep] shared prefix = {len(tok(SHARED).input_ids)} tok", flush=True)
def make(sid):
    uniq=f" Session {sid} confidential. "+(f"Private record {sid} item. "*600)
    full=SHARED + pad_to(uniq, 6000) + f"\nQ: Summarize session {sid}.\nA:"
    return full
PROMPTS=[make(s) for s in range(N)]
print(f"[prep] per-session tokens ~= {len(tok(PROMPTS[0]).input_ids)} | total ~= {N*len(tok(PROMPTS[0]).input_ids)/1000:.0f}K vs GPU 94.7K", flush=True)

def call(sid):
    body=json.dumps({"model":M,"prompt":PROMPTS[sid],"max_tokens":OUT,"temperature":0,"ignore_eos":True}).encode()
    r=urllib.request.Request(BASE,data=body,headers={"Content-Type":"application/json"})
    t=time.time()
    with urllib.request.urlopen(r,timeout=1200) as resp: d=json.loads(resp.read())
    return sid, time.time()-t

print(f"\n=== PASS A (cold): {N} sessions x 12K 동시 ===", flush=True)
t0=time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=N) as ex:
    A=[f.result() for f in concurrent.futures.as_completed([ex.submit(call,s) for s in range(N)])]
dtA=[r[1] for r in A]; wA=time.time()-t0
print(f"[A] wall={wA:.1f}s  per-req: avg={statistics.mean(dtA):.1f} med={statistics.median(dtA):.1f} min={min(dtA):.1f} max={max(dtA):.1f}", flush=True)
time.sleep(3)
print(f"=== PASS B (warm): 동일 재방문 ===", flush=True)
t0=time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=N) as ex:
    B=[f.result() for f in concurrent.futures.as_completed([ex.submit(call,s) for s in range(N)])]
dtB=[r[1] for r in B]; wB=time.time()-t0
print(f"[B] wall={wB:.1f}s  per-req: avg={statistics.mean(dtB):.1f} med={statistics.median(dtB):.1f} max={max(dtB):.1f}", flush=True)
print(f"\nwarm/cold wall ratio={wB/wA:.2f}", flush=True)
