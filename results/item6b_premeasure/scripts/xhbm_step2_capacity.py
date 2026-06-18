import torch, gc, time, statistics
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL="meta-llama/Llama-3.1-70B-Instruct"; CTX=8000; NDEC=24
N_SESS=16  # RAM 171GB / 10.49GB ≈ 16 세션
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
tok=AutoTokenizer.from_pretrained(MODEL)
print("[load] 70B NF4...", flush=True)
model=AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="cuda", dtype=torch.bfloat16)
nl=model.config.num_hidden_layers
import psutil; proc=psutil.Process()
def gpu(): return torch.cuda.memory_allocated()/1e9
def ram(): return proc.memory_info().rss/1e9
print(f"[load] done. layers={nl}, GPU {gpu():.1f}GB", flush=True)

# 세션당 정확히 8K 토큰 (토크나이저 실측, §18 교훈)
def make_ctx(sid, n=8000):
    base=f"Session {sid} record. "+("The agent logged each event in sequence and verified it. "*400)
    ids=tok(base).input_ids
    if len(ids)<n:
        unit=tok(" more").input_ids[-1]; ids=ids+[unit]*(n-len(ids))
    return torch.tensor([ids[:n]], device="cuda")

print(f"\n=== N={N_SESS} 세션 prefill → RAM demote (활성 외 전부 RAM) ===", flush=True)
sessions=[]  # each: dict(K_ram, V_ram, seed, baseline_tokens)
def greedy(Klist, Vlist, seed, n, on_gpu_lists=False):
    c=DynamicCache()
    for L in range(nl):
        k=Klist[L] if on_gpu_lists else Klist[L].to("cuda",non_blocking=True)
        v=Vlist[L] if on_gpu_lists else Vlist[L].to("cuda",non_blocking=True)
        c.update(k,v,L)
    torch.cuda.synchronize()
    cur=seed.clone(); out=[]
    with torch.no_grad():
        for _ in range(n):
            o=model(cur,past_key_values=c,use_cache=True)
            cur=o.logits[:,-1:].argmax(-1); c=o.past_key_values; out.append(int(cur.item()))
    return out

t0=time.time()
for s in range(N_SESS):
    ids=make_ctx(s)
    with torch.no_grad(): pf=model(ids,use_cache=True)
    pc=pf.past_key_values
    # baseline: GPU에서 바로 디코드 (demote 전 정답)
    Kg=[pc.layers[L].keys.clone() for L in range(nl)]
    Vg=[pc.layers[L].values.clone() for L in range(nl)]
    seed=ids[:,-1:].clone()
    base=greedy(Kg,Vg,seed,NDEC,on_gpu_lists=True)
    # demote → RAM
    Kr=[k.to("cpu").pin_memory() for k in Kg]; Vr=[v.to("cpu").pin_memory() for v in Vg]
    del Kg,Vg,pf,pc; torch.cuda.empty_cache(); gc.collect()
    sessions.append(dict(Kr=Kr,Vr=Vr,seed=seed,base=base))
    print(f"  session {s+1}/{N_SESS}: prefill+demote. GPU {gpu():.1f}GB | RAM {ram():.1f}GB", flush=True)
print(f"[hold] {N_SESS} 세션 동시 보유. GPU {gpu():.1f}GB (활성 KV 없음=가중치만) | RAM {ram():.1f}GB | {time.time()-t0:.0f}s", flush=True)
print(f"  → 순수 vLLM은 동일 GPU에서 ~14세션 큐잉 시작(§18). 여기선 {N_SESS}세션 보유 중.", flush=True)

# 라운드로빈: 각 세션 promote→디코드→correctness, 다시 demote
print(f"\n=== 라운드로빈 활성화 + 전수 correctness ===", flush=True)
results=[]; lat=[]
for s in range(N_SESS):
    t=time.time()
    out=greedy(sessions[s]['Kr'], sessions[s]['Vr'], sessions[s]['seed'], NDEC)
    dt=time.time()-t; lat.append(dt)
    m=sum(1 for a,b in zip(sessions[s]['base'],out) if a==b)
    results.append(m); torch.cuda.empty_cache()
    print(f"  session {s}: promote+decode {dt:.2f}s, agreement {m}/{NDEC}", flush=True)

perfect=sum(1 for m in results if m==NDEC)
print("\n================ STEP 2 RESULT ================")
print(f"동시 보유 세션: {N_SESS}  (vLLM 큐잉 한계 ~14 대비 {N_SESS/14:.2f}×)")
print(f"correctness: {perfect}/{N_SESS} 세션이 100% 일치  (전체 토큰 {sum(results)}/{N_SESS*NDEC})")
print(f"promote+decode latency: avg={statistics.mean(lat):.2f}s med={statistics.median(lat):.2f}s")
print("VERDICT:", "✅ PASS — N세션 보유 + 무손상 복원" if perfect==N_SESS else f"⚠️ {N_SESS-perfect}세션 불일치 — 원인규명")
