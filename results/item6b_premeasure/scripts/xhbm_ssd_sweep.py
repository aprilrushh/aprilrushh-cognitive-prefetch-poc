import torch, gc, time, os, statistics
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL="meta-llama/Llama-3.1-70B-Instruct"; CTX=8000; NDEC=24
TIER_DIR="/home/ubuntu/xhbm_ssd_tier"; os.makedirs(TIER_DIR,exist_ok=True)
SWEEP=[24,32,48,64]
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
tok=AutoTokenizer.from_pretrained(MODEL)
print("[load] 70B NF4 ...", flush=True)
model=AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="cuda", dtype=torch.bfloat16)
nl=model.config.num_hidden_layers
import psutil; proc=psutil.Process()
g=lambda: torch.cuda.memory_allocated()/1e9; r=lambda: proc.memory_info().rss/1e9
print(f"[load] done. layers={nl}, GPU {g():.1f}GB", flush=True)

def make_ctx(sid,n=CTX):
    base=f"Session {sid} record. "+("The agent logged each event in sequence and verified it. "*1200)
    ids=tok(base).input_ids; unit=tok(" detail").input_ids[-1]
    if len(ids)<n: ids=ids+[unit]*(n-len(ids))
    ids=ids[:n]; assert len(ids)==n
    return torch.tensor([ids],device="cuda")

def greedy(Klist,Vlist,seed,n):
    c=DynamicCache()
    for L in range(nl): c.update(Klist[L],Vlist[L],L)
    cur=seed.clone(); out=[]
    with torch.no_grad():
        for _ in range(n):
            o=model(cur,past_key_values=c,use_cache=True)
            cur=o.logits[:,-1:].argmax(-1); c=o.past_key_values; out.append(int(cur.item()))
    return out

def run(N):
    meta=[]; t0=time.time(); total_wb=0
    for s in range(N):
        ids=make_ctx(s)
        with torch.no_grad(): pf=model(ids,use_cache=True)
        pc=pf.past_key_values
        Kg=[pc.layers[L].keys for L in range(nl)]; Vg=[pc.layers[L].values for L in range(nl)]
        seed=ids[:,-1:].clone()
        base=greedy([k.clone() for k in Kg],[v.clone() for v in Vg],seed,NDEC)
        path=f"{TIER_DIR}/s{s}.pt"
        torch.save({"K":[k.cpu() for k in Kg],"V":[v.cpu() for v in Vg]}, path)
        total_wb+=os.path.getsize(path)
        del pf,pc,Kg,Vg; torch.cuda.empty_cache(); gc.collect()
        meta.append((seed,base))
    hold_gpu=g(); hold_ram=r(); demote_s=time.time()-t0
    # 라운드로빈 promote + correctness
    res=[]; lat=[]
    for s in range(N):
        obj=torch.load(f"{TIER_DIR}/s{s}.pt")
        Kg=[k.to("cuda",non_blocking=True) for k in obj["K"]]; Vg=[v.to("cuda",non_blocking=True) for v in obj["V"]]
        torch.cuda.synchronize()
        t=time.time(); out=greedy(Kg,Vg,meta[s][0],NDEC); lat.append(time.time()-t)
        res.append(sum(1 for a,b in zip(meta[s][1],out) if a==b))
        del Kg,Vg; torch.cuda.empty_cache()
    for s in range(N):
        p=f"{TIER_DIR}/s{s}.pt"; os.path.exists(p) and os.remove(p)
    perfect=sum(1 for m in res if m==NDEC)
    return dict(N=N, perfect=perfect, tok_ok=sum(res), tok_tot=N*NDEC,
                disk_gb=total_wb/1e9, hold_gpu=hold_gpu, hold_ram=hold_ram,
                demote_s=demote_s, lat=statistics.mean(lat))

print(f"\n{'N':>4} {'correct':>9} {'tok%':>6} {'disk_GB':>8} {'GPU':>6} {'RAM':>6} {'promote_s':>9} {'vs14':>6} {'verdict':>8}")
rows=[]
for N in SWEEP:
    try:
        d=run(N)
        v="PASS" if d['perfect']==N else f"{N-d['perfect']}fail"
        print(f"{d['N']:>4} {d['perfect']:>3}/{N:<5} {100*d['tok_ok']/d['tok_tot']:>5.1f} {d['disk_gb']:>8.1f} {d['hold_gpu']:>5.1f}G {d['hold_ram']:>5.0f}G {d['lat']:>9.2f} {N/14:>5.2f}x {v:>8}", flush=True)
        rows.append(d)
    except torch.cuda.OutOfMemoryError:
        print(f"{N:>4}  OOM — GPU 한계 도달 (이전 N이 최대 보유)", flush=True); torch.cuda.empty_cache(); break
    except Exception as e:
        print(f"{N:>4}  ERROR: {type(e).__name__}: {str(e)[:80]}", flush=True); torch.cuda.empty_cache(); break

print("\n=== SUMMARY ===")
if rows:
    best=rows[-1]
    print(f"최대 보유: {best['N']}세션 (vLLM ~14 대비 {best['N']/14:.2f}×), correctness {best['perfect']}/{best['N']}")
    print(f"전 구간 correctness: {'전부 100%' if all(d['perfect']==d['N'] for d in rows) else '일부 불일치(위 표)'}")
