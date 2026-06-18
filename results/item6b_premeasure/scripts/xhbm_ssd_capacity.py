import torch, gc, time, os, sys, statistics
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache
import src.kv_codec as codec

MODEL="meta-llama/Llama-3.1-70B-Instruct"; CTX=8000; NDEC=24
N_SESS=int(os.environ.get("N_SESS","16"))
MODE=os.environ.get("MODE","raw")   # raw | vonly
TIER_DIR="/home/ubuntu/xhbm_ssd_tier"
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
tok=AutoTokenizer.from_pretrained(MODEL)
print(f"[load] 70B NF4 ... (N={N_SESS}, MODE={MODE})", flush=True)
model=AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="cuda", dtype=torch.bfloat16)
nl=model.config.num_hidden_layers
import psutil; proc=psutil.Process()
g=lambda: torch.cuda.memory_allocated()/1e9; r=lambda: proc.memory_info().rss/1e9
print(f"[load] done. layers={nl}, GPU {g():.1f}GB", flush=True)

def make_ctx(sid,n=CTX):
    base=f"Session {sid} record. "+("The agent logged each event in sequence and verified it. "*1200)
    ids=tok(base).input_ids
    unit=tok(" detail").input_ids[-1]
    if len(ids)<n: ids=ids+[unit]*(n-len(ids))
    ids=ids[:n]
    assert len(ids)==n, f"ctx {len(ids)} != {n}"
    return torch.tensor([ids],device="cuda")

def greedy_from_gpu(Klist,Vlist,seed,n):
    c=DynamicCache()
    for L in range(nl): c.update(Klist[L],Vlist[L],L)
    cur=seed.clone(); out=[]
    with torch.no_grad():
        for _ in range(n):
            o=model(cur,past_key_values=c,use_cache=True)
            cur=o.logits[:,-1:].argmax(-1); c=o.past_key_values; out.append(int(cur.item()))
    return out

# demote: KV를 디스크로 저장 (raw=bf16 .pt / vonly=q4_0 압축)
def demote(sid,Kg,Vg):
    path=f"{TIER_DIR}/sess_{sid}_{MODE}.pt"; wb=0
    if MODE=="raw":
        obj={"K":[k.cpu() for k in Kg],"V":[v.cpu() for v in Vg]}
        torch.save(obj,path)
    else:  # vonly: K bf16, V q4_0
        Kc=[k.cpu() for k in Kg]; Venc=[]
        for L in range(nl):
            _,ve=codec.encode_v_only_kv(Kg[L].squeeze(0).cpu().contiguous(), Vg[L].squeeze(0).cpu().contiguous())
            Venc.append(ve)
        obj={"K":Kc,"Venc":Venc,"shape":[v.shape for v in Vg]}
        torch.save(obj,path)
    wb=os.path.getsize(path)
    return path,wb

def promote_and_decode(sid,seed,base):
    path=f"{TIER_DIR}/sess_{sid}_{MODE}.pt"
    obj=torch.load(path)
    if MODE=="raw":
        Kg=[k.to("cuda",non_blocking=True) for k in obj["K"]]
        Vg=[v.to("cuda",non_blocking=True) for v in obj["V"]]
    else:
        Kg=[k.to("cuda",non_blocking=True) for k in obj["K"]]
        Vg=[]
        for L in range(nl):
            vdec=codec.decode_q4_0(obj["Venc"][L]).to("cuda").reshape(obj["shape"][L])
            Vg.append(vdec)
    torch.cuda.synchronize()
    out=greedy_from_gpu(Kg,Vg,seed,NDEC)
    m=sum(1 for a,b in zip(base,out) if a==b)
    return m

print(f"\n=== N={N_SESS} prefill + demote to SSD ({MODE}) ===",flush=True)
meta=[]; t0=time.time(); total_wb=0
for s in range(N_SESS):
    ids=make_ctx(s)
    with torch.no_grad(): pf=model(ids,use_cache=True)
    pc=pf.past_key_values
    Kg=[pc.layers[L].keys for L in range(nl)]; Vg=[pc.layers[L].values for L in range(nl)]
    seed=ids[:,-1:].clone()
    base=greedy_from_gpu([k.clone() for k in Kg],[v.clone() for v in Vg],seed,NDEC)
    path,wb=demote(s,Kg,Vg); total_wb+=wb
    del pf,pc,Kg,Vg; torch.cuda.empty_cache(); gc.collect()
    meta.append(dict(seed=seed,base=base))
    if (s+1)%4==0 or s==N_SESS-1:
        print(f"  {s+1}/{N_SESS} demoted. GPU {g():.1f} RAM {r():.0f} | disk/sess {wb/1e9:.2f}GB",flush=True)
print(f"[hold] {N_SESS} 세션 SSD 보유. 총 디스크 {total_wb/1e9:.1f}GB | {time.time()-t0:.0f}s",flush=True)

print(f"\n=== 라운드로빈 promote(SSD→GPU) + 전수 correctness ===",flush=True)
res=[]; lat=[]
for s in range(N_SESS):
    t=time.time(); m=promote_and_decode(s,meta[s]['seed'],meta[s]['base']); lat.append(time.time()-t)
    res.append(m); torch.cuda.empty_cache()
perfect=sum(1 for m in res if m==NDEC)
print(f"\n========= N={N_SESS} MODE={MODE} RESULT =========")
print(f"동시 보유: {N_SESS} (vLLM ~14 대비 {N_SESS/14:.2f}×)")
print(f"correctness: {perfect}/{N_SESS} 세션 100% | 토큰 {sum(res)}/{N_SESS*NDEC} = {100*sum(res)/(N_SESS*NDEC):.1f}%")
print(f"총 SSD 쓰기: {total_wb/1e9:.1f}GB (세션당 {total_wb/N_SESS/1e9:.2f}GB)")
print(f"promote+decode: avg={statistics.mean(lat):.2f}s")
print("VERDICT:", "✅ PASS" if perfect==N_SESS else f"⚠️ {N_SESS-perfect}세션 불일치({MODE})")
# 정리
for s in range(N_SESS):
    p=f"{TIER_DIR}/sess_{s}_{MODE}.pt"; os.path.exists(p) and os.remove(p)
