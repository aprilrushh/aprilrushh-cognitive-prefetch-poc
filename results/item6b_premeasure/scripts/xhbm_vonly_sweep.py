import torch, gc, time, os, statistics
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache
import src.kv_codec as codec
torch.serialization.add_safe_globals([codec.Q4_0_Encoded])

MODEL="meta-llama/Llama-3.1-70B-Instruct"; CTX=8000; NDEC=24
TIER_DIR="/home/ubuntu/xhbm_ssd_tier"; os.makedirs(TIER_DIR,exist_ok=True)
SWEEP=[int(x) for x in os.environ.get("SWEEP","8").split(",")]
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
tok=AutoTokenizer.from_pretrained(MODEL)
print("[load] 70B NF4 ...", flush=True)
model=AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="cuda", dtype=torch.bfloat16)
nl=model.config.num_hidden_layers
import psutil; proc=psutil.Process()
g=lambda: torch.cuda.memory_allocated()/1e9; r=lambda: proc.memory_info().rss/1e9
print(f"[load] done. layers={nl}", flush=True)

def make_ctx(sid,n=CTX):
    base=f"Session {sid} record. "+("The agent logged each event in sequence and verified it. "*1200)
    ids=tok(base).input_ids; unit=tok(" detail").input_ids[-1]
    if len(ids)<n: ids=ids+[unit]*(n-len(ids))
    return torch.tensor([ids[:n]],device="cuda")

def greedy(Klist,Vlist,seed,n):
    c=DynamicCache()
    for L in range(nl): c.update(Klist[L],Vlist[L],L)
    cur=seed.clone(); out=[]
    with torch.no_grad():
        for _ in range(n):
            o=model(cur,past_key_values=c,use_cache=True)
            cur=o.logits[:,-1:].argmax(-1); c=o.past_key_values; out.append(int(cur.item()))
    return out

def demote_vonly(sid,Kg,Vg):
    K_cpu=[k.cpu() for k in Kg]; Venc=[]; shapes=[]
    for L in range(nl):
        Vsq=Vg[L].squeeze(0).cpu().contiguous()  # (kv, seq, dim)
        _,ve=codec.encode_v_only_kv(Kg[L].squeeze(0).cpu().contiguous(), Vsq)
        Venc.append(ve); shapes.append(tuple(Vg[L].shape))
    path=f"{TIER_DIR}/v{sid}.pt"; torch.save({"K":K_cpu,"Venc":Venc,"shapes":shapes},path)
    return os.path.getsize(path)

def promote_vonly(sid):
    obj=torch.load(f"{TIER_DIR}/v{sid}.pt", weights_only=False)
    Kg=[k.to("cuda",non_blocking=True) for k in obj["K"]]; Vg=[]
    for L in range(nl):
        vdec=codec.decode_q4_0(obj["Venc"][L])         # (kv*seq*dim,) or (kv,seq,dim)?
        vdec=vdec.reshape(obj["shapes"][L]).to("cuda")
        Vg.append(vdec)
    torch.cuda.synchronize(); return Kg,Vg

def run(N):
    meta=[]; wb=0; t0=time.time()
    for s in range(N):
        ids=make_ctx(s)
        with torch.no_grad(): pf=model(ids,use_cache=True)
        pc=pf.past_key_values
        Kg=[pc.layers[L].keys for L in range(nl)]; Vg=[pc.layers[L].values for L in range(nl)]
        seed=ids[:,-1:].clone()
        base=greedy([k.clone() for k in Kg],[v.clone() for v in Vg],seed,NDEC)
        wb+=demote_vonly(s,Kg,Vg)
        del pf,pc,Kg,Vg; torch.cuda.empty_cache(); gc.collect()
        meta.append((seed,base))
    res=[]; lat=[]; worst=99
    for s in range(N):
        Kg,Vg=promote_vonly(s)
        t=time.time(); out=greedy(Kg,Vg,meta[s][0],NDEC); lat.append(time.time()-t)
        m=sum(1 for a,b in zip(meta[s][1],out) if a==b); res.append(m); worst=min(worst,m)
        del Kg,Vg; torch.cuda.empty_cache()
    for s in range(N): p=f"{TIER_DIR}/v{s}.pt"; os.path.exists(p) and os.remove(p)
    perfect=sum(1 for m in res if m==NDEC)
    return dict(N=N,perfect=perfect,tok_ok=sum(res),tok_tot=N*NDEC,disk_gb=wb/1e9,worst=worst,lat=statistics.mean(lat))

print(f"\n{'N':>4} {'perfect':>9} {'tok%':>6} {'worst/sess':>10} {'disk_GB':>8} {'vs_raw':>7} {'promote_s':>9}")
RAW_GB={24:62.9,32:83.9,48:125.8,64:167.8}
for N in SWEEP:
    d=run(N)
    raw=RAW_GB.get(N, N*2.62); red=100*(raw-d['disk_gb'])/raw
    print(f"{d['N']:>4} {d['perfect']:>3}/{N:<5} {100*d['tok_ok']/d['tok_tot']:>5.1f} {d['worst']:>6}/{NDEC:<3} {d['disk_gb']:>8.1f} {red:>6.1f}% {d['lat']:>9.2f}", flush=True)
