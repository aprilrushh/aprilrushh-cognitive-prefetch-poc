import time, statistics, gc
import torch, psutil
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from pathlib import Path
proc=psutil.Process()
def log(m):
    print(f"[{torch.cuda.memory_allocated()/1e9:5.1f}GB GPU | {proc.memory_info().rss/1e9:5.1f}GB RAM] {m}", flush=True)
MODEL="meta-llama/Llama-3.1-70B-Instruct"; CTX=32000; NDEC=32; NMEAS=5
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
tok=AutoTokenizer.from_pretrained(MODEL)
log("load 70B NF4..."); model=AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="cuda", dtype=torch.bfloat16); log("loaded")
nl=model.config.num_hidden_layers; nkv=model.config.num_key_value_heads; hd=model.config.hidden_size//model.config.num_attention_heads
text=Path("data/corpus/raw/A_Study_in_Scarlet.txt").read_text(errors="replace")
ids=tok(text, return_tensors="pt").input_ids
ids=ids.repeat(1,(CTX//ids.shape[1])+1)[:,:CTX].to("cuda")

def measure(n_idle):
    idle=[]
    for _ in range(n_idle):
        idle.append([(torch.empty((1,nkv,CTX,hd),dtype=torch.bfloat16,pin_memory=True),
                      torch.empty((1,nkv,CTX,hd),dtype=torch.bfloat16,pin_memory=True)) for _ in range(nl)])
    with torch.no_grad():
        out=model(ids, use_cache=True); kv=out.past_key_values
    lat=[]
    for r in range(2+NMEAS):
        nxt=ids[:,-1:].clone()
        torch.cuda.synchronize(); t=[]
        cache=kv
        for _ in range(NDEC):
            s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
            with torch.no_grad():
                o=model(nxt, past_key_values=cache, use_cache=True)
            cache=o.past_key_values; nxt=o.logits[:,-1:].argmax(-1)
            e.record(); torch.cuda.synchronize(); t.append(s.elapsed_time(e))
        if r>=2: lat.append(statistics.mean(t))
        kv=out.past_key_values  # reset to post-prefill each round
    del idle; gc.collect()
    return statistics.mean(lat), statistics.stdev(lat) if len(lat)>1 else 0, min(lat), max(lat)

print("\n===== IDLE SWEEP (active decode ms/tok vs idle conv count) =====")
print(f"{'n_idle':>6} {'RAM_GB':>7} {'avg_ms':>8} {'stdev':>7} {'min':>7} {'max':>7} {'vs_n0_%':>8}")
base=None
for n in (0,1,3,7):
    avg,sd,mn,mx=measure(n)
    if base is None: base=avg
    ram=proc.memory_info().rss/1e9
    print(f"{n:>6} {ram:>7.1f} {avg:>8.2f} {sd:>7.2f} {mn:>7.2f} {mx:>7.2f} {100*(avg-base)/base:>+8.2f}")
