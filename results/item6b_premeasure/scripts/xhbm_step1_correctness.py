import torch, gc
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL="meta-llama/Llama-3.1-70B-Instruct"; CTX=8000; NDEC=40
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
tok=AutoTokenizer.from_pretrained(MODEL)
print("[load] 70B NF4...", flush=True)
model=AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="cuda", dtype=torch.bfloat16)
nl=model.config.num_hidden_layers
print(f"[load] done. layers={nl}", flush=True)

# 컨텍스트 prefill
text=("The investigator examined the evidence and recorded each detail carefully. "*200)
ids=tok(text, return_tensors="pt").input_ids[:,:CTX].to("cuda")
print(f"[ctx] {ids.shape[1]} tokens", flush=True)

def greedy_decode(make_cache_fn, n):
    """make_cache_fn() -> (cache, first_input_id). greedy n토큰 생성, 토큰 리스트 반환."""
    cache, cur = make_cache_fn()
    out_ids=[]
    with torch.no_grad():
        for _ in range(n):
            o=model(cur, past_key_values=cache, use_cache=True)
            cur=o.logits[:,-1:].argmax(-1); cache=o.past_key_values
            out_ids.append(int(cur.item()))
    return out_ids

# prefill 한 번 → K/V 템플릿 추출 (m1_4 패턴)
with torch.no_grad():
    pf=model(ids, use_cache=True)
pc=pf.past_key_values
K_gpu=[pc.layers[L].keys.clone() for L in range(nl)]
V_gpu=[pc.layers[L].values.clone() for L in range(nl)]
seed=ids[:,-1:].clone()
del pf, pc; torch.cuda.empty_cache()

# --- Baseline: demote 없이 GPU 그대로 디코드 ---
def make_baseline():
    c=DynamicCache()
    for L in range(nl): c.update(K_gpu[L].clone(), V_gpu[L].clone(), L)
    return c, seed.clone()
base_tokens=greedy_decode(make_baseline, NDEC)
print(f"[baseline] {NDEC} tokens 생성 완료", flush=True)

# --- XHBM 경로: KV를 RAM으로 demote → GPU 해제 → promote → 디코드 ---
# demote: GPU→CPU(pinned)
K_ram=[k.to("cpu", non_blocking=False).pin_memory() for k in K_gpu]
V_ram=[v.to("cpu", non_blocking=False).pin_memory() for v in V_gpu]
del K_gpu, V_gpu; torch.cuda.empty_cache(); gc.collect()
print(f"[demote] KV → RAM, GPU KV 해제. GPU now {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
def make_promoted():
    c=DynamicCache()
    for L in range(nl):
        kp=K_ram[L].to("cuda", non_blocking=True); vp=V_ram[L].to("cuda", non_blocking=True)
        c.update(kp, vp, L)
    torch.cuda.synchronize()
    return c, seed.clone()
xhbm_tokens=greedy_decode(make_promoted, NDEC)
print(f"[promote+decode] {NDEC} tokens 생성 완료", flush=True)

# --- CORRECTNESS GATE ---
match=sum(1 for a,b in zip(base_tokens,xhbm_tokens) if a==b)
print("\n================ CORRECTNESS GATE ================")
print(f"baseline first10: {base_tokens[:10]}")
print(f"xhbm     first10: {xhbm_tokens[:10]}")
print(f"token agreement: {match}/{NDEC} = {100*match/NDEC:.1f}%")
print("VERDICT:", "✅ PASS (demote/promote 출력 무손상)" if match==NDEC else f"❌ FAIL ({NDEC-match} 토큰 불일치 — 원인규명 필요)")
