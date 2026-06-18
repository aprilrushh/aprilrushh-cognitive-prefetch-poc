import torch, time, numpy as np
import src.kv_codec as codec
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL="meta-llama/Llama-3.1-70B-Instruct"; CTX=32000
print(f"[load] {MODEL} NF4 ...", flush=True)
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="cuda", dtype=torch.bfloat16)
print(f"[load] done. GPU {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

base="The quick brown fox jumps over the lazy dog. Sherlock Holmes observed the room carefully and deduced the truth. "
ids=tok(base*2000, return_tensors="pt").input_ids[:,:CTX].to("cuda")
print(f"[ctx] tokens={ids.shape[1]}", flush=True)
with torch.no_grad():
    out=model(ids, use_cache=True)
kv=out.past_key_values
L=len(kv.layers) if hasattr(kv,'layers') else len(kv)
print(f"[kv] layers={L}", flush=True)

raw_bytes=0; enc_bytes=0; cos_list=[]; t0=time.time()
for i in range(L):
    if hasattr(kv,'layers'): K=kv.layers[i].keys; V=kv.layers[i].values
    else: K,V=kv[i]
    K=K.squeeze(0).contiguous().cpu(); V=V.squeeze(0).contiguous().cpu()
    K_out, V_enc = codec.encode_v_only_kv(K, V)
    info = codec.kv_round_trip_size_bytes(K_out, V_enc)
    raw_bytes += info["total_original"]      # K bf16 + V bf16
    enc_bytes += info["total_encoded"]       # K bf16 + V q4_0
    if i in (0, L//2, L-1):
        K_dec, V_dec = codec.decode_v_only_kv(K_out, V_enc)
        a=V.float().flatten().numpy(); b=V_dec.float().flatten().numpy()
        cs=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
        cos_list.append((i,min(1.0,cs)))
dt=time.time()-t0; tokens=ids.shape[1]; scale=32000.0/tokens
print("\n=========== V-ONLY COMPRESSION (REAL, 70B 32K) ===========")
print(f"measured tokens:   {tokens}  ({L} layers)")
print(f"raw KV (bf16):     {raw_bytes/1e9:.3f} GB   (32K 환산 {raw_bytes/1e9*scale:.2f} GB; anchor 10.49)")
print(f"V-only q4_0:       {enc_bytes/1e9:.3f} GB   (32K 환산 {enc_bytes/1e9*scale:.2f} GB; anchor calc 6.72)")
print(f"reduction:         {100*(raw_bytes-enc_bytes)/raw_bytes:.2f}%   (anchor calc 35.9%)")
print(f"V-recon cos_sim:   " + ", ".join(f"L{i}={c:.6f}" for i,c in cos_list))
print(f"encode time:       {dt:.1f}s")
