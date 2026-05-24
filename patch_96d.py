import re

with open('src/cognitive_cache_v2.py', 'r') as f:
    text = f.read()

# 1. GPU로 전체를 보내던 로직을 CPU 로컬 변수 참조로 교체
text = re.sub(
    r'K_full\s*=\s*self\._cpu_K\[([^\]]+)\]\.to\(self\.device_target[^)]*\)\.to\(torch\.float16\)',
    r'cpu_k = self._cpu_K[\1]', text
)
text = re.sub(
    r'V_full\s*=\s*self\._cpu_V\[([^\]]+)\]\.to\(self\.device_target[^)]*\)\.to\(torch\.float16\)',
    r'cpu_v = self._cpu_V[\1]', text
)

# 2. 인덱스 텐서를 GPU가 아닌 CPU 메모리(cpu_k.device)로 매핑
text = text.replace(
    'expand(B, KVH, U, D).to(K_full.device)',
    'expand(B, KVH, U, D).to(cpu_k.device)'
)

# 3. CPU에서 Gather를 수행한 후, 잘려진 조각(subset)만 GPU로 전송
text = re.sub(
    r'K_subset\s*=\s*torch\.gather\(K_full,\s*2,\s*idx_exp\)',
    r'K_subset_cpu = torch.gather(cpu_k, 2, idx_exp)\n        K_subset = K_subset_cpu.to(self.device_target, non_blocking=True).to(torch.float16)', text
)
text = re.sub(
    r'V_subset\s*=\s*torch\.gather\(V_full,\s*2,\s*idx_exp\)',
    r'V_subset_cpu = torch.gather(cpu_v, 2, idx_exp)\n        V_subset = V_subset_cpu.to(self.device_target, non_blocking=True).to(torch.float16)', text
)

# 4. 바이트 측정용으로 남아있을 수 있는 K_full 참조 안전하게 교체
text = text.replace('K_full.numel()', 'cpu_k.numel()')
text = text.replace('V_full.numel()', 'cpu_v.numel()')

with open('src/cognitive_cache_v2.py', 'w') as f:
    f.write(text)

print("9-6d K_subset-only mode patch applied successfully.")
