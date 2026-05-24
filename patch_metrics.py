import re

with open('src/cognitive_cache_v2.py', 'r') as f:
    text = f.read()

# CPU에서 슬라이싱된 Subset의 크기를 캐시 통계 모듈에 명시적으로 누적하는 로직 추가
target_hook = r'(K_subset_cpu = torch\.gather\(cpu_k, 2, idx_exp\))'
metrics_patch = """\\1
        # --- Metrics Reconnection ---
        # CPU에서 잘라낸 핵심 토큰의 개수를 통계에 누적하여 데모 스크린에 반영
        try:
            subset_tokens = idx_exp.shape[2]
            if not hasattr(self, 'sparse_used'):
                self.sparse_used = 0
                self.total_prefill = 0
            self.sparse_used += subset_tokens
            self.total_prefill += cpu_k.shape[2]
        except Exception:
            pass
        # ----------------------------"""

text = re.sub(target_hook, metrics_patch, text)

# 스크립트에서 get_stats 등을 호출할 때 이 변수들을 반환하도록 보정 (방어적 패치)
stat_hook = r'(def get_stats\(self\):)'
stat_patch = """\\1
        if hasattr(self, 'sparse_used') and self.total_prefill > 0:
            self.stats = getattr(self, 'stats', {})
            self.stats['union_size'] = self.sparse_used
            self.stats['prefill_size'] = self.total_prefill"""

text = re.sub(stat_hook, stat_patch, text)

with open('src/cognitive_cache_v2.py', 'w') as f:
    f.write(text)

print("Metrics reconnection patch applied. Running final validation...")
