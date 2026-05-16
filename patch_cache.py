import re

# 기존 cognitive_cache_v2.py 파일 읽기
with open('src/cognitive_cache_v2.py', 'r') as f:
    content = f.read()

# 1. NixlInterceptor 임포트 및 초기화 추가
import_statement = "import torch\nfrom src.nixl_interceptor import NixlInterceptor\n"
content = content.replace("import torch\n", import_statement)

init_hook = """
        self.predictor = CognitivePredictor(config)
        
        # [NIXL Plugin] 가로채기 파이프라인 초기화
        self.nixl_interceptor = NixlInterceptor()
        self.total_bytes_intercepted = 0
"""
content = content.replace("self.predictor = CognitivePredictor(config)", init_hook)

# 2. update 메서드 (혹은 offload 발생 지점)에 가로채기 로직 추가
# Llama 모델이 과거 KV를 캐시에 업데이트할 때 가로챕니다.
update_hook = """
        # --- NIXL Intercept Tollgate ---
        # 상위 엔진 모르게 텐서를 1차원으로 펴서 가상 버퍼에 담습니다.
        if layer_idx is not None:
            self.nixl_interceptor.intercept_and_serialize(
                layer_idx, 
                self.key_cache[layer_idx], 
                self.value_cache[layer_idx]
            )
            # 디버깅용: 가로챈 용량 누적
            buf = self.nixl_interceptor.virtual_buffer[layer_idx]
            self.total_bytes_intercepted += buf['total_bytes']
        # -------------------------------
"""
# 보통 update 메서드의 마지막 반환 직전에 넣습니다.
content = re.sub(r'(return self\.key_cache\[layer_idx\], self\.value_cache\[layer_idx\])', r'%s\n        \1' % update_hook, content)

# 수정된 내용 덮어쓰기
with open('src/cognitive_cache_v2.py', 'w') as f:
    f.write(content)

print("Patching complete. Verifying...")
