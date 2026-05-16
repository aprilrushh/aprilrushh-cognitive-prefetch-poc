import re

# 기존 cognitive_cache_v2.py 원본 읽기
with open('src/cognitive_cache_v2.py', 'r') as f:
    content = f.read()

# 1. NixlSolidigmPlugin 임포트 추가
import_statement = "import torch\nimport sys, os\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\nfrom src.nixl_plugin_manager import NixlSolidigmPlugin\n"
content = content.replace("import torch\n", import_statement)

# 2. Plugin 초기화
init_hook = """
        self.predictor = CognitivePredictor(config)
        
        # [NIXL Plugin] 메인 통합 관제탑 탑재
        self.nixl_plugin = NixlSolidigmPlugin(num_workers=4)
"""
content = content.replace("self.predictor = CognitivePredictor(config)", init_hook)

# 3. update 메서드 (혹은 offload 발생 지점)에 통합 파이프라인 연동
update_hook = """
        # --- NIXL Solidigm Tunnel ---
        # 상위 엔진 모르게 가로채기 -> 직렬화 -> 64KB 셰이핑 -> 비동기 압축 워커 던지기
        if layer_idx is not None:
            self.nixl_plugin.process_layer_offload(
                layer_idx, 
                self.key_cache[layer_idx], 
                self.value_cache[layer_idx]
            )
        # -------------------------------
"""
content = re.sub(r'(return self\.key_cache\[layer_idx\], self\.value_cache\[layer_idx\])', r'%s\n        \1' % update_hook, content)

# 4. 소멸자 추가: 테스트 종료 시 비동기 압축이 끝날 때까지 대기 및 지표 출력
del_hook = """
    def __del__(self):
        try:
            m = self.nixl_plugin.get_metrics()
            if m['total_original_bytes'] > 0:
                print(f"\\n[NIXL Backend Metric] Intercepted {m['intercept_calls']} calls")
                print(f"[NIXL Backend Metric] Total Original: {m['total_original_bytes']} bytes")
                print(f"[NIXL Backend Metric] Total Compressed: {m['total_compressed_bytes']} bytes")
                print(f"[NIXL Backend Metric] PCIe BW Saved (Est): {(1 - m['total_compressed_bytes']/m['total_original_bytes'])*100:.1f}%")
        except:
            pass
"""
# 클래스 끝부분에 소멸자 삽입
content = content + "\n" + del_hook

with open('src/cognitive_cache_v2.py', 'w') as f:
    f.write(content)

print("Final patching complete. Running Hybrid mode E2E integration...")
