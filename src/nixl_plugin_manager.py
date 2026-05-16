import time
import sys
import os

# 상위 디렉토리를 파이썬 경로에 추가하여 'src' 모듈을 정상적으로 찾게 만듭니다.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nixl_interceptor import NixlInterceptor
from src.nixl_core import NixlSerializationCore
from src.nixl_codec import NixlAsyncCodec

class NixlSolidigmPlugin:
    """
    이 코드가 하는 일: Llama 캐시 모듈이 복잡한 작업 없이
    오직 '처리해줘' 한마디만 하면 가로채기, 정렬, 압축, 
    솔리다임 64KB 매핑을 한 번에 비동기로 처리하는 통합 관제탑입니다.
    """
    def __init__(self, num_workers=4):
        self.interceptor = NixlInterceptor()
        self.core = NixlSerializationCore()
        self.codec = NixlAsyncCodec(max_workers=num_workers)
        self.metrics = {
            'total_original_bytes': 0,
            'total_compressed_bytes': 0,
            'intercept_calls': 0
        }

    def process_layer_offload(self, layer_idx, key_tensor, value_tensor):
        """
        메인 스레드(GPU)가 호출하는 유일한 메서드입니다.
        엄청나게 빨리 끝납니다 (Fire and Forget).
        """
        # 1. 가로채기 및 직렬화 (Zero-copy)
        self.interceptor.intercept_and_serialize(layer_idx, key_tensor, value_tensor)
        buf = self.interceptor.virtual_buffer[layer_idx]
        
        # 2. 솔리다임 64KB 셰이핑 분석 (메타데이터 로깅용)
        shape_info = self.core.shape_and_buffer(layer_idx, buf['k_stream'], buf['v_stream'])
        
        # 3. 비동기 압축 워커에 던지기 (Non-blocking)
        future = self.codec.submit_compression(layer_idx, buf['k_stream'], buf['v_stream'])
        
        # 4. 실측 메트릭 업데이트
        self.metrics['intercept_calls'] += 1
        self.metrics['total_original_bytes'] += shape_info['total_bytes']
        
        return future

    def get_metrics(self):
        return self.metrics

if __name__ == "__main__":
    import torch
    print("=== NIXL Plugin Manager E2E Test ===")
    plugin = NixlSolidigmPlugin()
    
    # 4개 레이어 동시 오프로딩 시뮬레이션
    futures = []
    for layer in range(4):
        k = torch.randn(1, 8, 128, 128, dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")
        v = torch.randn(1, 8, 128, 128, dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")
        
        start = time.perf_counter()
        f = plugin.process_layer_offload(layer, k, v)
        end = time.perf_counter()
        
        print(f"L{layer} Dispatch Time: {(end-start)*1000:.2f} ms")
        futures.append(f)
        
    print("\nWaiting for background compressions...")
    for f in futures:
        res = f.result()
        plugin.metrics['total_compressed_bytes'] += res.compressed_bytes
        print(f"L{res.layer} compressed to {res.compressed_bytes} bytes in {res.time_taken_ms:.2f} ms")
        
    m = plugin.get_metrics()
    print(f"\n[Final Metric] Original: {m['total_original_bytes']} bytes")
    print(f"[Final Metric] Compressed: {m['total_compressed_bytes']} bytes")
    print(f"[Final Metric] Estimated PCIe Save: {(1 - m['total_compressed_bytes']/m['total_original_bytes'])*100:.1f}%")
    print("Manager PASS")
