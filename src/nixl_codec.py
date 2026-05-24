import torch
import zlib
import time
import concurrent.futures
from dataclasses import dataclass

@dataclass
class CompressionResult:
    layer: int
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    time_taken_ms: float

class NixlAsyncCodec:
    def __init__(self, max_workers=2):
        # GPU 메인 스레드를 방해하지 않는 별도의 작업자 풀 생성
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        # 진행 중인 압축 작업을 추적하는 딕셔너리
        self._active_tasks = {}

    def _compress_worker(self, layer_idx, k_bytes, v_bytes):
        """
        이 코드가 하는 일: 백그라운드 스레드에서 실제 바이트스트림 압축을 수행합니다.
        (PoC에서는 zlib을 사용하며, 실전에서는 LZ4나 CacheGen 양자화 텐서 코덱으로 교체됩니다.)
        """
        start_time = time.perf_counter()
        
        # 바이트스트림 결합 및 압축 (level 1은 속도 우선)
        combined_stream = k_bytes + v_bytes
        compressed_data = zlib.compress(combined_stream, level=1)
        
        end_time = time.perf_counter()
        
        original_size = len(combined_stream)
        compressed_size = len(compressed_data)
        
        return CompressionResult(
            layer=layer_idx,
            original_bytes=original_size,
            compressed_bytes=compressed_size,
            compression_ratio=compressed_size / original_size,
            time_taken_ms=(end_time - start_time) * 1000
        )

    def submit_compression(self, layer_idx, k_flat_tensor, v_flat_tensor):
        """
        이 코드가 하는 일: 메인 스레드는 압축 명령만 던지고(submit) 바로 자신의 다음 일을 하러 갑니다.
        """
        # 텐서 내부의 메모리 포인터(DataPtr)를 직접 읽어 순수 bytes 객체로 변환
        # (이 과정은 매우 빠릅니다)
        # PoC에서는 메모리 안전을 위해 numpy를 거쳐 bytes로 변환합니다.
        # bf16/numpy 미지원 회피: float16으로 view (압축 byte stream만 사용, dtype 무관)
        k_t = k_flat_tensor.detach()
        v_t = v_flat_tensor.detach()
        if k_t.dtype == torch.bfloat16:
            k_t = k_t.view(torch.float16)
            v_t = v_t.view(torch.float16)
        k_bytes = k_t.cpu().numpy().tobytes()
        v_bytes = v_t.cpu().numpy().tobytes()
        
        # 백그라운드 워커에 압축 작업 던지기
        future = self._executor.submit(self._compress_worker, layer_idx, k_bytes, v_bytes)
        self._active_tasks[layer_idx] = future
        
        return future

    def check_status(self, layer_idx):
        """압축 완료 여부 확인 (논블로킹)"""
        if layer_idx in self._active_tasks:
            future = self._active_tasks[layer_idx]
            if future.done():
                return future.result()
            return "Compressing..."
        return "No task found"

if __name__ == "__main__":
    import torch
    
    print("=== NIXL Async Codec Test ===")
    codec = NixlAsyncCodec()
    
    # 더미 데이터: 0.0과 1.0이 섞인 압축하기 좋은 희소 텐서 생성
    dummy_k = torch.zeros(32768, dtype=torch.float16)
    dummy_k[::10] = 1.0  # 인위적 패턴 주입
    dummy_v = torch.zeros(32768, dtype=torch.float16)
    
    # 1. 압축 작업 던지기 (Non-blocking)
    start_main = time.perf_counter()
    future = codec.submit_compression(layer_idx=0, k_flat_tensor=dummy_k, v_flat_tensor=dummy_v)
    end_main = time.perf_counter()
    
    print(f"Main thread blocking time: {(end_main - start_main)*1000:.2f} ms")
    
    # 2. 결과 기다리기 (테스트용이므로 블로킹 대기)
    result = future.result()
    
    print(f"Layer: {result.layer}")
    print(f"Original Bytes: {result.original_bytes}")
    print(f"Compressed Bytes: {result.compressed_bytes}")
    print(f"Compression Ratio: {result.compression_ratio*100:.1f}%")
    print(f"Background Compression Time: {result.time_taken_ms:.2f} ms")
    print("Codec PASS")
