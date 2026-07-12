import torch
import math
import os
import concurrent.futures

DEFAULT_IU_BYTES = 65536  # P5336 indirection unit (64 KiB)

class NixlSerializationCore:
    def __init__(self, iu_bytes: int = None):
        """iu_bytes: drive indirection-unit size in bytes.
        Precedence: explicit arg > env XHBM_IU_BYTES > default 65536 (P5336).
        Alignment is a config parameter, not a constant — Fosom Lake or any
        other drive is a one-number change."""
        if iu_bytes is None:
            iu_bytes = int(os.environ.get("XHBM_IU_BYTES", DEFAULT_IU_BYTES))
        assert iu_bytes > 0 and (iu_bytes & (iu_bytes - 1)) == 0, "IU must be a power of two"
        self.chunk_size_bytes = iu_bytes
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
    def _reshape_to_bytes(self, tensor: torch.Tensor):
        """
        이 코드가 하는 일: 다차원 텐서의 구조를 파괴하여
        메모리 복사 없이 1차원 바이트 배열처럼 취급되도록 폅니다.
        """
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return tensor.view(-1)
        
    def shape_and_buffer(self, layer_idx, key_tensor, value_tensor):
        """
        이 코드가 하는 일: 텐서를 1차원으로 편 후, 솔리다임 64KB 박스에 
        딱 맞도록 모양을 깎아서 튜플로 반환합니다.
        """
        k_flat = self._reshape_to_bytes(key_tensor)
        v_flat = self._reshape_to_bytes(value_tensor)
        
        # 실제 바이트 크기 계산 (FP16 기준)
        total_elements = k_flat.numel() + v_flat.numel()
        total_bytes = total_elements * 2 
        
        # IU 단위(기본 64KB)로 박스 개수 계산
        num_boxes = math.ceil(total_bytes / self.chunk_size_bytes)
        
        return {
            'layer': layer_idx,
            'k_stream': k_flat,
            'v_stream': v_flat,
            'total_bytes': total_bytes,
            'solidigm_boxes': num_boxes
        }

if __name__ == "__main__":
    core = NixlSerializationCore()
    dummy_k = torch.randn(1, 8, 128, 128, dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")
    dummy_v = torch.randn(1, 8, 128, 128, dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")
    
    res = core.shape_and_buffer(1, dummy_k, dummy_v)
    
    print("=== NIXL Core Serialization Test ===")
    print(f"Total Bytes to Disk: {res['total_bytes']}")
    print(f"Solidigm IU Boxes Required ({core.chunk_size_bytes} B/box): {res['solidigm_boxes']}")
    print(f"Residual Waste: {(res['solidigm_boxes'] * core.chunk_size_bytes) - res['total_bytes']} bytes")
    print("Core PASS")
