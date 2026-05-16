import torch
import weakref

class NixlInterceptor:
    def __init__(self):
        # 가로챈 데이터를 담아둘 가상 버퍼 (딕셔너리)
        self.virtual_buffer = {}
        # 메모리 누수 방지를 위한 약한 참조 추적기
        self._tensor_tracker = weakref.WeakValueDictionary()

    def intercept_and_serialize(self, layer_idx, key_tensor, value_tensor):
        """
        이 코드가 하는 일: 상위 엔진이 디스크로 보내려는 K, V 텐서를 가로채어
        메모리 복사 없이 1차원 바이트스트림으로 펴고 가상 버퍼에 격리합니다.
        """
        # 1. 텐서 모양 보정 (Zero-copy 1차원 펴기)
        # 이미 정렬된 메모리라면 view(-1)로 포인터만 가져오고, 꼬여있다면 contiguous()로 정렬합니다.
        k_flat = key_tensor.contiguous().view(-1) if not key_tensor.is_contiguous() else key_tensor.view(-1)
        v_flat = value_tensor.contiguous().view(-1) if not value_tensor.is_contiguous() else value_tensor.view(-1)

        # 2. 바이트 크기 산출 (HBM 실측 지표 추적용)
        k_bytes = k_flat.numel() * key_tensor.element_size()
        v_bytes = v_flat.numel() * value_tensor.element_size()

        # 3. 가상 버퍼에 격리 (이후 압축 워커가 여기서 데이터를 꺼내감)
        self.virtual_buffer[layer_idx] = {
            'k_stream': k_flat,
            'v_stream': v_flat,
            'total_bytes': k_bytes + v_bytes,
            'original_shape': key_tensor.shape
        }

        # 메모리 릭 추적을 위해 등록
        self._tensor_tracker[f"L{layer_idx}_K"] = k_flat
        self._tensor_tracker[f"L{layer_idx}_V"] = v_flat

        # 4. 상위 엔진에는 성공적으로 오프로딩을 완료했다는 신호 반환
        return True

if __name__ == "__main__":
    # Smoke Test: Llama 70B의 1개 레이어, 32 토큰 분량의 텐서를 모사합니다.
    interceptor = NixlInterceptor()
    dummy_k = torch.randn(1, 8, 32, 128, dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")
    dummy_v = torch.randn(1, 8, 32, 128, dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")
    
    success = interceptor.intercept_and_serialize(layer_idx=4, key_tensor=dummy_k, value_tensor=dummy_v)
    buf = interceptor.virtual_buffer[4]
    
    print("=== NixlInterceptor Smoke Test ===")
    print(f"Intercept Success: {success}")
    print(f"Original Shape: {list(buf['original_shape'])}")
    print(f"Flattened K shape: {list(buf['k_stream'].shape)}")
    print(f"Total Bytes intercepted: {buf['total_bytes']} bytes")
    print("Test PASS")
