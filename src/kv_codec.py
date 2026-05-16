"""V-only Q4_0 KV cache codec.

Spec source: configs/hardware/q4_0_quant.json (Step 0 verified).
Reference impl: llama.cpp ggml-quants.c quantize_row_q4_0_ref.

Design contract:
  K = bf16 passthrough (no quant)
  V = q4_0 (block size 32, per-block fp16 scale, symmetric, signed-4-bit shifted to [0,15])

Storage layout per block of 32 V elements:
  bytes [0:2]   = fp16 scale d
  bytes [2:18]  = 16 bytes packed (low nibble for x[0..15], high nibble for x[16..31])

Compression: bf16 (16 bpw) -> q4_0 (4.5 bpw) = 3.56x smaller
Used by: M5 service emulator (Step 3+), virtual SSD storage (Step 2)

NOT used by: M1.x measurements (those are bf16-native, dual-evidence per v1.4 § E).
"""
from dataclasses import dataclass
import numpy as np
import torch

Q4_0_BLOCK_SIZE = 32
Q4_0_BYTES_PER_BLOCK = 18  # 2 byte fp16 scale + 16 byte packed 4-bit
Q4_0_BPW = 4.5


@dataclass
class Q4_0_Encoded:
    """One tensor encoded with q4_0. Self-contained (carries shape + dtype for decode)."""
    scales: np.ndarray       # fp16, shape (n_blocks,)
    quants: np.ndarray       # uint8, shape (n_blocks, 16) — each byte packs 2 nibbles
    original_shape: tuple
    original_dtype: torch.dtype

    @property
    def storage_bytes(self) -> int:
        return self.scales.nbytes + self.quants.nbytes

    @property
    def n_elements(self) -> int:
        n = 1
        for s in self.original_shape:
            n *= s
        return n

    @property
    def effective_bpw(self) -> float:
        return self.storage_bytes * 8 / self.n_elements


def encode_q4_0(t: torch.Tensor) -> Q4_0_Encoded:
    """Encode bf16/fp16/fp32 tensor to q4_0. CPU only.

    Last-axis-flatten, must be multiple of block size (32).
    Reference: llama.cpp ggml-quants.c quantize_row_q4_0_ref (line 1639).
    """
    assert t.is_cpu, "encode_q4_0 expects CPU tensor"
    original_shape = tuple(t.shape)
    original_dtype = t.dtype
    n_elements = t.numel()
    assert n_elements % Q4_0_BLOCK_SIZE == 0, \
        f"tensor numel {n_elements} not multiple of {Q4_0_BLOCK_SIZE}"
    n_blocks = n_elements // Q4_0_BLOCK_SIZE

    # to fp32 for numerical stability of the scale calc, then reshape (n_blocks, 32)
    flat = t.detach().contiguous().float().view(n_blocks, Q4_0_BLOCK_SIZE).numpy()

    # per-block scale: d = max_signed / -8  where max_signed = element with max |value|, keeping sign
    abs_max_idx = np.abs(flat).argmax(axis=1)  # (n_blocks,)
    max_signed = flat[np.arange(n_blocks), abs_max_idx]  # (n_blocks,) signed
    d = (max_signed / -8.0).astype(np.float32)  # (n_blocks,)
    # avoid div-by-zero (constant-zero block)
    # avoid div-by-zero warning: np.where evaluates BOTH branches.
    # use np.divide with `where=` and `out=` for masked compute.
    inv_d = np.divide(1.0, d, out=np.zeros_like(d), where=(d != 0.0)).astype(np.float32)

    # quantize: x_q = clamp(round(x * inv_d + 8.5), 0, 15)  -> uint4
    scaled = flat * inv_d[:, None] + 8.5
    q = np.clip(np.floor(scaled).astype(np.int32), 0, 15).astype(np.uint8)  # (n_blocks, 32)

    # pack: low nibble = x[0..15], high nibble = x[16..31]
    low = q[:, :16]
    high = q[:, 16:]
    packed = (low | (high << 4)).astype(np.uint8)  # (n_blocks, 16)

    # store scale as fp16
    scales_fp16 = d.astype(np.float16)

    return Q4_0_Encoded(
        scales=scales_fp16,
        quants=packed,
        original_shape=original_shape,
        original_dtype=original_dtype,
    )


def decode_q4_0(enc: Q4_0_Encoded) -> torch.Tensor:
    """Decode q4_0 back to original dtype. CPU only.

    Reference: x = (x_q - 8) * d
    """
    n_blocks = enc.quants.shape[0]
    # unpack: low nibble + high nibble
    low = (enc.quants & 0x0F).astype(np.int32)              # (n_blocks, 16)
    high = ((enc.quants >> 4) & 0x0F).astype(np.int32)      # (n_blocks, 16)
    q = np.concatenate([low, high], axis=1)                 # (n_blocks, 32)

    # dequantize: x = (q - 8) * d
    d_fp32 = enc.scales.astype(np.float32)                  # (n_blocks,)
    x = (q - 8).astype(np.float32) * d_fp32[:, None]        # (n_blocks, 32)

    flat = x.reshape(-1)
    t = torch.from_numpy(flat).to(enc.original_dtype).view(*enc.original_shape)
    return t


def encode_v_only_kv(K: torch.Tensor, V: torch.Tensor):
    """V-only KV codec: K passthrough, V q4_0 encoded."""
    assert K.dtype == V.dtype
    K_out = K.detach().clone()        # passthrough (caller can mmap/save raw bytes)
    V_enc = encode_q4_0(V)
    return K_out, V_enc


def decode_v_only_kv(K_passthrough: torch.Tensor, V_enc: Q4_0_Encoded):
    """V-only KV codec decode."""
    V_dec = decode_q4_0(V_enc)
    return K_passthrough, V_dec


def kv_round_trip_size_bytes(K: torch.Tensor, V_enc: Q4_0_Encoded) -> dict:
    """Report storage breakdown."""
    k_bytes = K.numel() * K.element_size()
    v_bytes = V_enc.storage_bytes
    orig_v_bytes = V_enc.n_elements * 2  # bf16 = 2 byte
    return {
        "K_bytes": k_bytes,
        "V_bytes_encoded": v_bytes,
        "V_bytes_original": orig_v_bytes,
        "V_compression_ratio": orig_v_bytes / v_bytes,
        "V_effective_bpw": V_enc.effective_bpw,
        "total_encoded": k_bytes + v_bytes,
        "total_original": k_bytes + orig_v_bytes,
        "total_reduction_pct": (1 - (k_bytes + v_bytes) / (k_bytes + orig_v_bytes)) * 100,
    }


# ============================================================================
# Self-test / invariants — run with: python -m src.kv_codec
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("kv_codec.py self-test")
    print("=" * 70)

    torch.manual_seed(42)

    # Test 1: small random bf16 tensor round-trip
    print("\n[Test 1] Random bf16 1x8x256x128 (matches GQA shape, 256 tokens, head_dim=128)")
    V = torch.randn(1, 8, 256, 128, dtype=torch.bfloat16) * 0.5  # realistic V scale
    enc = encode_q4_0(V)
    V_dec = decode_q4_0(enc)
    err = (V.float() - V_dec.float()).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()
    rmse = err.pow(2).mean().sqrt().item()
    # cos_sim via numpy for numerical stability on huge tensors
    # (F.cosine_similarity with eps can return >1.0 due to float32 accumulation)
    a = V.float().flatten().numpy(); b = V_dec.float().flatten().numpy()
    cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    cos_sim = min(1.0, max(-1.0, cos_sim))
    print(f"  shape: {tuple(V.shape)}, numel: {V.numel():,}")
    print(f"  encoded storage: {enc.storage_bytes:,} bytes ({enc.effective_bpw:.3f} bpw)")
    print(f"  max abs error:   {max_err:.6f}")
    print(f"  mean abs error:  {mean_err:.6f}")
    print(f"  rmse:            {rmse:.6f}")
    print(f"  cos_sim:         {cos_sim:.8f}")
    assert enc.effective_bpw == 4.5, f"bpw {enc.effective_bpw} != 4.5"
    # q4_0 symmetric ceiling: |d| ~= amax/8, signed 4-bit asymmetry (range -8..7)
    # gives worst-case element error up to |d|*1.0. For randn*0.5 inputs, amax~2 -> d~0.25,
    # bf16 rounding adds further, so ~0.3-0.4 is the realistic ceiling. 0.4 = safe with margin.
    assert max_err < 0.4, f"max_err {max_err} >= 0.4 (q4_0 ceiling violated, real bug)"
    assert cos_sim > 0.99, f"cos_sim {cos_sim} <= 0.99"
    print(f"  [OK] effective bpw exactly 4.5, max_err < 0.4, cos_sim > 0.99")

    # Test 2: edge — constant zero block (div-by-zero safety)
    print("\n[Test 2] All-zeros tensor (div-by-zero safety)")
    Z = torch.zeros(1, 1, 32, 1, dtype=torch.bfloat16)
    enc_z = encode_q4_0(Z)
    Z_dec = decode_q4_0(enc_z)
    assert torch.allclose(Z, Z_dec, atol=0), "zero round-trip failed"
    print(f"  [OK] zero block survives encode/decode")

    # Test 3: edge — single block 32 elements, all distinct
    print("\n[Test 3] Single block 32 elements (boundary test)")
    S = torch.randn(32, dtype=torch.bfloat16)
    enc_s = encode_q4_0(S)
    S_dec = decode_q4_0(enc_s)
    assert enc_s.quants.shape == (1, 16), f"shape {enc_s.quants.shape} != (1, 16)"
    assert enc_s.scales.shape == (1,), f"scales shape {enc_s.scales.shape}"
    a_s = S.float().numpy(); b_s = S_dec.float().numpy()
    cos_s = float(np.dot(a_s, b_s) / (np.linalg.norm(a_s) * np.linalg.norm(b_s) + 1e-12))
    cos_s = min(1.0, max(-1.0, cos_s))
    print(f"  shape: {tuple(S.shape)}, encoded blocks: 1, cos_sim: {cos_s:.6f}")
    print(f"  [OK] single block boundary verified")

    # Test 4: full Llama 3.1 70B per-layer V slab (1 layer, 32K context)
    print("\n[Test 4] Llama 70B per-layer V slab: 1x8x32000x128 (62.5 MB bf16)")
    V_layer = torch.randn(1, 8, 32000, 128, dtype=torch.bfloat16) * 0.3
    import time
    t_enc_s = time.time()
    enc_l = encode_q4_0(V_layer)
    t_enc_e = time.time()
    t_dec_s = time.time()
    V_layer_dec = decode_q4_0(enc_l)
    t_dec_e = time.time()
    err_l = (V_layer.float() - V_layer_dec.float()).abs()
    max_err_l = err_l.max().item()
    a_l = V_layer.float().flatten().numpy(); b_l = V_layer_dec.float().flatten().numpy()
    cos_l = float(np.dot(a_l, b_l) / (np.linalg.norm(a_l) * np.linalg.norm(b_l) + 1e-12))
    cos_l = min(1.0, max(-1.0, cos_l))
    print(f"  shape: {tuple(V_layer.shape)}, numel: {V_layer.numel():,}")
    print(f"  encoded: {enc_l.storage_bytes/1e6:.2f} MB (bpw {enc_l.effective_bpw:.3f})")
    print(f"  original bf16: {V_layer.numel()*2/1e6:.2f} MB")
    print(f"  compression: {(V_layer.numel()*2)/enc_l.storage_bytes:.3f}x")
    print(f"  encode time: {(t_enc_e-t_enc_s)*1000:.0f} ms")
    print(f"  decode time: {(t_dec_e-t_dec_s)*1000:.0f} ms")
    print(f"  max abs error: {max_err_l:.6f}, cos_sim: {cos_l:.6f}")
    assert cos_l > 0.99
    print(f"  [OK] per-layer V slab round-trip")

    # Test 5: V-only KV high-level API
    print("\n[Test 5] encode_v_only_kv / decode_v_only_kv (high-level API)")
    K = torch.randn(1, 8, 256, 128, dtype=torch.bfloat16) * 0.5
    V = torch.randn(1, 8, 256, 128, dtype=torch.bfloat16) * 0.5
    K_out, V_enc = encode_v_only_kv(K, V)
    assert torch.equal(K, K_out), "K must be passthrough (identical)"
    K_dec, V_dec = decode_v_only_kv(K_out, V_enc)
    assert torch.equal(K, K_dec), "K passthrough must survive decode"
    sizes = kv_round_trip_size_bytes(K, V_enc)
    print(f"  K bytes (passthrough):  {sizes['K_bytes']:,}")
    print(f"  V bytes (q4_0):         {sizes['V_bytes_encoded']:,}")
    print(f"  V bytes (orig bf16):    {sizes['V_bytes_original']:,}")
    print(f"  V compression:          {sizes['V_compression_ratio']:.3f}x")
    print(f"  Total reduction:        {sizes['total_reduction_pct']:.2f}%")
    expected = 35.9  # see configs/hardware/q4_0_quant.json
    actual = sizes["total_reduction_pct"]
    assert abs(actual - expected) < 1.0, \
        f"Total reduction {actual:.2f}% vs expected {expected}% (off by >1%)"
    print(f"  [OK] K bit-exact passthrough, total reduction matches expected {expected}%")

    print("\n" + "=" * 70)
    print("All 5 tests PASS")
    print("=" * 70)
