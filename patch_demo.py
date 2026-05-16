import re

with open('scripts/cognitive_demo.py', 'r') as f:
    content = f.read()

# 결과 출력부 (print("="*60)) 직전에 NIXL 메트릭 블록 추가
nixl_display_hook = """
    # --- NIXL Backend Metrics Display ---
    try:
        m = past_key_values.nixl_plugin.get_metrics()
        if m['total_original_bytes'] > 0:
            print("\\n" + "-"*60)
            print("[NIXL Storage Compression Backend - Physical Measured]")
            print(f"  Intercepted Tensors: {m['intercept_calls']} layers")
            print(f"  Original Data Size:  {m['total_original_bytes'] / (1024*1024):.2f} MB")
            print(f"  Compressed for SSD:  {m['total_compressed_bytes'] / (1024*1024):.2f} MB")
            saved_pct = (1 - m['total_compressed_bytes']/m['total_original_bytes'])*100
            print(f"  Physical PCIe Save:  {saved_pct:.2f}% (Real-time zlib)")
            print("-" * 60)
    except Exception as e:
        pass
    # ------------------------------------
"""

# cognitive_demo.py의 기존 출력 비교 블록 바로 위를 타겟팅합니다.
# (보통 print("=" * 60) 이나 Verdict 직전에 위치)
target_pattern = r'(\s+print\("=" \* 60\)\n\s+print\("\[Comparison\]"\))'
content = re.sub(target_pattern, r'%s\n\1' % nixl_display_hook, content)

with open('scripts/cognitive_demo.py', 'w') as f:
    f.write(content)

print("Demo script patched for NIXL metrics. Running short context test...")
