with open('scripts/cognitive_demo.py', 'r') as f:
    content = f.read()

# 파이썬 가비지 컬렉터로 메모리 상의 플러그인을 강제 추적하여 메트릭을 뽑아냅니다.
nixl_hook = """
    # --- NIXL Backend Metrics Display ---
    import gc
    for obj in gc.get_objects():
        if type(obj).__name__ == 'NixlSolidigmPlugin':
            m = obj.get_metrics()
            if m['total_original_bytes'] > 0:
                print("\\n" + "="*72)
                print("  NIXL STORAGE COMPRESSION BACKEND (PHYSICAL MEASURED)")
                print("="*72)
                print(f"   Intercepted Tensors: {m['intercept_calls']} calls")
                print(f"   Original Data Size:  {m['total_original_bytes'] / (1024*1024):.2f} MB")
                print(f"   Compressed for SSD:  {m['total_compressed_bytes'] / (1024*1024):.2f} MB")
                saved_pct = (1 - m['total_compressed_bytes']/m['total_original_bytes'])*100
                print(f"   Physical PCIe Save:  {saved_pct:.2f}% (Real-time zlib)\\n")
            break
"""

# HONEST LIMITATIONS 블록 직전에 안전하게 삽입합니다.
content = content.replace('print("  HONEST LIMITATIONS', nixl_hook + '\n    print("  HONEST LIMITATIONS')

with open('scripts/cognitive_demo.py', 'w') as f:
    f.write(content)

print("Demo script repatched using GC inspection.")
