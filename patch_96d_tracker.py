with open('src/cognitive_cache_v2.py', 'r') as f:
    lines = f.readlines()

print("=== 9-6d Target Lines for CPU Slicing ===")
for i, line in enumerate(lines):
    # GPU 전송(to(device))과 관련된 캐시 전송 로직 추적
    if '.to(' in line and ('key_cache' in line or 'value_cache' in line or 'K' in line or 'V' in line or 'subset' in line):
        print(f"Line {i+1}: {line.strip()}")
print("=========================================")
