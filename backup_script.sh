#!/bin/bash
set -e

echo "=== 1. Git Status Check ==="
cd ~/cognitive-prefetch-poc
git status -s

echo -e "\n=== 2. Staging NIXL Plugin Files ==="
git add src/nixl_interceptor.py
git add src/nixl_core.py
git add src/nixl_codec.py
git add src/nixl_plugin_manager.py
git add src/cognitive_cache_v2.py
git add scripts/cognitive_demo.py
echo "Files staged."

echo -e "\n=== 3. Committing to Local Git ==="
git commit -m "feat(nixl): integrate async byte-stream compression plugin

- Add NixlInterceptor for zero-copy tensor hooking
- Add NixlSerializationCore for Solidigm 64KB shaping
- Add NixlAsyncCodec for non-blocking zlib compression
- Add NixlSolidigmPlugin manager and wire into cognitive_cache_v2.py
- Patch cognitive_demo.py to display GC-inspected physical HBM/PCIe metrics"
echo "Commit successful."

echo -e "\n=== 4. Creating Tarball Backup ==="
cd ~
tar czf /tmp/cog-backup-nixl.tar.gz --exclude='*.venv*' --exclude='*__pycache__*' --exclude='*.pyc' cognitive-prefetch-poc/
ls -lh /tmp/cog-backup-nixl.tar.gz
echo "Tarball created at /tmp/cog-backup-nixl.tar.gz"

echo -e "\n=== 5. SCP Instruction ==="
echo "Run this command on your Macbook to download the backup:"
echo "scp ubuntu@192.222.55.44:/tmp/cog-backup-nixl.tar.gz ~/Desktop/"
