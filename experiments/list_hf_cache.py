import os
from huggingface_hub import constants

cache_dir = constants.HF_HUB_CACHE
print(f"HF_CACHE_HOME: {cache_dir}")
for root, dirs, files in os.walk(cache_dir):
    for f in files:
        if "Qwen" in root or "Qwen" in f:
            full_path = os.path.join(root, f)
            print(f"{full_path} - {os.path.getsize(full_path)} bytes")
