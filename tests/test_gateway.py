import uvicorn
import torch
from runtime_core.kv_cache_manager import KVCacheManager

# Patch the KVCacheManager to have a small pool for testing OOM
original_init = KVCacheManager.__init__
def new_init(self, device, max_blocks=20, dtype=torch.bfloat16): 
    original_init(self, device, max_blocks=max_blocks, dtype=dtype)
KVCacheManager.__init__ = new_init

if __name__ == "__main__":
    uvicorn.run("gateway:app", port=8004, log_level="info")
