import hashlib

# BLOCK_SIZE is duplicated from kv_cache_manager.py on purpose. 
# Importing it from kv_cache_manager.py would create a circular import once 
# kv_cache_manager starts importing from this file in a later sub-phase.
BLOCK_SIZE = 16

# GENESIS_PARENT_HASH exists so that block 0 has a defined parent, even though 
# there is no real block before it. We use a 32-byte zero hash.
GENESIS_PARENT_HASH = b'\x00' * 32

def compute_block_hashes(token_ids: list[int]) -> list[tuple[int, str]]:
    """
    Computes a chain of SHA-256 hashes for each complete BLOCK_SIZE chunk of token_ids.
    
    This function processes only complete 16-token blocks. Any trailing tokens 
    that do not form a complete block are ignored.
    
    The hashing chains blocks together: Block N's hash is computed by concatenating
    the bytes of Block N-1's hash with its own 16 token IDs (each encoded as a 
    little-endian 4-byte integer). This chaining behavior means that changing 
    block 0 will change block 1's hash, and so forth.
    """
    if not isinstance(token_ids, list):
        raise ValueError(f"token_ids must be a list, got {type(token_ids)}")
        
    for idx, t in enumerate(token_ids):
        if not isinstance(t, int) or isinstance(t, bool):
            raise ValueError(f"token_ids must contain only ints, found {type(t)} at index {idx}")
        if t < 0:
            raise ValueError(f"token_ids must contain only non-negative ints, found {t} at index {idx}")

    num_complete_blocks = len(token_ids) // BLOCK_SIZE
    if num_complete_blocks == 0:
        return []
        
    results = []
    prev_hash_bytes = GENESIS_PARENT_HASH
    
    for block_idx in range(num_complete_blocks):
        start_idx = block_idx * BLOCK_SIZE
        end_idx = start_idx + BLOCK_SIZE
        block_tokens = token_ids[start_idx:end_idx]
        
        hasher = hashlib.sha256()
        hasher.update(prev_hash_bytes)
        
        for token in block_tokens:
            hasher.update(token.to_bytes(4, byteorder='little'))
            
        current_hash_bytes = hasher.digest()
        results.append((block_idx, current_hash_bytes.hex()))
        
        prev_hash_bytes = current_hash_bytes
        
    return results

def max_shareable_blocks(prompt_len: int) -> int:
    """
    Returns the maximum number of blocks that can be shared in the prefix cache.
    
    We subtract one from prompt_len because at least one prompt token must always 
    be freshly computed to trigger the generation loop naturally. This means the 
    very last block is never treated as fully shareable, even if it happens to 
    align exactly on a 16-token boundary.
    """
    return max(0, (prompt_len - 1) // BLOCK_SIZE)
