import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache, DynamicLayer
import json
import time
from typing import List, Dict, Optional

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
# Starting constants for tuning in Stage 2.
# MAX_SLOTS: Maximum number of concurrent requests in the batch.
MAX_SLOTS = 4
# TOKEN_BUDGET: Maximum combined sum of active tokens (prompt + generated) allowed in a batch.
TOKEN_BUDGET = 10000

class RequestState:
    """
    Holds the per-request state for dynamic continuous batching.
    """
    def __init__(self, request_id: str, prompt: str, prompt_len: int):
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_len = prompt_len
        self.token_ids: List[int] = []
        self.finished = False
        self.tokens_produced = 0
        self.slot_index = -1
        
        # HuggingFace DynamicCache representing this sequence's KV cache.
        self.cache: Optional[DynamicCache] = None
        # The most recently generated token ID (shape: [1, 1])
        self.latest_token: Optional[torch.Tensor] = None

def combine_caches(caches: List[DynamicCache], max_len: int) -> DynamicCache:
    """
    Combines individual sequence caches into a single padded batch cache.
    Because sequences have different lengths, we left-pad shorter KV tensors 
    with zeros up to max_len so they can be stacked into a single tensor 
    for the batched forward pass.
    """
    combined = DynamicCache()
    if not caches:
        return combined
    
    num_layers = len(caches[0].layers) if hasattr(caches[0], 'layers') else len(caches[0].key_cache)
    
    for layer_idx in range(num_layers):
        k_tensors = []
        v_tensors = []
        for c in caches:
            if hasattr(c, 'layers'):
                k = c.layers[layer_idx].keys
                v = c.layers[layer_idx].values
            else:
                k = c.key_cache[layer_idx]
                v = c.value_cache[layer_idx]
                
            pad_len = max_len - k.shape[2]
            if pad_len > 0:
                # Left-pad on the seq_len dimension (dim 2)
                k = torch.nn.functional.pad(k, (0, 0, pad_len, 0))
                v = torch.nn.functional.pad(v, (0, 0, pad_len, 0))
            k_tensors.append(k)
            v_tensors.append(v)
            
        combined_k = torch.cat(k_tensors, dim=0)
        combined_v = torch.cat(v_tensors, dim=0)
        
        if hasattr(combined, 'layers'):
            layer = DynamicLayer()
            layer.keys = combined_k
            layer.values = combined_v
            layer.is_initialized = True
            combined.layers.append(layer)
        else:
            combined.key_cache.append(combined_k)
            combined.value_cache.append(combined_v)
            
    return combined

def split_cache(combined: DynamicCache, orig_lens: List[int]) -> List[DynamicCache]:
    """
    Splits a padded batch cache back into individual sequence caches.
    Extracts only the valid (unpadded) tokens for each sequence, which now
    includes the newly generated token (orig_len + 1).
    """
    caches = [DynamicCache() for _ in orig_lens]
    
    num_layers = len(combined.layers) if hasattr(combined, 'layers') else len(combined.key_cache)
    
    for layer_idx in range(num_layers):
        if hasattr(combined, 'layers'):
            k_comb = combined.layers[layer_idx].keys
            v_comb = combined.layers[layer_idx].values
        else:
            k_comb = combined.key_cache[layer_idx]
            v_comb = combined.value_cache[layer_idx]
            
        for b, orig_len in enumerate(orig_lens):
            new_len = orig_len + 1
            # Slice out the valid tokens from the right side of the padded tensor
            k_slice = k_comb[b:b+1, :, -new_len:, :]
            v_slice = v_comb[b:b+1, :, -new_len:, :]
            
            if hasattr(caches[b], 'layers'):
                layer = DynamicLayer()
                layer.keys = k_slice
                layer.values = v_slice
                layer.is_initialized = True
                caches[b].layers.append(layer)
            else:
                caches[b].key_cache.append(k_slice)
                caches[b].value_cache.append(v_slice)
            
    return caches

class BatchEngine:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        print("Initializing BatchEngine...")
        # Load tokenizer with left-padding. 
        # Left-padding ensures that the most recent token (the end of the sequence) 
        # is aligned across all batch items. This makes position IDs and causal masks 
        # much simpler to handle for next-token generation on variable-length sequences.
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        
        # Qwen models usually don't have a pad_token set by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("Loading model in 4-bit...")
        # Load in 4-bit as verified in the preflight
        quant_config = BitsAndBytesConfig(load_in_4bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto"
        )
        self.device = self.model.device
        
        # Initialize event log
        with open("batch_events.jsonl", "w") as f:
            pass

    def _log_event(self, event: dict):
        event["timestamp"] = time.time()
        with open("batch_events.jsonl", "a") as f:
            f.write(json.dumps(event) + "\n")

    def run_queue(self, initial_queue: List[Dict], arrival_delays: List[float] = None) -> Dict[str, str]:
        """
        Runs the full continuous batching loop until all requests are finished.
        Allows simulating arrival delays.
        """
        queue = initial_queue.copy()
        active_slots: List[RequestState] = []
        completed_outputs = {}
        
        start_time = time.time()
        
        while queue or active_slots:
            current_time = time.time() - start_time
            
            # 1. Slot Admission (Prefill)
            admitted_this_step = []
            
            # We can only admit if we have free slots and there are requests in the queue
            while len(active_slots) < MAX_SLOTS and queue:
                # Check simulated arrival time
                idx = len(initial_queue) - len(queue)
                if arrival_delays and current_time < arrival_delays[idx]:
                    break # Not arrived yet
                    
                next_req = queue[0]
                # Pre-calculate prompt length to check token budget
                inputs = self.tokenizer(next_req["prompt"], return_tensors="pt").to(self.device)
                prompt_len = inputs.input_ids.shape[1]
                
                current_live_tokens = sum((s.prompt_len + s.tokens_produced) for s in active_slots)
                if current_live_tokens + prompt_len > TOKEN_BUDGET:
                    break # Budget exceeded, cannot admit
                
                # We have capacity, admit the request!
                queue.pop(0)
                state = RequestState(next_req["id"], next_req["prompt"], prompt_len)
                
                # Perform the PREFILL pass separately for this single request
                # This initializes its KV cache with the full prompt context.
                cache = DynamicCache()
                with torch.no_grad():
                    outputs = self.model(**inputs, past_key_values=cache)
                
                state.cache = outputs.past_key_values
                state.latest_token = outputs.logits[0, -1].argmax().unsqueeze(0)
                state.token_ids.append(state.latest_token.item())
                state.tokens_produced += 1
                state.slot_index = len(active_slots)
                
                active_slots.append(state)
                admitted_this_step.append(state.request_id)
                
            # If nothing is active and nothing arrived, just sleep to simulate time passing
            if not active_slots:
                time.sleep(0.1)
                continue
                
            # 2. Decode Step (Continuous Batching)
            # Gather the single "latest token" from every active slot.
            input_ids = torch.cat([s.latest_token.view(1, 1) for s in active_slots], dim=0)
            
            # Original lengths of the KV caches
            if hasattr(active_slots[0].cache, 'layers'):
                orig_lens = [s.cache.layers[0].keys.shape[2] for s in active_slots]
            else:
                orig_lens = [s.cache.key_cache[0].shape[2] for s in active_slots]
                
            max_len = max(orig_lens)
            
            # Combine all individual caches into one unified batch cache
            combined_cache = combine_caches([s.cache for s in active_slots], max_len)
            
            # Create attention mask to ignore the left-padding in the combined cache
            attention_mask = torch.zeros((len(active_slots), max_len + 1), dtype=torch.long, device=self.device)
            for b, orig_len in enumerate(orig_lens):
                # The valid tokens are the most recent `orig_len + 1` tokens 
                # (the existing cache + the 1 new input token)
                attention_mask[b, -(orig_len + 1):] = 1
                
            # Run one batched forward pass for all active requests!
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=combined_cache
                )
                
            # Split the unified cache back into individual sequence caches
            split_caches = split_cache(outputs.past_key_values, orig_lens)
            
            # Update states and check for completion
            freed_this_step = []
            for b, state in enumerate(active_slots):
                state.cache = split_caches[b]
                new_token = outputs.logits[b, -1].argmax().unsqueeze(0)
                state.latest_token = new_token
                state.token_ids.append(new_token.item())
                state.tokens_produced += 1
                
                # Check completion: EOS token or safety limit (e.g. 200 tokens)
                # 200 is a reasonable default to prevent infinite loops while allowing substantial responses.
                if new_token.item() == self.tokenizer.eos_token_id or state.tokens_produced >= 200:
                    state.finished = True
                    freed_this_step.append(state.request_id)
                    completed_outputs[state.request_id] = self.tokenizer.decode(state.token_ids, skip_special_tokens=True)
                    
            # 3. Log event
            current_live_tokens = sum((s.prompt_len + s.tokens_produced) for s in active_slots)
            self._log_event({
                "active_requests": [s.request_id for s in active_slots],
                "tokens_processed": len(active_slots),
                "slots_freed": freed_this_step,
                "requests_admitted": admitted_this_step,
                "current_total_live_tokens": current_live_tokens
            })
            
            # 4. Remove finished slots
            active_slots = [s for s in active_slots if not s.finished]
            
        return completed_outputs
