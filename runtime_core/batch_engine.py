import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache, DynamicLayer
import json
import time
import uuid
import threading
import queue
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
    def __init__(self, request_id: str, prompt: str, prompt_len: int, response_queue: queue.Queue):
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_len = prompt_len
        self.response_queue = response_queue
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("Loading model in 4-bit...")
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
            
        self.pending_queue = queue.Queue()
        self.thread = None
        self.running = False
        
    def start(self):
        """Start the background thread for continuous batching."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
            
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def submit(self, prompt: str) -> tuple[str, queue.Queue]:
        """
        Thread-safe method to submit a prompt.
        Returns the request_id and a queue.Queue that will receive tokens.
        """
        req_id = f"req-{uuid.uuid4().hex[:8]}"
        resp_q = queue.Queue()
        self.pending_queue.put({"id": req_id, "prompt": prompt, "response_queue": resp_q})
        return req_id, resp_q

    def get_stats(self) -> dict:
        """Helper to return current stats safely."""
        return {
            "status": "running" if self.running else "stopped",
            "pending": self.pending_queue.qsize()
        }

    def _log_event(self, event: dict):
        event["timestamp"] = time.time()
        try:
            with open("batch_events.jsonl", "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass

    def _run_loop(self):
        """
        The continuous batching core loop, running in a dedicated thread.
        """
        active_slots: List[RequestState] = []
        
        while self.running:
            admitted_this_step = []
            
            # 1. Slot Admission (Prefill)
            while len(active_slots) < MAX_SLOTS and not self.pending_queue.empty():
                # Peek at the next request without blocking
                next_req = self.pending_queue.queue[0]
                
                # Check token budget
                inputs = self.tokenizer(next_req["prompt"], return_tensors="pt").to(self.device)
                prompt_len = inputs.input_ids.shape[1]
                
                current_live_tokens = sum((s.prompt_len + s.tokens_produced) for s in active_slots)
                if current_live_tokens + prompt_len > TOKEN_BUDGET:
                    break # Budget exceeded
                
                # Admit
                next_req = self.pending_queue.get()
                state = RequestState(next_req["id"], next_req["prompt"], prompt_len, next_req["response_queue"])
                
                cache = DynamicCache()
                with torch.no_grad():
                    outputs = self.model(**inputs, past_key_values=cache)
                
                state.cache = outputs.past_key_values
                state.latest_token = outputs.logits[0, -1].argmax().unsqueeze(0)
                state.token_ids.append(state.latest_token.item())
                state.tokens_produced += 1
                state.slot_index = len(active_slots)
                
                # Send the first generated token back immediately
                first_token = self.tokenizer.decode([state.latest_token.item()], skip_special_tokens=True)
                if first_token:
                    state.response_queue.put({"type": "token", "content": first_token})
                
                active_slots.append(state)
                admitted_this_step.append(state.request_id)
                
            if not active_slots:
                time.sleep(0.01) # Sleep briefly to prevent 100% CPU loop
                continue
                
            # 2. Decode Step (Continuous Batching)
            input_ids = torch.cat([s.latest_token.view(1, 1) for s in active_slots], dim=0)
            
            if hasattr(active_slots[0].cache, 'layers'):
                orig_lens = [s.cache.layers[0].keys.shape[2] for s in active_slots]
            else:
                orig_lens = [s.cache.key_cache[0].shape[2] for s in active_slots]
                
            max_len = max(orig_lens)
            combined_cache = combine_caches([s.cache for s in active_slots], max_len)
            
            attention_mask = torch.zeros((len(active_slots), max_len + 1), dtype=torch.long, device=self.device)
            for b, orig_len in enumerate(orig_lens):
                attention_mask[b, -(orig_len + 1):] = 1
                
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=combined_cache
                )
                
            split_caches = split_cache(outputs.past_key_values, orig_lens)
            
            freed_this_step = []
            for b, state in enumerate(active_slots):
                state.cache = split_caches[b]
                new_token = outputs.logits[b, -1].argmax().unsqueeze(0)
                state.latest_token = new_token
                state.token_ids.append(new_token.item())
                state.tokens_produced += 1
                
                # Stream the new token
                token_str = self.tokenizer.decode([new_token.item()], skip_special_tokens=True)
                if token_str:
                    state.response_queue.put({"type": "token", "content": token_str})
                
                if new_token.item() == self.tokenizer.eos_token_id or state.tokens_produced >= 200:
                    state.finished = True
                    state.response_queue.put({"type": "done"})
                    freed_this_step.append(state.request_id)
                    
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
