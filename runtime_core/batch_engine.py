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
    Holds the state for a single user's request while it is being processed.
    This keeps track of what the user asked, how much of the response has been
    generated so far, and the model's memory of this specific conversation.
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
    Takes the individual memories (caches) of several different conversations and 
    stacks them together into one big block so the AI model can process them all at once.
    
    Inputs:
    - caches: A list of individual conversation memories.
    - max_len: The length of the longest conversation currently being processed.
    
    Returns:
    - A single combined memory block that contains everyone's conversation, ready for the model.
    """
    combined = DynamicCache()
    if not caches:
        return combined
    
    num_layers = len(caches[0].layers) if hasattr(caches[0], 'layers') else len(caches[0].key_cache)
    
    for layer_idx in range(num_layers):
        keys_list = []
        values_list = []
        for cache in caches:
            if hasattr(cache, 'layers'):
                keys = cache.layers[layer_idx].keys
                values = cache.layers[layer_idx].values
            else:
                keys = cache.key_cache[layer_idx]
                values = cache.value_cache[layer_idx]
                
            # Some requests' conversations are longer than others.
            # Before we can process them together, we need to make them all the same length
            # by adding harmless filler (zeros) to the shorter ones — like adding blank pages 
            # to a short book so it's as thick as the others on the shelf. We add this filler 
            # to the *left* side (the beginning) so the newest, most important words align 
            # on the right side.
            pad_len = max_len - keys.shape[2]
            if pad_len > 0:
                keys = torch.nn.functional.pad(keys, (0, 0, pad_len, 0))
                values = torch.nn.functional.pad(values, (0, 0, pad_len, 0))
            keys_list.append(keys)
            values_list.append(values)
            
        # Stack all the identically-sized memories on top of each other into a single block
        combined_keys = torch.cat(keys_list, dim=0)
        combined_values = torch.cat(values_list, dim=0)
        
        if hasattr(combined, 'layers'):
            layer = DynamicLayer()
            layer.keys = combined_keys
            layer.values = combined_values
            layer.is_initialized = True
            combined.layers.append(layer)
        else:
            combined.key_cache.append(combined_keys)
            combined.value_cache.append(combined_values)
            
    return combined

def split_cache(combined: DynamicCache, original_lengths: List[int]) -> List[DynamicCache]:
    """
    Takes the big combined memory block returned by the model and splits it back up 
    into individual memories for each person's conversation.
    
    Inputs:
    - combined: The big combined memory block updated by the model.
    - original_lengths: A list showing how long each person's real conversation actually was 
      (ignoring the blank filler pages we added earlier).
      
    Returns:
    - A list of individual, un-padded memories that can be safely stored until the next step.
    """
    caches = [DynamicCache() for _ in original_lengths]
    
    num_layers = len(combined.layers) if hasattr(combined, 'layers') else len(combined.key_cache)
    
    for layer_idx in range(num_layers):
        if hasattr(combined, 'layers'):
            combined_keys = combined.layers[layer_idx].keys
            combined_values = combined.layers[layer_idx].values
        else:
            combined_keys = combined.key_cache[layer_idx]
            combined_values = combined.value_cache[layer_idx]
            
        for batch_idx, original_length in enumerate(original_lengths):
            # The model just added 1 new word to the conversation.
            new_len = original_length + 1
            
            # We slice the tensor to keep ONLY the real words from the right side of the block,
            # completely discarding the blank filler pages we added on the left side earlier.
            keys_slice = combined_keys[batch_idx:batch_idx+1, :, -new_len:, :]
            values_slice = combined_values[batch_idx:batch_idx+1, :, -new_len:, :]
            
            if hasattr(caches[batch_idx], 'layers'):
                layer = DynamicLayer()
                layer.keys = keys_slice
                layer.values = values_slice
                layer.is_initialized = True
                caches[batch_idx].layers.append(layer)
            else:
                caches[batch_idx].key_cache.append(keys_slice)
                caches[batch_idx].value_cache.append(values_slice)
            
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
        """Starts the background engine thread so it can begin processing requests continuously."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
            
    def stop(self):
        """Stops the background engine thread safely."""
        self.running = False
        if self.thread:
            self.thread.join()

    def submit(self, prompt: str) -> tuple[str, queue.Queue]:
        """
        Accepts a new user's question and queues it up to be answered by the model.
        
        Inputs:
        - prompt: The text of the user's question.
        
        Returns:
        - A unique ID for the request, and a communication channel (queue.Queue) where 
          the model will drop the answer word-by-word as it thinks of them.
        """
        req_id = f"req-{uuid.uuid4().hex[:8]}"
        resp_q = queue.Queue()
        self.pending_queue.put({"id": req_id, "prompt": prompt, "response_queue": resp_q})
        return req_id, resp_q

    def get_stats(self) -> dict:
        """Returns basic statistics about how busy the engine currently is."""
        return {
            "status": "running" if self.running else "stopped",
            "pending": self.pending_queue.qsize()
        }

    def _log_event(self, event: dict):
        """Writes a record of what the engine did (e.g. starting or finishing a request) to a log file."""
        event["timestamp"] = time.time()
        try:
            with open("batch_events.jsonl", "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass

    def _run_loop(self):
        """
        The main engine loop that runs continuously in the background.
        It continuously grabs new questions, batches them together, asks the model to generate 
        the next word for everyone simultaneously, and sends those words back to the users.
        """
        active_slots: List[RequestState] = []
        
        while self.running:
            admitted_this_step = []
            
            # 1. Slot Admission: Check if we have room to take on new users' questions.
            while len(active_slots) < MAX_SLOTS and not self.pending_queue.empty():
                # Peek at the next request without removing it yet
                next_req = self.pending_queue.queue[0]
                
                # We need to make sure admitting this new person won't crash the computer by using too much memory.
                inputs = self.tokenizer(next_req["prompt"], return_tensors="pt").to(self.device)
                prompt_len = inputs.input_ids.shape[1]
                
                current_live_tokens = sum((s.prompt_len + s.tokens_produced) for s in active_slots)
                if current_live_tokens + prompt_len > TOKEN_BUDGET:
                    break # The computer's memory is too full; we must wait for someone else to finish first.
                
                # We have capacity, so officially accept the request from the waiting line.
                next_req = self.pending_queue.get()
                
                state = RequestState(next_req["id"], next_req["prompt"], prompt_len, next_req["response_queue"])
                
                # "Prefill" phase: The model reads the user's entire prompt all at once to build its initial memory.
                cache = DynamicCache()
                with torch.no_grad():
                    outputs = self.model(**inputs, past_key_values=cache)
                
                state.cache = outputs.past_key_values
                state.latest_token = outputs.logits[0, -1].argmax().unsqueeze(0)
                state.token_ids.append(state.latest_token.item())
                state.tokens_produced += 1
                state.slot_index = len(active_slots)
                
                # Send the very first word back to the user immediately so they know we started.
                first_token = self.tokenizer.decode([state.latest_token.item()], skip_special_tokens=True)
                if first_token:
                    state.response_queue.put({"type": "token", "content": first_token})
                
                active_slots.append(state)
                admitted_this_step.append(state.request_id)
                
            # If no one is asking questions right now, take a brief nap so we don't overwork the computer.
            if not active_slots:
                time.sleep(0.01)
                continue
                
            # 2. Decode Step: Generate exactly one new word for everyone currently in a slot, simultaneously.
            
            # Gather the last word everyone just said, so the model knows what to continue from.
            input_ids = torch.cat([s.latest_token.view(1, 1) for s in active_slots], dim=0)
            
            if hasattr(active_slots[0].cache, 'layers'):
                original_lengths = [s.cache.layers[0].keys.shape[2] for s in active_slots]
            else:
                original_lengths = [s.cache.key_cache[0].shape[2] for s in active_slots]
                
            max_len = max(original_lengths)
            
            # Combine everyone's memory into one big padded block.
            combined_cache = combine_caches([s.cache for s in active_slots], max_len)
            
            # The model needs to know which parts of the combined memory block are real conversation,
            # and which parts are just the blank filler zeros we added to make them all the same length.
            # If we don't tell the model to ignore the zeros, it will read them as actual words (like "blank blank blank")
            # and generate complete gibberish as a response. We do this by creating a "mask" where 1 means "real" 
            # and 0 means "ignore this".
            attention_mask = torch.zeros((len(active_slots), max_len + 1), dtype=torch.long, device=self.device)
            for batch_idx, original_length in enumerate(original_lengths):
                # We mark the end of the block as "1" (real data), matching the length of their true conversation
                # plus the 1 new word we are asking the model to generate right now.
                attention_mask[batch_idx, -(original_length + 1):] = 1
                
            # Run the model once to generate the next word for ALL active users at the exact same time.
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=combined_cache
                )
                
            # Split the model's updated memory back into individual pieces so they don't get mixed up.
            split_caches = split_cache(outputs.past_key_values, original_lengths)
            
            freed_this_step = []
            for batch_idx, state in enumerate(active_slots):
                state.cache = split_caches[batch_idx]
                new_token = outputs.logits[batch_idx, -1].argmax().unsqueeze(0)
                state.latest_token = new_token
                state.token_ids.append(new_token.item())
                state.tokens_produced += 1
                
                # Translate the computer's token ID back into a readable word, and send it to the user.
                token_str = self.tokenizer.decode([new_token.item()], skip_special_tokens=True)
                if token_str:
                    state.response_queue.put({"type": "token", "content": token_str})
                
                # Check if the model said "I'm done" (the eos token), or if the conversation is dragging on too long (limit 200).
                if new_token.item() == self.tokenizer.eos_token_id or state.tokens_produced >= 200:
                    state.finished = True
                    state.response_queue.put({"type": "done"})
                    freed_this_step.append(state.request_id)
                    
            # 3. Log what just happened so we can track the system's performance.
            current_live_tokens = sum((s.prompt_len + s.tokens_produced) for s in active_slots)
            self._log_event({
                "active_requests": [s.request_id for s in active_slots],
                "tokens_processed": len(active_slots),
                "slots_freed": freed_this_step,
                "requests_admitted": admitted_this_step,
                "current_total_live_tokens": current_live_tokens
            })
            
            # 4. Kick out anyone who finished their response, freeing up their slot for the next person in line.
            active_slots = [s for s in active_slots if not s.finished]
