import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache, DynamicLayer
import json
import time
import uuid
import threading
import queue
from typing import List, Dict, Optional, Any
from .kv_cache_manager import KVCacheManager

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
    def __init__(self, request_id: str, prompt: str, prompt_len: int, response_queue: queue.Queue, tier: str = "free", max_tokens: int = 100):
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_len = prompt_len
        self.response_queue = response_queue
        self.tier = tier
        self.max_tokens = max_tokens
        self.token_ids: List[int] = []
        self.finished = False
        self.tokens_produced = 0
        self.slot_index = -1
        
        # List of physical block IDs assigned to this sequence by the KV Cache Manager.
        self.cache: Optional[List[int]] = []
        # The most recently generated token ID (shape: [1, 1])
        self.latest_token: Optional[torch.Tensor] = None

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
        
        print("Initializing KV Cache Manager...")
        self.kv_manager = KVCacheManager(device=self.device, dtype=self.model.dtype)
        
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

    def submit(self, prompt: str, tier: str = "free", max_tokens: int = 100) -> tuple[str, queue.Queue]:
        """
        Takes a new question from a user and puts it in the waiting line.
        Returns a unique ID for the request and a personal mailbox (queue) where
        the words will be dropped as they are generated.
        """
        req_id = str(uuid.uuid4())
        resp_q = queue.Queue()
        self.pending_queue.put({"id": req_id, "prompt": prompt, "response_queue": resp_q, "tier": tier, "max_tokens": max_tokens})
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
        except Exception as e:
            print(f"Failed to log batch event: {e}")

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
                
                # DECISION: TOKEN_BUDGET admission check was deliberately kept token-based. 
                # While a block-count-based check would reflect the allocator's true state, 
                # the block allocator itself now enforces the hard physical memory ceiling underneath 
                # via its eviction policy. Keeping the token budget provides a soft limit for admission, 
                # while eviction dynamically handles the real hard limit.
                current_live_tokens = sum((s.prompt_len + s.tokens_produced) for s in active_slots)
                if current_live_tokens + prompt_len > TOKEN_BUDGET:
                    break # The computer's memory is too full; we must wait for someone else to finish first.
                
                # We have capacity, so officially accept the request from the waiting line.
                next_req = self.pending_queue.get()
                
                state = RequestState(
                    request_id=next_req["id"],
                    prompt=next_req["prompt"],
                    prompt_len=prompt_len,
                    response_queue=next_req["response_queue"],
                    tier=next_req.get("tier", "free"),
                    max_tokens=next_req.get("max_tokens", 100)
                )
                
                # "Prefill" phase: The model reads the user's entire prompt all at once to build its initial memory.
                cache = DynamicCache()
                with torch.no_grad():
                    outputs = self.model(**inputs, past_key_values=cache)
                
                # Immediately ingest the prefill cache into the block allocator
                self.kv_manager.ingest_prefill(state, outputs.past_key_values, active_slots)
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
            
            original_lengths = [s.prompt_len + s.tokens_produced for s in active_slots]
            max_len = max(original_lengths)
            
            # Combine everyone's memory into one big padded block by reading scattered physical blocks.
            combined_cache = self.kv_manager.reconstruct_caches(active_slots, max_len)
                
            
            # Combine everyone's memory into one big padded block.
            
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
            # Distribute the newly generated keys and values back into the scattered blocks.
            self.kv_manager.redistribute_caches(active_slots, outputs.past_key_values)
            
            freed_this_step = []
            for batch_idx, state in enumerate(active_slots):
                new_token = outputs.logits[batch_idx, -1].argmax().unsqueeze(0)
                state.latest_token = new_token
                state.token_ids.append(new_token.item())
                state.tokens_produced += 1
                
                # Translate the computer's token ID back into a readable word, and send it to the user.
                token_str = self.tokenizer.decode([new_token.item()], skip_special_tokens=True)
                if token_str:
                    state.response_queue.put({"type": "token", "content": token_str})
                
                # Check if the model said "I'm done" (the eos token), or if the conversation is dragging on too long (limit max_tokens).
                if new_token.item() == self.tokenizer.eos_token_id or state.tokens_produced >= state.max_tokens:
                    state.finished = True
                    state.response_queue.put({"type": "done"})
                    freed_this_step.append(state.request_id)
                    # Free the sequence immediately when it finishes!
                    self.kv_manager.free_sequence(state.request_id)
                    
            # 4. Kick out anyone who finished their response, freeing up their slot for the next person in line.
            # We do this before logging so the logs show the immediately freed blocks and updated live tokens.
            active_slots = [s for s in active_slots if not s.finished]

            # 3. Log what just happened so we can track the system's performance.
            current_live_tokens = sum((s.prompt_len + s.tokens_produced) for s in active_slots)
            self._log_event({
                "active_requests": [s.request_id for s in active_slots],
                "tokens_processed": len(active_slots) + len(freed_this_step), # include those processed this step
                "slots_freed": freed_this_step,
                "requests_admitted": admitted_this_step,
                "current_total_live_tokens": current_live_tokens
            })
            
            # Log fragmentation if needed (once per second)
            self.kv_manager.log_fragmentation_if_needed(active_slots, self._log_event)
