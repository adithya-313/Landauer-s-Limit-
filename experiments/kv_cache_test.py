import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

# Clean up any cached memory before we start
gc.collect()
torch.cuda.empty_cache()

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

print("--- STEP 1: LOAD MODEL ---")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

print(f"\n--- STEP 2: VRAM BASELINE ---")
print(f"torch.cuda.memory_allocated() after load: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

prompt = "The capital of France is"
print(f"\n--- STEP 3: TOKENIZE PROMPT ---")
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
print(f"Input tokens: {inputs['input_ids']}")

print("\n--- STEP 4: PREFILL (FIRST FORWARD PASS) ---")
try:
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    past_key_values = outputs.past_key_values
    print("DIR of past_key_values:")
    print(dir(past_key_values))
    
    print(f"\nCache length (layers): {len(past_key_values.layers)}")
    print(f"Layer 0 key tensor shape: {past_key_values.layers[0].keys.shape}")
    print(f"Layer 0 value tensor shape: {past_key_values.layers[0].values.shape}")

    print("\n--- LOGITS INSPECTION (PREFILL) ---")
    print(f"Raw logits shape: {outputs.logits.shape}")
    vocab_size = outputs.logits.shape[-1]
    print(f"Vocab size: {vocab_size}")
    
    # Extract the single token logits for printing
    print_logits = outputs.logits[0, -1, :]
    
    # Original logic for the forward pass
    next_token_logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)
    
    # Highest scoring token
    max_logit_value = print_logits[next_token[0]].item()
    print(f"Predicted next token id: {next_token[0].item()}")
    print(f"Predicted next token text: {ascii(tokenizer.decode(next_token[0]))}")
    print(f"Raw logit value of highest-scoring token: {max_logit_value:.4f}")
    
    # Top 10 tokens
    probs = torch.softmax(print_logits, dim=-1)
    top10_probs, top10_indices = torch.topk(probs, 10)
    print("Top 10 candidate tokens:")
    for prob, idx in zip(top10_probs, top10_indices):
        token_text = tokenizer.decode(idx.item())
        print(f"  - Token: {ascii(token_text)} | Prob: {prob.item() * 100:.2f}%")
        
except Exception as e:
    print(f"Error in step 4: {e}")
    exit(1)

print("\n--- STEP 5: MANUAL DECODE (SECOND FORWARD PASS) ---")
# pass only the new token
new_input = {
    "input_ids": next_token.unsqueeze(0), 
    "attention_mask": torch.cat([inputs["attention_mask"], torch.ones((1, 1), device="cuda")], dim=-1)
}

try:
    with torch.no_grad():
        outputs2 = model(**new_input, past_key_values=past_key_values, use_cache=True)
    print("Second forward pass succeeded.")
    
    past_key_values2 = outputs2.past_key_values
    print(f"\nNEW Layer 0 key tensor shape: {past_key_values2.layers[0].keys.shape}")
    print(f"NEW Layer 0 value tensor shape: {past_key_values2.layers[0].values.shape}")

    print("\n--- LOGITS INSPECTION (MANUAL DECODE) ---")
    print_logits2 = outputs2.logits[0, -1, :]
    
    next_token_logits2 = outputs2.logits[:, -1, :]
    next_token2 = torch.argmax(next_token_logits2, dim=-1)
    
    max_logit_value2 = print_logits2[next_token2[0]].item()
    print(f"Predicted next token id: {next_token2[0].item()}")
    print(f"Predicted next token text: {ascii(tokenizer.decode(next_token2[0]))}")
    print(f"Raw logit value of highest-scoring token: {max_logit_value2:.4f}")
    
    probs2 = torch.softmax(print_logits2, dim=-1)
    top10_probs2, top10_indices2 = torch.topk(probs2, 10)
    print("Top 10 candidate tokens:")
    for prob, idx in zip(top10_probs2, top10_indices2):
        token_text = tokenizer.decode(idx.item())
        print(f"  - Token: {ascii(token_text)} | Prob: {prob.item() * 100:.2f}%")

except Exception as e:
    print(f"Error in second forward pass: {e}")
    exit(1)

print(f"\n--- STEP 6: VRAM AFTER TWO STEPS ---")
print(f"torch.cuda.memory_allocated() after two steps: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

print("\n--- STEP 7: LOOP 3 MORE TIMES ---")
full_text = prompt + tokenizer.decode(next_token) + tokenizer.decode(next_token2)
current_token = next_token2
current_past = past_key_values2
current_mask = new_input["attention_mask"]

try:
    for i in range(3):
        current_mask = torch.cat([current_mask, torch.ones((1, 1), device="cuda")], dim=-1)
        step_input = {"input_ids": current_token.unsqueeze(0), "attention_mask": current_mask}
        with torch.no_grad():
            outputs_loop = model(**step_input, past_key_values=current_past, use_cache=True)
        
        current_past = outputs_loop.past_key_values
        next_token_logits_loop = outputs_loop.logits[:, -1, :]
        current_token = torch.argmax(next_token_logits_loop, dim=-1)
        full_text += tokenizer.decode(current_token)
        print(f"Loop {i+1} predicted token: {ascii(tokenizer.decode(current_token))}")
except Exception as e:
    print(f"Error in loop: {e}")

print(f"\nFull generated text: {ascii(full_text)}")
