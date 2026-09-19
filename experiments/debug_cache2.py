import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs, use_cache=True)

past_key_values = outputs.past_key_values
print("len of layers:", len(past_key_values.layers))
print("DIR of layers[0]:", dir(past_key_values.layers[0]))
