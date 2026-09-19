import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def run_test():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("Loading model in 8-bit...")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    print(f"Model device: {model.device}")
    
    vram_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    vram_res = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"VRAM Allocated: {vram_alloc:.2f} MB")
    print(f"VRAM Reserved: {vram_res:.2f} MB")
    
    print("Generating...")
    inputs = tokenizer("Hello, how are you today?", return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=20)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Output: {text}")
    
if __name__ == "__main__":
    run_test()
