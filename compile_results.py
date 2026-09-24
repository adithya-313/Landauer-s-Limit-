import json
import os

def load_vllm(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r") as f:
        return json.load(f)

def summarize_vllm(data):
    if not data:
        return {}
    
    ttfts = [d["ttft"] for d in data[1:]] # skip first
    avg_ttft = sum(ttfts)/len(ttfts) if ttfts else 0
    return {
        "hit_rate": None,
        "avg_saved_per_hit": None,
        "avg_ttft_hits": None,
        "avg_ttft_misses": None,
        "avg_ttft_all": avg_ttft,
        "raw": data
    }

def main():
    # Load BatchEngine results
    with open("results.json", "r") as f:
        out = json.load(f)
        
    # Load vLLM results
    vllm_disabled = load_vllm("vllm_results_200_False.json")
    vllm_20 = load_vllm("vllm_results_20_True.json")
    vllm_200 = load_vllm("vllm_results_200_True.json")
    
    # Add to output
    out["vllm_cache_disabled"] = summarize_vllm(vllm_disabled)
    out["vllm_cache_20"] = summarize_vllm(vllm_20)
    out["vllm_cache_200"] = summarize_vllm(vllm_200)
    
    # Compile markdown table
    md = "# Benchmark Results: BatchEngine vs vLLM\n\n"
    md += "| Engine | Condition | Hit Rate | Avg Tokens Saved | Avg TTFT (Hits) | Avg TTFT (Misses) | Avg TTFT (All, unknown hit/miss) |\n"
    md += "|--------|-----------|----------|------------------|-----------------|-------------------|----------------------------------|\n"
    
    def add_row(engine, condition, data, is_vllm=False):
        if not data:
            return "| " + engine + " | " + condition + " | N/A | N/A | N/A | N/A | N/A |\n"
            
        if is_vllm:
            avg_ttft = data["avg_ttft_all"]
            return f"| {engine} | {condition} | N/A | N/A | N/A | N/A | {avg_ttft:.4f}s |\n"
        else:
            hit_rate = data["hit_rate"]
            avg_saved = data["avg_saved_per_hit"]
            ttft_hits = data["avg_ttft_hits"]
            ttft_misses = data["avg_ttft_misses"]
            return f"| {engine} | {condition} | {hit_rate:.0%} | {avg_saved:.1f} | {ttft_hits:.4f}s | {ttft_misses:.4f}s | N/A |\n"

    md += add_row("BatchEngine", "Cache Disabled (200-token)", out["cache_disabled"])
    md += add_row("BatchEngine", "Cache Enabled (20-token)", out["cache_20"])
    md += add_row("BatchEngine", "Cache Enabled (200-token)", out["cache_200"])
    
    md += add_row("vLLM", "Cache Disabled (200-token)", out["vllm_cache_disabled"], True)
    md += add_row("vLLM", "Cache Enabled (20-token)", out["vllm_cache_20"], True)
    md += add_row("vLLM", "Cache Enabled (200-token)", out["vllm_cache_200"], True)
    
    md += "\n> **LIMITATION**: vLLM's AsyncLLMEngine does not expose an internal cache hit/miss confirmation, so the 'cache disabled' vs 'cache enabled' comparison relies on trusting the enable_prefix_caching flag at face value rather than independent verification.\n"
    md += "> **NOTE**: The loop-detection heuristic (identifying low-entropy trailing tokens) has a known minor false-positive mode (approx. 8%) on certain tokenizer chunks (e.g., whitespace/punctuation streams). This is worth revisiting if this benchmark harness is reused for precise failure rate tracking.\n"
    
    with open("dashboard2_results.md", "w") as f:
        f.write(md)
        
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)
        
    print("Generated dashboard2_results.md and updated results.json")

if __name__ == "__main__":
    main()
