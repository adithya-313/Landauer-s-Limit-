# Phase 5 / 8 Quantization Matrix Specification

## 1. Context

Operating under strict hardware constraints (a single laptop GPU with 6GB VRAM, such as an RTX 4050, and zero cloud allocation), maximizing throughput while retaining model quality requires evaluating multiple precision and quantization strategies. Running full-precision or half-precision (FP16) models quickly saturates limited GPU memory, leading to Out-Of-Memory (OOM) failures or severe swapping overhead. Quantization reduces per-parameter memory footprint (from 16 bits down to 8 or 4 bits), enabling local inference within restricted VRAM boundaries. To support Phase 8's drift evaluation dashboard, this project establishes a comparative matrix to benchmark target model precisions across local inference engines.

## 2. Quantization Matrix

| Precision | Engine | Expected VRAM Footprint | Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **FP16** | vLLM / PyTorch | ~14.5 GB [ESTIMATE] (7B model) | Blocked (Exceeds capacity) | Standard unquantized baseline; exceeds 6GB VRAM limit. |
| **INT8** | vLLM | ~7.5 GB [ESTIMATE] (7B model) | Planned | 8-bit quantization via bitsandbytes/vLLM; marginal fit on 6GB. |
| **AWQ (4-bit)** | vLLM | ~4.2 GB [ESTIMATE] (7B model) | Planned | Activation-aware Weight Quantization; optimal low-latency target on vLLM. |
| **GGUF (Q4_K_M)** | Ollama / llama.cpp | ~4.0 GB [ESTIMATE] (7B model) | Planned | Medium 4-bit quantization with CPU offloading fallback; primary local tier. |
| **GPTQ (4-bit)** | vLLM | ~4.5 GB [ESTIMATE] (7B model) | Nice-to-have | Secondary 4-bit format; optional fallback if AWQ weights are unavailable. |

## 3. FP16 Baseline Feasibility & Fallback Plan

* **Feasibility:** An unquantized FP16 baseline for 7B parameter LLMs requires approximately 14.5 GB [ESTIMATE] of VRAM (14 GB for weights plus ~0.5 GB [ESTIMATE] minimum KV cache allocation). On local 6GB VRAM hardware, loading a 7B FP16 model directly on GPU memory is **not feasible** [ESTIMATE] and will trigger an OOM exception.
* **Phase 8 Fallback Strategy:** Because an FP16 baseline cannot fit in 6GB VRAM, Phase 8's quality and drift evaluation dashboard will compare quantized engines (e.g., AWQ vs. GGUF Q4_K_M vs. INT8) against each other directly, using relative performance and accuracy metrics rather than requiring an absolute FP16 baseline.

## 4. Known Implementation & Infrastructure Gaps

1. **Missing Container & Launch Configuration:** The codebase currently lacks Docker container configurations (`Dockerfile`, `docker-compose.yml`) or startup flags (`--gpu-memory-utilization`, `--max-model-len`, `--quantization awq`) necessary to launch a real, memory-constrained vLLM or llama.cpp instance.
2. **Execution Scope:** This document defines the target quantization matrix and memory bounds. Physical deployment, containerization, and empirical latency benchmarking across these quantized formats are scheduled as follow-up implementation work.
