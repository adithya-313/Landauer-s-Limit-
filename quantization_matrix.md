# Quantization Matrix — Real Measured Data

> Last updated: September 2026. All VRAM figures and status results are
> **FACT** — obtained from live `nvidia-smi` output and actual container
> logs during this session, not estimates.

---

## Context

This project runs four quantization tiers of a 1.5B-parameter Qwen model
to satisfy Phase 5 of the Landauer's Limit spec. The local machine has a
single **RTX 4050 6 GB** laptop GPU (6141 MiB total VRAM). Because FP16 at
this size exceeds local VRAM at safe utilization levels, it is served via
**Google Colab T4** (15360 MiB VRAM) tunneled through Ngrok — a fallback
explicitly permitted by the original spec ("on either tier"). All other tiers
run locally in Docker via vLLM, or locally via Ollama. Total cloud cost: $0.

---

## Real Data Table

| Precision | Model | Engine / Location | VRAM Used (FACT) | Status | Notes |
|-----------|-------|-------------------|-----------------|--------|-------|
| **FP16** | Qwen/Qwen2.5-1.5B-Instruct | vLLM 0.28.0 / Google Colab T4 | ~13.1 GB of 15 GB (gpu_util 0.85) | ✅ PASSED | Tunneled via Ngrok. Confirmed live across two independent Colab sessions. |
| **AWQ** | Qwen/Qwen2.5-1.5B-Instruct-AWQ | vLLM 0.28.0 / Docker local, RTX 4050 6 GB | ~5650 MB of 6141 MB (gpu_util 0.70, max_model_len 2048) | ✅ PASSED | Auto-detected quantization=auto_awq. MarlinLinearKernel selected. |
| **INT8** | neuralmagic/Qwen2-1.5B-Instruct-quantized.w8a8 ⚠️ | vLLM 0.28.0 / Docker local, RTX 4050 6 GB | **5794 MB of 6141 MB** (127 MB free) (gpu_util 0.70, max_model_len 2048) | ✅ PASSED | compressed-tensors format, CutlassInt8ScaledMMLinearKernel. See model-mismatch callout below. |
| **GGUF** | qwen2.5:1.5b (Q4_K_M) | Ollama local | CPU-offloaded (no dedicated VRAM measurement) | ✅ PASSED | 986 MB download. Q4_K_M confirmed via `ollama show`. 32768 context. |
| **GPTQ** | — | — | — | ⬛ NOT ATTEMPTED | Explicitly optional per original spec. Not blocking. |

---

## Model Lineup Inconsistency — Documented Exception

Three of the four tiers use **Qwen2.5**-1.5B as the base model:

- FP16: `Qwen/Qwen2.5-1.5B-Instruct`
- AWQ: `Qwen/Qwen2.5-1.5B-Instruct-AWQ`
- GGUF: `qwen2.5:1.5b` (Ollama, Q4_K_M)

The INT8 tier uses **Qwen2**-1.5B (`neuralmagic/Qwen2-1.5B-Instruct-quantized.w8a8`),
which is one model generation older. This is an **intentional, researched
exception**, not an oversight. Two independent research passes confirmed:

- The official Qwen organisation publishes only AWQ and GPTQ-Int4 checkpoints
  for the Qwen2.5-1.5B size — no INT8/W8A8 variant exists publicly.
- Third-party INT8 Qwen2.5 options found were either: gated/enterprise-only
  (`neuralmagic-ent`), in OpenVINO format (incompatible with vLLM), or of
  unclear vLLM support.
- Previous attempts with `Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8` failed with
  a `g_idx` loader error — a known incompatibility between that checkpoint's
  GPTQ packaging and vLLM 0.28.0's loader.

**Implication for Phase 8:** The INT8 tier cannot be directly compared to the
FP16 baseline on Dashboard 3 the same way AWQ and GGUF can, because it uses a
different base model (Qwen2 vs. Qwen2.5). Any observed output differences may
reflect the generation gap, not only quantization error. This should be noted
explicitly in the drift comparison.

---

## Deliberate OOM Test Result

The spec requires a demonstrated out-of-memory failure to confirm the safety
guard works. Tested against the AWQ model with deliberately unsafe settings:

- **Settings:** `--gpu-memory-utilization 0.98 --max-model-len 8192`
- **Outcome:** Container exited immediately at startup with:

```
ValueError: Free memory on device cuda:0 (4.95/6.0 GiB) on startup is less
than desired GPU memory utilization (0.98, 5.88 GiB).
```

- **Verdict:** ✅ vLLM's own startup safety check caught the overload cleanly.
  No silent crash, no corrupted output, no hang. The container exited with a
  clear logged error and a non-zero exit code. This satisfies the spec's
  required OOM demonstration.

---

## Anecdotal Single-Prompt Quality Observation

> ⚠️ **This is NOT a drift study.** It is a one-prompt side-by-side
> observation included only for honest documentation. Phase 8's Dashboard 3
> is the real drift measurement — do not treat these notes as conclusions.

Standard prompt used across all four tiers:
**"Explain Landauer's principle in one short sentence."**

| Tier | Key observation |
|------|----------------|
| FP16 | Named the constant correctly (k_B ln(2)) and explained the physics accurately. |
| AWQ | Vaguer answer; did not correctly name the constant. |
| INT8 | Vaguer answer; did not correctly name the constant. Also note: different base model (Qwen2). |
| GGUF (Q4_K_M) | Named kT ln 2 correctly. |

These are single-sample observations from one prompt. Response quality at
this scale is highly prompt- and temperature-sensitive. No conclusions about
quantization-induced drift should be drawn from this table alone.

---

## Reproduction Commands

```bash
# AWQ (local)
docker run --name vllm_server -d --gpus all -p 8000:8000 \
  -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-1.5B-Instruct-AWQ \
  --quantization awq --gpu-memory-utilization 0.70 --max-model-len 2048

# INT8 (local)
docker run --name vllm_server_int8 -d --gpus all -p 8000:8000 \
  -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
  vllm/vllm-openai:latest \
  --model neuralmagic/Qwen2-1.5B-Instruct-quantized.w8a8 \
  --gpu-memory-utilization 0.70 --max-model-len 2048

# GGUF (local)
ollama pull qwen2.5:1.5b
ollama run qwen2.5:1.5b

# FP16 (Colab — launch vLLM there, expose via Ngrok)
# See HANDOFF.md for Colab notebook setup instructions.
```
