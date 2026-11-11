"""
drift_eval.py
=============
Collects the data behind Dashboard 3 (Quantization Quality Drift).

For the 20 drift_eval prompts in test_dataset.json (first user turn only) it
gets an answer from an unquantized reference (Qwen/Qwen2.5-1.5B-Instruct, fp16,
transformers) and from each quantized engine, then scores every pair with the
cosine similarity of all-MiniLM-L6-v2 embeddings (guardrail.embed_text).

Decoding is pure greedy for every engine: temperature 0, no repetition penalty,
max 128 new tokens, identical system prompt (Qwen's default).

Usage (PowerShell, Python 3.12, from the repo root):
    python drift_eval.py reference                      # writes drift_eval_reference.jsonl
    python drift_eval.py candidate --engine llamacpp_local
    python drift_eval.py candidate --engine custom_runtime
    python drift_eval.py candidate --engine vllm_local  # needs the vLLM Docker server, see below
    python drift_eval.py all                            # reference + every candidate, one process each

GPU note: run one stage at a time and unload other models first (the GPU has 6 GB).

vllm_local is not installed on Windows. Start a temporary container, run the stage, remove it:
    docker run --name vllm_drift_tmp -d --gpus all -p 8001:8000 -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
      vllm/vllm-openai:latest --model Qwen/Qwen2.5-1.5B-Instruct-AWQ --quantization awq \
      --gpu-memory-utilization 0.70 --max-model-len 2048

Output: drift_eval_results.jsonl, one flat record per (prompt_id, engine):
    prompt_id, prompt, engine, quantization, reference_output, candidate_output,
    cosine_similarity, timestamp
Re-running an engine replaces that engine's records. generate_report.py builds
Dashboard 3 from this file only and never calls a model.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "test_dataset.json"
REFERENCE_PATH = ROOT / "drift_eval_reference.jsonl"
RESULTS_PATH = ROOT / "drift_eval_results.jsonl"

REFERENCE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
MAX_TOKENS = 128

OLLAMA_URL = os.environ.get("LLAMACPP_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = "qwen2.5:1.5b"
VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8001/v1").rstrip("/")
VLLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct-AWQ"

CANDIDATES = {
    "llamacpp_local": "gguf_q4",
    "custom_runtime": "int4",
    "vllm_local": "awq",
}


class EngineUnavailable(Exception):
    """The engine could not start or answer; it is skipped, not scored."""


def log(msg):
    print(f"[drift_eval] {msg}", file=sys.stderr, flush=True)


def load_prompts():
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    prompts = []
    for item in data:
        if item.get("drift_eval") is True:
            first_user = next(t["content"] for t in item["turns"] if t["role"] == "user")
            prompts.append({"prompt_id": item["id"], "prompt": first_user})
    if len(prompts) != 20:
        raise SystemExit(f"expected 20 drift_eval prompts, found {len(prompts)}")
    return prompts


def chat_messages(prompt):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def http_json(url, payload=None, timeout=600):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Reference (fp16, transformers)
# ---------------------------------------------------------------------------

def run_reference():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = load_prompts()
    log(f"loading {REFERENCE_MODEL} in fp16 on the GPU")
    tok = AutoTokenizer.from_pretrained(REFERENCE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(REFERENCE_MODEL, dtype=torch.float16).to("cuda").eval()

    records = []
    for i, p in enumerate(prompts, start=1):
        inputs = tok.apply_chat_template(
            chat_messages(p["prompt"]), add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs, do_sample=False, max_new_tokens=MAX_TOKENS,
                repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        records.append({
            "prompt_id": p["prompt_id"], "prompt": p["prompt"],
            "reference_output": text, "reference_model": REFERENCE_MODEL,
            "timestamp": time.time(),
        })
        log(f"reference {i}/{len(prompts)} done ({len(text)} chars)")

    with open(REFERENCE_PATH, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"wrote {REFERENCE_PATH.name} ({len(records)} records)")


def load_reference(prompts):
    if not REFERENCE_PATH.exists():
        raise SystemExit(f"{REFERENCE_PATH.name} missing - run: python drift_eval.py reference")
    ref = {}
    for line in REFERENCE_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            ref[rec["prompt_id"]] = rec
    for p in prompts:
        if p["prompt_id"] not in ref or ref[p["prompt_id"]]["prompt"] != p["prompt"]:
            raise SystemExit(f"reference file out of date for prompt {p['prompt_id']} - rerun the reference stage")
    return ref


# ---------------------------------------------------------------------------
# Candidate engines
# ---------------------------------------------------------------------------

def generate_llamacpp(prompts):
    try:
        tags = http_json(f"{OLLAMA_URL}/api/tags", timeout=10)
    except (urllib.error.URLError, OSError) as exc:
        raise EngineUnavailable(f"Ollama not reachable at {OLLAMA_URL}: {exc}")
    if not any(m.get("name") == OLLAMA_MODEL for m in tags.get("models", [])):
        raise EngineUnavailable(f"Ollama has no model {OLLAMA_MODEL}")

    outputs = []
    try:
        for i, p in enumerate(prompts, start=1):
            resp = http_json(f"{OLLAMA_URL}/api/chat", {
                "model": OLLAMA_MODEL, "messages": chat_messages(p["prompt"]), "stream": False,
                "options": {"temperature": 0, "top_k": 1, "top_p": 1.0, "repeat_penalty": 1.0,
                            "seed": 0, "num_predict": MAX_TOKENS},
            })
            outputs.append(resp["message"]["content"])
            log(f"llamacpp_local {i}/{len(prompts)} done")
    except (urllib.error.URLError, OSError, KeyError) as exc:
        raise EngineUnavailable(f"Ollama request failed: {exc}")
    finally:
        try:
            http_json(f"{OLLAMA_URL}/api/generate", {"model": OLLAMA_MODEL, "keep_alive": 0}, timeout=30)
        except Exception:
            pass
    return outputs


def generate_vllm(prompts):
    try:
        http_json(f"{VLLM_URL}/models", timeout=10)
    except (urllib.error.URLError, OSError) as exc:
        raise EngineUnavailable(f"vLLM server not reachable at {VLLM_URL}: {exc}")

    outputs = []
    try:
        for i, p in enumerate(prompts, start=1):
            resp = http_json(f"{VLLM_URL}/chat/completions", {
                "model": VLLM_MODEL, "messages": chat_messages(p["prompt"]),
                "max_tokens": MAX_TOKENS, "temperature": 0, "top_p": 1.0, "top_k": -1,
                "repetition_penalty": 1.0, "seed": 0, "stream": False,
            })
            outputs.append(resp["choices"][0]["message"]["content"])
            log(f"vllm_local {i}/{len(prompts)} done")
    except (urllib.error.URLError, OSError, KeyError) as exc:
        raise EngineUnavailable(f"vLLM request failed: {exc}")
    return outputs


def generate_custom_runtime(prompts):
    # BatchEngine.__init__ truncates ./batch_events.jsonl (read by Dashboard 2B), so run from a scratch dir.
    sys.path.insert(0, str(ROOT))
    workdir = tempfile.mkdtemp(prefix="drift_eval_cwd_")
    os.chdir(workdir)
    engine = None
    try:
        try:
            from runtime_core.batch_engine import BatchEngine
            engine = BatchEngine()
            engine.start()
        except Exception as exc:
            raise EngineUnavailable(f"BatchEngine failed to start: {exc!r}")

        import queue as _queue
        outputs = []
        for i, p in enumerate(prompts, start=1):
            _, resp_q = engine.submit(p["prompt"], tier="free", max_tokens=MAX_TOKENS,
                                      messages=chat_messages(p["prompt"]))
            pieces = []
            while True:
                try:
                    msg = resp_q.get(timeout=600)
                except _queue.Empty:
                    raise EngineUnavailable(f"BatchEngine produced no output for prompt {p['prompt_id']}")
                if msg["type"] == "token":
                    pieces.append(msg["content"])
                elif msg["type"] == "done":
                    break
                elif msg["type"] == "error":
                    raise EngineUnavailable(f"BatchEngine error: {msg.get('error')}")
            outputs.append("".join(pieces))
            log(f"custom_runtime {i}/{len(prompts)} done")
        return outputs
    finally:
        if engine is not None:
            engine.stop()
        os.chdir(ROOT)
        shutil.rmtree(workdir, ignore_errors=True)


GENERATORS = {
    "llamacpp_local": generate_llamacpp,
    "custom_runtime": generate_custom_runtime,
    "vllm_local": generate_vllm,
}


def run_candidate(engine_name):
    prompts = load_prompts()
    ref = load_reference(prompts)
    try:
        outputs = GENERATORS[engine_name](prompts)
    except EngineUnavailable as exc:
        log(f"SKIPPED {engine_name}: {exc}")
        return 3

    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    sys.path.insert(0, str(ROOT))
    from guardrail import embed_text

    now = time.time()
    new_records = []
    for p, candidate in zip(prompts, outputs):
        reference_output = ref[p["prompt_id"]]["reference_output"]
        sim = float(embed_text(reference_output)[0] @ embed_text(candidate)[0])
        new_records.append({
            "prompt_id": p["prompt_id"], "prompt": p["prompt"],
            "engine": engine_name, "quantization": CANDIDATES[engine_name],
            "reference_output": reference_output, "candidate_output": candidate,
            "cosine_similarity": round(sim, 6), "timestamp": now,
        })

    kept = []
    if RESULTS_PATH.exists():
        for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("engine") != engine_name:
                kept.append(line)
    tmp_path = RESULTS_PATH.with_suffix(".jsonl.tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        for line in kept:
            fh.write(line + "\n")
        for r in new_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp_path.replace(RESULTS_PATH)

    sims = [r["cosine_similarity"] for r in new_records]
    log(f"{engine_name}: wrote {len(new_records)} records, mean cosine {sum(sims) / len(sims):.4f}, min {min(sims):.4f}")
    return 0


def run_all():
    me = [sys.executable, str(Path(__file__).resolve())]
    if subprocess.call(me + ["reference"]) != 0:
        raise SystemExit("reference stage failed")
    for name in CANDIDATES:
        code = subprocess.call(me + ["candidate", "--engine", name])
        log(f"{name}: exit code {code}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("reference")
    cand = sub.add_parser("candidate")
    cand.add_argument("--engine", required=True, choices=sorted(CANDIDATES))
    sub.add_parser("all")
    args = parser.parse_args()

    if args.stage == "reference":
        run_reference()
    elif args.stage == "candidate":
        sys.exit(run_candidate(args.engine))
    else:
        run_all()


if __name__ == "__main__":
    main()
