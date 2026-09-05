# Landauer's Limit — Project Handoff

## 1. PROJECT INCEPTION & LEGACY MILESTONES

- **Dataset Pipeline (Phase 1):** `prepare_dataset.py` — HuggingFace `Aeala/ShareGPT_Vicuna_unfiltered` streamed (5K raw), filtered to 200 validated prompt/response pairs via `is_valid_conversation()` (≥2 turns, non-empty). Stratified sampling across 5 categories (general/coding/reasoning/creative_writing/knowledge) with keyword-based `classify_conversation()`. 20 samples tagged `drift_eval=True` for future evaluation. Output schema: `{id, turns[{role, content}], category, drift_eval}` persisted as `test_dataset.json`. Validated by `validate.py` (count, duplicates, empty-prompts, token-length histogram, multi-turn ratio).

- **Benchmark Micro-Lab (Phase 3b):** `micro_lab.py` — Benchmarked all-MiniLM-L6-v2 across 3 CPU backends: PyTorch Eager (p95=75.13ms), PyTorch Compiled (skipped — Windows), ONNX Runtime CPU (p95=65.32ms). Winner: ONNX. Derived production timeout = 65.32ms × 1.5 = **97.98ms (0.098s)**. Output: `runtime_bench.md` with latency tables, ASCII histograms, timeout derivation. ONNX model exported via `torch.onnx.export` with dynamic axes → `all-MiniLM-L6-v2.onnx`.

- **Git History (single `main` branch):** 6 commits spanning dataset → benchmark → gateway → guardrail → cache. No parallel branches.

## 2. UNIFIED QOS GATEWAY (Phases 1-4)

### Architecture Overview
Single FastAPI app (`gateway.py`) exposing OpenAI-compatible `POST /v1/chat/completions`. Three-layer middleware stack: (1) Zero-copy payload rejection, (2) CPU guardrail with fail-open, (3) deterministic semantic cache.

### Phase 1 — Dataset Curation
- `prepare_dataset.py`: ShareGPT → 200 clean pairs stratified by 5 categories, 20 drift-eval flagged
- Schema: `{id: str, turns: [{role: "user"|"assistant", content: str}], category: str, drift_eval: bool}`

### Phase 2 — FastAPI Gateway
- **Zero-copy Content-Length middleware:** `check_payload_size()` inspects `Content-Length` header before body read. Missing header → **411 Length Required**. Exceeds 64KB → **413 Payload Too Large** (bytes rejected before any buffer allocation). GET/DELETE/HEAD/OPTIONS bypassed.
- **OpenAI-compatible schema:** `ChatCompletionRequest` (model, messages[role,content], optional temperature/max_tokens), `ChatCompletionResponse` (id, object, created, model, choices[{index, message, finish_reason}], usage).
- Mock response: `"This is a mock response from the Landauer's Limit gateway. You said: \"{prompt}\""`.

### Phase 3 — Sub-100ms ML Guardrail
- **`guardrail.py`:** Module-level singleton ONNX session (`onnxruntime.InferenceSession`, CPUExecutionProvider) + HuggingFace AutoTokenizer (shared with Phase 4 cache). `embed_text()` pipeline: tokenize → ONNX forward → mean pooling → L2 normalisation → 384-dim float32 vector.
- **FAISS HNSW index:** `IndexHNSWFlat(384, 32, METRIC_INNER_PRODUCT)` seeded with 68 injection seed phrases. `efConstruction=64`.
- **`CpuGuardrail.check(prompt)`:** Embeds prompt → FAISS search k=1 → cosine similarity clamped [0,1] → threshold=0.70 → returns `(is_malicious, score)`.
- **Hard timeout:** 97.98ms via `asyncio.wait_for(asyncio.to_thread(...))`. TimeoutError or generic Exception → sets `degraded=True` flag, **proceeds** (fail-open QoS). Does NOT return 400.
- **Malicious halt:** `is_malicious==True` → HTTP 400 `{"error":"Prompt Injection Detected","similarity_score":float}`.

### Phase 4 — Deterministic Semantic Cache (Dual-Lock)
- **`semantic_cache.py`:** Two-factor cache verification.
- **Factor 1 — Lock string (deterministic):** `RuleExtractor` uses flashtext `KeywordProcessor` (7 seed entities: python/java/docker/kubernetes/react/node/postgres) + regex rules (version `\d+\.\d+(\.\d+)?`, OS patterns, action verb extraction). Lock format: `"{entity}|{version}|{os}|{action}"`. History scanning for coreference resolution (pronouns like "it" → scan all prior user messages; only cache if exactly 1 entity found — zero or multiple = abort).
- **Factor 2 — FAISS vector:** `IndexIDMap(IndexFlatIP(384))` for exact inner-product search, top-k=5. Cosine threshold=0.88.
- **Hit condition:** Vector similarity ≥ 0.88 **AND** saved lock string == current lock string (exact). Both must pass.
- **LRU eviction:** `OrderedDict` store, max 10,000 entries. `_evict_one()` pops oldest (first-inserted) item + removes from FAISS (tracked in separate deletion set).
- **Fail-safe:** `generate_lock()` returns `None` on unknown entity or ambiguous coreference → cache **bypassed entirely** (returns miss, no insert). All FAISS/embedding exceptions caught → miss (fail-open).
- **Global singleton:** `semantic_cache = DeterministicSemanticCache()` at module level, reuses guardrail's `embed_text`.

### POST Route Execution Flow (current)
```
1. Parse body → extract messages_dicts[{role, content}], last_user_message
2. guardrail.check(prompt) via asyncio.wait_for(to_thread, timeout=0.098)
   2a. Malicious → 400 (halt)
   2b. TimeoutError/Exception → degraded=True (proceed, fail-open)
3. cache.check_cache(messages_dicts) → Optional[Dict] (full response dict or None)
4. Hit → 200 + X-Cache: HIT [+ X-Guardrail-Degraded: true if degraded]
5. Miss → build ChatCompletionResponse → cache.insert() → 200 + X-Cache: MISS [+ X-Guardrail-Degraded]
```

### Test Suite
| File | Type | Tests | Coverage |
|------|------|-------|----------|
| `test_phase2.py` | Sequential script | 3 | normal 200, oversized 413, missing Content-Length 411 |
| `test_phase3.py` | Sequential script | 3 | safety (400 on DAN injection), concurrency (20×), fail-open timeout (1ms → degraded) |
| `test_phase4.py` | pytest | 3 | version collision (3.14≠3.15), coreference defense (docker≠java), unknown entity bypass (Rust) |

### Data Flow Diagram
```
Client → POST /v1/chat/completions
  → [Middleware] check_payload_size() (Content-Length ≤ 64KB)
  → [Route] Extract messages_dicts, last_user_message
  → asyncio.wait_for(to_thread(guardrail.check))
    ├─ malicious  ───→ 400 (halt)
    └─ pass/timeout ──→ degraded=flag_if_timeout
  → await cache.check_cache(messages_dicts)
    ├─ hit  ─────────→ 200 + cached response + X-Cache: HIT [+ degraded]
    └─ miss ─────────→ build mock → cache.insert() → 200 + X-Cache: MISS [+ degraded]
```

## 3. FRONTEND / CLIENT TOPOLOGY

- **No frontend exists.** The project is backend-only (API gateway + ML inference engine).
- The client interface is OpenAI-compatible HTTP: any HTTP client (curl, requests, OpenAI Python SDK) sends `POST /v1/chat/completions` with `{model, messages[{role, content}]}` and receives `{id, object, created, model, choices[{index, message, finish_reason}], usage}`.
- No browser UI, no WebSocket, no streaming. Phase 5 may introduce SSE streaming when real LLM engines are plugged in.

## 4. CURRENT STATE & RESUMPTION POINT

- **Active branch:** `main` (single branch, no forks).
- **Latest commit:** `b697529 feat(phase4): implement semantic cache with FAISS IndexIDMap, LRU eviction, and gateway integration`
- **Working tree:** clean (no uncommitted changes outside this handoff document).
- **Execution flow at this exact second:** `gateway.py` line 292 — `POST /v1/chat/completions` → guardrail (fail-open) → semantic cache (dual-lock) → mock response. Full test suite passes (9/9 tests across all 3 test files).
- **Phase 5 (immediate next step): Multi-Engine Router.**
  - Replace the mock response in `gateway.py` with a routing layer that dispatches to real LLM backends (e.g., local ONNX model, remote OpenAI API, Anthropic API) based on model name, request priority, or cost budget.
  - Design a `Router` class with pluggable backends, retry logic, circuit breaker, and optional SSE streaming.
  - Integrate with the semantic cache: on cache miss → route to selected engine → cache the real response.
  - Define engine health-check and fallback chain (primary → secondary → tertiary).
