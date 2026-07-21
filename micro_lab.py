"""
micro_lab.py
============
PHASE 3b — Runtime Micro-Lab: Benchmark all-MiniLM-L6-v2 across three CPU backends.

This script benchmarks the sentence-transformer model "all-MiniLM-L6-v2" on three
different execution backends so we can pick the fastest one for our guardrail:

    1. PyTorch Eager        — the standard eager-mode forward pass.
    2. PyTorch Compiled     — torch.compile() graph capture (if the platform supports it).
    3. ONNX Runtime CPU     — a static-graph ONNX model served by onnxruntime.

We run 10 warmup iterations followed by 100 timed iterations per engine, then
report p50 / p95 / p99 latencies. The engine with the lowest p95 is declared the
winner, and we compute a production timeout as `winner_p95 * 1.5`.

Results are saved to `runtime_bench.md`.
"""

import json
import time
import statistics
import platform
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
import onnxruntime
from transformers import AutoTokenizer, AutoModel


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
WARMUP_RUNS: int = 10
TIMED_RUNS: int = 100
OUTPUT_FILE: str = "runtime_bench.md"
RANDOM_SEED: int = 42

# A short, medium, and longer prompt so we test realistic variability.
SAMPLE_TEXTS: List[str] = [
    "What is the capital of France?",
    (
        "Explain the difference between gradient descent and stochastic gradient "
        "descent in the context of training deep neural networks."
    ),
    (
        "The quick brown fox jumps over the lazy dog. This pangram contains every "
        "letter of the English alphabet at least once. It has been used for decades "
        "to test typewriters, computer keyboards, and fonts because of its diverse "
        "character set. Many typing tutors also use it for practice sessions."
    ),
    "Python",
]

# ONNX export path
ONNX_EXPORT_PATH: str = "all-MiniLM-L6-v2.onnx"


# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------

@dataclass
class LatencyStats:
    """Store latency statistics for one engine."""
    engine_name: str
    latencies_ms: List[float] = field(default_factory=list)
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0

    def compute(self):
        """Compute p50, p95, p99 from the collected latencies."""
        sorted_lats = sorted(self.latencies_ms)
        n = len(sorted_lats)
        self.p50_ms = sorted_lats[int(n * 0.50)]
        self.p95_ms = sorted_lats[int(n * 0.95)]
        self.p99_ms = sorted_lats[int(n * 0.99)]
        return self

    def mean_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0


@dataclass
class BenchmarkResult:
    """Aggregated result from benchmarking all engines."""
    stats: List[LatencyStats] = field(default_factory=list)
    winner: str = ""
    winner_p95_ms: float = 0.0
    derived_timeout_ms: float = 0.0


# ---------------------------------------------------------------------------
# ENGINE LOADERS
# ---------------------------------------------------------------------------

def load_pytorch_eager(model_name: str, device: str = "cpu") -> nn.Module:
    """
    Load the transformer model in standard PyTorch eager mode.

    We load the model directly from HuggingFace and set it to evaluation mode
    so that dropout and other training-only layers are turned off.
    """
    print(f"  Loading PyTorch eager model '{model_name}' ...")
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return model


def load_pytorch_compiled(
    model_name: str, device: str = "cpu"
) -> Optional[nn.Module]:
    """
    Load the model and wrap it with torch.compile().

    On Windows, torch.compile may fail because the Triton compiler is not
    available. We catch the error gracefully and return None so the benchmark
    can skip this engine on unsupported platforms.
    """
    print(f"  Loading PyTorch compiled model '{model_name}' ...")
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    try:
        # Attempt to compile. On Windows this will typically fall back to eager
        # or throw. We use the default backend ('inductor' on Linux, 'eager' if
        # unavailable).
        compiled = torch.compile(model, backend="eager" if platform.system() == "Windows" else "inductor")
        # Run a tiny forward pass to confirm compilation works.
        dummy = {
            "input_ids": torch.randint(0, 100, (1, 16)),
            "attention_mask": torch.ones((1, 16), dtype=torch.long),
        }
        with torch.no_grad():
            _ = compiled(**dummy)
        print(f"  torch.compile succeeded with backend: "
              f"{torch._dynamo.list_backends() if hasattr(torch._dynamo, 'list_backends') else 'default'}")
        return compiled
    except Exception as exc:
        print(f"  Warning: torch.compile not available on this platform "
              f"({exc}). Skipping compiled backend.")
        return None


class BertONNXWrapper(nn.Module):
    """
    A thin wrapper around the HuggingFace BertModel that exposes only the
    two inputs we need (input_ids, attention_mask). This avoids tracing
    conflicts with optional kwargs like `use_cache` that newer HF
    transformer versions inject during torch.onnx.export tracing.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        # Call with keyword-only arguments to avoid positional arg clashes.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        # Return the two standard Bert outputs: sequence of hidden states
        # and pooled representation.
        return outputs.last_hidden_state, outputs.pooler_output


def load_onnx_model(model_name: str, onnx_path: str) -> Optional[onnxruntime.InferenceSession]:
    """
    Export the model to ONNX and create an onnxruntime InferenceSession.

    Steps:
    1. Wrap the HuggingFace model so only input_ids + attention_mask are traced.
    2. Export to ONNX format using torch.onnx.export with dynamic axes.
    3. Create an ONNX Runtime session with the CPU execution provider.
    """
    print(f"  Exporting model '{model_name}' to ONNX ...")

    # Load the base model and wrap it.
    base_model = AutoModel.from_pretrained(model_name)
    base_model.eval()
    model = BertONNXWrapper(base_model)

    # Create dummy input tensors for the export trace.
    dummy_input_ids = torch.randint(0, 100, (1, 64), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, 64), dtype=torch.long)

    # Export the model with dynamic axes so that the ONNX model accepts
    # variable-length inputs (batch_size and sequence_length).
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
                "pooler_output": {0: "batch_size"},
            },
            opset_version=14,
        )

    print(f"  ONNX model saved to '{onnx_path}'")

    # Create the ONNX Runtime session on CPU.
    so = onnxruntime.SessionOptions()
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(
        onnx_path,
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    print(f"  ONNX Runtime session created (provider: {session.get_providers()[0]})")
    return session


# ---------------------------------------------------------------------------
# TOKENISER HELPER
# ---------------------------------------------------------------------------

def tokenize_inputs(
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: str = "cpu",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, np.ndarray]]:
    """
    Tokenise the sample texts once, returning both PyTorch tensors and NumPy
    arrays so each backend can directly use its preferred format.
    """
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    pt_inputs = {k: v.to(device) for k, v in encoded.items()}
    np_inputs = {k: v.cpu().numpy() for k, v in encoded.items()}
    return pt_inputs, np_inputs


# ---------------------------------------------------------------------------
# BENCHMARKING ENGINE
# ---------------------------------------------------------------------------

def run_benchmark(
    engine_name: str,
    forward_fn,
    pt_inputs: Dict[str, torch.Tensor],
    np_inputs: Dict[str, np.ndarray],
    warmup: int,
    runs: int,
) -> LatencyStats:
    """
    Run warmup + timed iterations for a given engine.

    Parameters
    ----------
    engine_name : str
        Human-readable name for logging.
    forward_fn : callable
        A function that accepts inputs and returns outputs. For PyTorch engines
        this is a model call; for ONNX Runtime it wraps session.run().
    pt_inputs : dict
        PyTorch tensor inputs (used by PyTorch engines).
    np_inputs : dict
        NumPy array inputs (used by ONNX Runtime).
    warmup : int
        Number of warmup iterations (not timed).
    runs : int
        Number of timed iterations.

    Returns
    -------
    LatencyStats
        Populated latency statistics.
    """
    stats = LatencyStats(engine_name=engine_name)

    # --- Warmup ---
    print(f"  Warmup ({warmup} iterations) ...")
    for _ in range(warmup):
        forward_fn(pt_inputs, np_inputs)

    # --- Timed runs ---
    print(f"  Benchmarking ({runs} iterations) ...")
    latencies = []
    for i in range(runs):
        start = time.perf_counter()
        forward_fn(pt_inputs, np_inputs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed_ms)

    stats.latencies_ms = latencies
    stats.compute()
    print(f"  Done. p50={stats.p50_ms:.2f} ms  p95={stats.p95_ms:.2f} ms  "
          f"p99={stats.p99_ms:.2f} ms")
    return stats


# ---------------------------------------------------------------------------
# FORWARD FUNCTIONS (one per engine)
# ---------------------------------------------------------------------------

def make_pytorch_forward(model: nn.Module, device: str = "cpu"):
    """Return a forward function that runs the PyTorch model with torch.no_grad()."""
    def forward(pt_inputs: Dict[str, torch.Tensor], _np_inputs: Dict[str, np.ndarray]):
        with torch.no_grad():
            _ = model(**pt_inputs)
    return forward


def make_onnx_forward(session: onnxruntime.InferenceSession):
    """Return a forward function that runs ONNX Runtime inference."""
    def forward(_pt_inputs: Dict[str, torch.Tensor], np_inputs: Dict[str, np.ndarray]):
        _ = session.run(
            output_names=["last_hidden_state", "pooler_output"],
            input_feed={
                "input_ids": np_inputs["input_ids"],
                "attention_mask": np_inputs["attention_mask"],
            },
        )
    return forward


# ---------------------------------------------------------------------------
# REPORT GENERATION
# ---------------------------------------------------------------------------

def generate_markdown_report(result: BenchmarkResult) -> str:
    """
    Build a formatted Markdown file with the benchmark comparison table,
    the winner announcement, and the derived timeout calculation.
    """
    lines = []
    lines.append("# Runtime Benchmark Report — all-MiniLM-L6-v2")
    lines.append("")
    lines.append(f"- **Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Platform:** {platform.system()} {platform.release()} "
                 f"({platform.processor() or 'N/A'})")
    lines.append(f"- **PyTorch Version:** {torch.__version__}")
    lines.append(f"- **ONNX Runtime Version:** {onnxruntime.__version__}")
    lines.append(f"- **Warmup Iterations:** {WARMUP_RUNS}")
    lines.append(f"- **Timed Iterations:** {TIMED_RUNS}")
    lines.append(f"- **Sample Texts:** {len(SAMPLE_TEXTS)}")
    lines.append("")

    # Comparison table
    lines.append("## Latency Comparison (ms)")
    lines.append("")
    lines.append("| Engine | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) |")
    lines.append("|--------|----------|----------|----------|-----------|")
    for stat in result.stats:
        lines.append(
            f"| {stat.engine_name} | "
            f"{stat.p50_ms:.2f} | "
            f"{stat.p95_ms:.2f} | "
            f"{stat.p99_ms:.2f} | "
            f"{stat.mean_ms():.2f} |"
        )
    lines.append("")

    # P95 distribution histogram (ASCII)
    lines.append("## p95 Latency Distribution")
    lines.append("")
    lines.append("```")
    for stat in result.stats:
        bar_len = max(1, int(stat.p95_ms / 5))
        bar = "#" * bar_len
        lines.append(f"  {stat.engine_name:30s} {stat.p95_ms:8.2f} ms  {bar}")
    lines.append("```")
    lines.append("")

    # Winner and derived timeout
    lines.append("## Winner & Derived Timeout")
    lines.append("")
    lines.append(f"The **{result.winner}** engine has the lowest p95 latency "
                 f"({result.winner_p95_ms:.2f} ms).")
    lines.append("")
    lines.append("### Timeout Calculation")
    lines.append("")
    lines.append("```")
    lines.append(f"  base_p95      = {result.winner_p95_ms:.2f} ms")
    lines.append(f"  safety_factor = 1.5")
    lines.append(f"  {'-' * 29}")
    lines.append(f"  derived_timeout = {result.winner_p95_ms:.2f} × 1.5")
    lines.append(f"                   = {result.derived_timeout_ms:.2f} ms")
    lines.append(f"                   = {result.derived_timeout_ms / 1000:.3f} s")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("RUNTIME MICRO-LAB — all-MiniLM-L6-v2 Benchmark")
    print("=" * 60)
    print()
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"PyTorch:  {torch.__version__}")
    print(f"ONNX:     {onnxruntime.__version__}")
    print()

    # Set seeds for reproducibility.
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = "cpu"

    # -----------------------------------------------------------------------
    # Load tokenizer and tokenise sample texts once.
    # -----------------------------------------------------------------------
    print("[1/4] Loading tokenizer and preparing inputs ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pt_inputs, np_inputs = tokenize_inputs(tokenizer, SAMPLE_TEXTS, device=device)
    print(f"      Input shape: {pt_inputs['input_ids'].shape}")
    print()

    # -----------------------------------------------------------------------
    # Load all three engine backends.
    # -----------------------------------------------------------------------
    print("[2/4] Loading engine backends ...")
    print()

    # --- PyTorch Eager ---
    print("  Engine 1/3: PyTorch Eager")
    model_eager = load_pytorch_eager(MODEL_NAME, device=device)
    forward_eager = make_pytorch_forward(model_eager, device=device)
    print()

    # --- PyTorch Compiled ---
    print("  Engine 2/3: PyTorch Compiled (torch.compile)")
    model_compiled = load_pytorch_compiled(MODEL_NAME, device=device)
    forward_compiled = None
    if model_compiled is not None:
        forward_compiled = make_pytorch_forward(model_compiled, device=device)
    print()

    # --- ONNX Runtime ---
    print("  Engine 3/3: ONNX Runtime CPU")
    onnx_session = load_onnx_model(MODEL_NAME, ONNX_EXPORT_PATH)
    forward_onnx = make_onnx_forward(onnx_session)
    print()

    # -----------------------------------------------------------------------
    # Run benchmarks.
    # -----------------------------------------------------------------------
    print("[3/4] Running benchmarks ...")
    print()

    result = BenchmarkResult()

    engines = [
        ("PyTorch Eager", forward_eager),
    ]
    if forward_compiled is not None:
        engines.append(("PyTorch Compiled", forward_compiled))
    engines.append(("ONNX Runtime CPU", forward_onnx))

    for name, forward_fn in engines:
        print(f"  --- {name} ---")
        stats = run_benchmark(
            engine_name=name,
            forward_fn=forward_fn,
            pt_inputs=pt_inputs,
            np_inputs=np_inputs,
            warmup=WARMUP_RUNS,
            runs=TIMED_RUNS,
        )
        result.stats.append(stats)
        print()

    # -----------------------------------------------------------------------
    # Determine winner and compute derived timeout.
    # -----------------------------------------------------------------------
    print("[4/4] Determining winner and generating report ...")
    print()

    # Winner = engine with lowest p95.
    best = min(result.stats, key=lambda s: s.p95_ms)
    result.winner = best.engine_name
    result.winner_p95_ms = best.p95_ms
    result.derived_timeout_ms = round(best.p95_ms * 1.5, 2)

    print(f"  Winning engine: {result.winner}")
    print(f"  Base p95:       {result.winner_p95_ms:.2f} ms")
    print(f"  Derived timeout: {result.derived_timeout_ms:.2f} ms "
          f"({result.derived_timeout_ms / 1000:.3f} s)")
    print()

    # -----------------------------------------------------------------------
    # Save report to Markdown file.
    # -----------------------------------------------------------------------
    report = generate_markdown_report(result)
    output_path = Path(OUTPUT_FILE)
    output_path.write_text(report, encoding="utf-8")
    print(f"  Report saved to '{output_path.absolute()}'")
    print()

    # -----------------------------------------------------------------------
    # Print console summary.
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for stat in result.stats:
        print(f"  {stat.engine_name:25s}  "
              f"p50={stat.p50_ms:8.2f} ms  "
              f"p95={stat.p95_ms:8.2f} ms  "
              f"p99={stat.p99_ms:8.2f} ms")
    print()
    print(f"  WINNER:    {result.winner}")
    print(f"  TIMEOUT:   {result.derived_timeout_ms:.2f} ms "
          f"({result.derived_timeout_ms / 1000:.3f} s)")
    print("=" * 60)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
