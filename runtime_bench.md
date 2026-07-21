# Runtime Benchmark Report — all-MiniLM-L6-v2

- **Date:** 2026-07-21 15:01:31
- **Platform:** Windows 11 (AMD64 Family 25 Model 124 Stepping 0, AuthenticAMD)
- **PyTorch Version:** 2.5.1+cu121
- **ONNX Runtime Version:** 1.24.2
- **Warmup Iterations:** 10
- **Timed Iterations:** 100
- **Sample Texts:** 4

## Latency Comparison (ms)

| Engine | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) |
|--------|----------|----------|----------|-----------|
| PyTorch Eager | 49.25 | 75.13 | 188.64 | 52.73 |
| ONNX Runtime CPU | 33.13 | 65.32 | 80.39 | 37.14 |

## p95 Latency Distribution

```
  PyTorch Eager                     75.13 ms  ###############
  ONNX Runtime CPU                  65.32 ms  #############
```

## Winner & Derived Timeout

The **ONNX Runtime CPU** engine has the lowest p95 latency (65.32 ms).

### Timeout Calculation

```
  base_p95      = 65.32 ms
  safety_factor = 1.5
  -----------------------------
  derived_timeout = 65.32 × 1.5
                   = 97.98 ms
                   = 0.098 s
```
