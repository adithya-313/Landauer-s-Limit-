# Proof of Work

This folder contains snapshots of our log files and test outputs. We keep these here as permanent evidence that our continuous batching engine works exactly as designed, because the live log files in the project change every time we run the server.

## Files in this folder

### `stage2_standalone_batch_events.jsonl`
**What it is:** A copy of the log file generated during the Stage 2 test (`test_stage2.py`), before we wired the engine up to the main server.
**What it proves:** It shows the engine correctly grouping incoming requests into batches and processing them together without needing the main server to manage anything.

### `stage3_staggered_realtraffic_batch_events.jsonl`
**What it is:** A snapshot of the live server log (`batch_events.jsonl`) captured during the Stage 3 staggered traffic test (`test_stage3_staggered.py`). 
**What it proves:** It proves the "revolving door" effect works under real HTTP traffic. If you look closely at the logs, you'll see a moment where one request finishes and leaves its slot, and a new request is immediately admitted into that exact vacated slot, even though the other concurrent requests are still generating. The engine doesn't wait for everyone to finish before letting new people in.

### `stage3_forced_error_test_output.txt`
**What it is:** The terminal output captured when we ran the forced error test (`test_stage3_error.py`). In this test, we temporarily wired the engine to crash whenever it saw the prompt `"crash_test"`.
**What it proves:** It shows that if the engine fails while processing a specific request, it sends a clean error message back to *just that one person*, without bringing down the whole server. The other requests running at the exact same time ("What is 2+2?" and "What is 3+3?") finished successfully.
