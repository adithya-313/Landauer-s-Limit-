"""
test_vram_oom_guard.py
======================
PHASE 5 — VLLMAdapter Preemptive VRAM OOM Guard Test Suite.

What this file is:
  This test suite verifies the VRAM safety guard built into VLLMAdapter._check_vram()
  and VLLMAdapter.generate(). It mocks the nvidia-smi subprocess call to simulate
  both low-VRAM (OOM risk) and healthy-VRAM hardware states.

Why it exists:
  Phase 5 requires proving that VLLMAdapter catches hardware VRAM saturation
  preemptively before attempting HTTP inference requests, logging the exact memory
  statistics and raising a clean RuntimeError rather than causing a silent crash or
  leaking unhandled exceptions to the gateway.

How to run:
  pytest test_vram_oom_guard.py -v
"""

import asyncio
from unittest.mock import patch, AsyncMock

import pytest

from adapters.vllm_adapter import VLLMAdapter, MIN_VRAM_MB


class DummyProcess:
    """
    Mock process object simulating asyncio.create_subprocess_shell() output.
    """
    def __init__(self, stdout_text: str, stderr_text: str = ""):
        self._stdout_bytes = stdout_text.encode("utf-8")
        self._stderr_bytes = stderr_text.encode("utf-8")
        self.returncode = 0

    async def communicate(self):
        return self._stdout_bytes, self._stderr_bytes


@pytest.mark.asyncio
async def test_vllm_oom_guard_low_vram_raises_and_logs(caplog):
    """
    Test the VRAM OOM safety guard under low available memory conditions.

    Mocks nvidia-smi output to return 6144 MB total and 100 MB free VRAM (below the
    512 MB MIN_VRAM_MB threshold). Verifies that VLLMAdapter.generate() raises a
    RuntimeError with 'OOM risk detected' in the message and logs the VRAM check details.
    """
    adapter = VLLMAdapter()

    # Mock output format from nvidia-smi: "total_mb, free_mb"
    mock_nvidia_smi_output = "6144, 100"
    mock_process = DummyProcess(mock_nvidia_smi_output)

    with patch("asyncio.create_subprocess_shell", AsyncMock(return_value=mock_process)):
        with caplog.at_level("INFO"):
            with pytest.raises(RuntimeError) as exc_info:
                # Iterate on generate() to trigger the preemptive VRAM check
                async for _ in adapter.generate("Test prompt under low memory"):
                    pass

            error_msg = str(exc_info.value)
            assert "OOM risk detected" in error_msg, (
                f"Expected 'OOM risk detected' in exception message, got: {error_msg}"
            )
            assert f"Free VRAM = 100 MB (< {MIN_VRAM_MB} MB threshold)" in error_msg, (
                f"Exception message missing exact memory figures: {error_msg}"
            )

            # Assert VRAM stats were logged
            assert any("VRAM check:" in record.message for record in caplog.records), (
                "Expected VRAM check details to be logged in caplog."
            )

    await adapter.close()


@pytest.mark.asyncio
async def test_vllm_oom_guard_healthy_vram_allows_check():
    """
    Test the VRAM OOM safety guard under normal, healthy memory conditions.

    Mocks nvidia-smi output to return 6144 MB total and 4096 MB free VRAM.
    Directly invokes adapter._check_vram() and asserts that oom_risk is False,
    free_mb is reported as 4096 MB, and no exception is raised.
    """
    adapter = VLLMAdapter()

    mock_nvidia_smi_output = "6144, 4096"
    mock_process = DummyProcess(mock_nvidia_smi_output)

    with patch("asyncio.create_subprocess_shell", AsyncMock(return_value=mock_process)):
        vram = await adapter._check_vram()

        assert vram["oom_risk"] is False, f"Expected oom_risk=False, got {vram['oom_risk']}"
        assert vram["total_mb"] == 6144, f"Expected total_mb=6144, got {vram['total_mb']}"
        assert vram["free_mb"] == 4096, f"Expected free_mb=4096, got {vram['free_mb']}"

    await adapter.close()
