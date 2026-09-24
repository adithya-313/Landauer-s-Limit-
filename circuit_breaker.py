"""
circuit_breaker.py
==================
PHASE 7a-4 — Circuit Breaker (Closed → Open → Half-Open).

Wraps each engine adapter call. Tracks consecutive failures. After a
threshold, trips open. After a cooldown, goes half-open and tries one
test call.

States
------
CLOSED    : Normal operation. Calls pass through.
OPEN      : Tripped. All calls rejected immediately.
HALF_OPEN : Cooldown expired. One test call allowed; success → CLOSED,
            failure → OPEN (reset cooldown).
"""

import time
from typing import Any, Callable, Dict

from protective_actions import force_circuit_breaker_open

# ---------------------------------------------------------------------------
# STATE CONSTANTS
# ---------------------------------------------------------------------------
_CLOSED = "closed"
_OPEN = "open"
_HALF_OPEN = "half_open"


# ---------------------------------------------------------------------------
# MODULE-LEVEL REGISTRY
# ---------------------------------------------------------------------------
# One CircuitBreaker instance per engine adapter, keyed by engine_id.
# Admission controller checks this registry to decide routing.
_breakers: Dict[str, "CircuitBreaker"] = {}


def get_circuit_breaker(engine_id: str, failure_threshold: int = 3, cooldown_seconds: float = 30.0) -> "CircuitBreaker":
    """
    Returns (or creates) the CircuitBreaker for the given engine_id.

    Parameters
    ----------
    engine_id : str
        Matches the adapter key used in EngineRouter (e.g. "vllm_local").
    failure_threshold : int
        Consecutive failures before tripping. Ignored if breaker already exists.
    cooldown_seconds : float
        Seconds to stay OPEN before going HALF_OPEN. Ignored if breaker already exists.
    """
    if engine_id not in _breakers:
        _breakers[engine_id] = CircuitBreaker(engine_id, failure_threshold, cooldown_seconds)
    return _breakers[engine_id]


# ---------------------------------------------------------------------------
# CIRCUIT BREAKER CLASS
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """
    Wraps an engine adapter call with closed → open → half-open logic.

    Parameters
    ----------
    engine_id : str
        Identifier for the engine this breaker protects.
    failure_threshold : int
        How many consecutive failures before tripping (default: 3).
    cooldown_seconds : float
        How long to stay OPEN before going HALF_OPEN (default: 30.0).
    """

    def __init__(
        self,
        engine_id: str,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30.0,
    ) -> None:
        self.engine_id = engine_id
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds

        self._state: str = _CLOSED
        self._failure_count: int = 0
        self._last_opened_at: float = 0.0  # monotonic timestamp of last OPEN trip

    # ------------------------------------------------------------------
    # STATE QUERIES
    # ------------------------------------------------------------------

    def is_open(self) -> bool:
        """
        Returns True if the breaker is in OPEN state and the cooldown has NOT expired.
        If cooldown HAS expired, transitions to HALF_OPEN and returns False.
        """
        if self._state == _OPEN:
            elapsed = time.monotonic() - self._last_opened_at
            if elapsed >= self.cooldown_seconds:
                # Cooldown expired — allow one test call.
                self._state = _HALF_OPEN
                return False
            return True
        return False

    def state(self) -> str:
        """Returns the current state string for introspection / tests."""
        # Refresh state (handles cooldown expiry side-effect).
        self.is_open()
        return self._state

    # ------------------------------------------------------------------
    # RECORDING
    # ------------------------------------------------------------------

    def record_success(self) -> None:
        """
        Call after a successful engine call.
        Resets failure count. If HALF_OPEN, closes the breaker.
        """
        self._failure_count = 0
        if self._state in (_HALF_OPEN, _OPEN):
            self._state = _CLOSED

    def record_failure(self) -> None:
        """
        Call after a failed engine call.
        Increments failure count. If count >= threshold, opens the breaker
        and calls force_circuit_breaker_open() from protective_actions.
        """
        self._failure_count += 1
        if self._failure_count >= self.failure_threshold and self._state != _OPEN:
            self._state = _OPEN
            self._last_opened_at = time.monotonic()
            force_circuit_breaker_open(
                engine_id=self.engine_id,
                reason=(
                    f"Consecutive failure threshold reached "
                    f"({self._failure_count}/{self.failure_threshold})"
                ),
            )

    # ------------------------------------------------------------------
    # CALL WRAPPER
    # ------------------------------------------------------------------

    async def call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        The main entry point. Checks if open; raises immediately if so.
        Otherwise calls fn(*args, **kwargs) and records success or failure.

        Parameters
        ----------
        fn : Callable
            The async (or sync) function to call — typically adapter.generate.
        *args, **kwargs
            Forwarded to fn.

        Raises
        ------
        RuntimeError
            If the breaker is currently OPEN (circuit tripped).
        """
        if self.is_open():
            raise RuntimeError(
                f"CircuitBreaker OPEN for engine '{self.engine_id}'. "
                f"Cooldown: {self.cooldown_seconds}s"
            )

        try:
            result = await fn(*args, **kwargs)
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise
