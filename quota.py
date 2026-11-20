"""Token quota and daily spend cap management for multi-tenant admission control."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import time
from typing import Callable, Literal, Optional
import uuid

from tenants import Tenant


def estimate_prompt_tokens(messages: list[dict[str, str]]) -> int:
    """Estimates prompt token count from a list of chat messages using a character heuristic.

    Purpose:
        Provide a fast, deterministic, tokenizer-free upper-bound estimate of prompt token
        consumption during request admission control.

    Inputs:
        messages: List of message dictionaries, each containing 'role' and/or 'content' strings.

    Outputs:
        int: Estimated token count, with a strict floor of 1.

    Key Logic:
        Aggregates character lengths of 'role' and 'content' fields across all messages, divides
        by 4, rounds up to the nearest integer (ceil), and enforces a minimum of 1 token.
        WHY: Loading a full BPE/WordPiece tokenizer (e.g. HuggingFace or tiktoken) inside the
        critical admission path introduces unacceptable latency, memory overhead, and GIL
        contention. For English text and code, 1 token ~ 4 characters is accurate within ~15%,
        providing a reliable heuristic reservation without tokenizing overhead.
    """
    total_chars = 0
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role")
            if isinstance(role, str):
                total_chars += len(role)
            content = msg.get("content")
            if isinstance(content, str):
                total_chars += len(content)
    tokens = math.ceil(total_chars / 4)
    return max(1, tokens)


@dataclass
class Reservation:
    """Represents a provisional token allocation debited from a tenant's token bucket."""

    reservation_id: str
    tenant_id: str
    reserved_tokens: int
    settled: bool = False


class TokenBucket:
    """Per-tenant token bucket implementing continuous refill and reservation mechanics."""

    def __init__(
        self,
        capacity: float,
        refill_per_second: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initializes a token bucket with capacity and refill rate.

        Inputs:
            capacity: Maximum burst capacity in tokens (tokens_per_minute).
            refill_per_second: Tokens added per second. Defaults to capacity / 60.0.
            clock: Monotonic time callable for deterministic testing.
        """
        if capacity <= 0.0:
            raise ValueError(f"TokenBucket capacity must be > 0, got {capacity}")
        self.capacity: float = float(capacity)
        self.refill_per_second: float = (
            float(refill_per_second) if refill_per_second is not None else (self.capacity / 60.0)
        )
        self.balance: float = float(capacity)
        self.clock: Callable[[], float] = clock
        self.last_refill: float = self.clock()

    def _refill(self) -> None:
        """Refills tokens based on elapsed monotonic time, capped at capacity."""
        now = self.clock()
        elapsed = now - self.last_refill
        if elapsed > 0:
            self.balance = min(self.capacity, self.balance + elapsed * self.refill_per_second)
            self.last_refill = now

    def try_reserve(self, n: int) -> bool:
        """Attempts to reserve n tokens from the bucket balance.

        Inputs:
            n: Number of tokens requested.

        Outputs:
            bool: True if reservation succeeded and balance was debited; False otherwise.

        Key Logic:
            Refills first. If n exceeds total bucket capacity, the request can never be satisfied
            within a single minute bucket, so return False immediately without debiting.
            If available balance is at least n, debit n from balance and return True.
            Otherwise, return False without altering balance.
        """
        if n > self.capacity:
            return False

        self._refill()

        if self.balance >= n:
            self.balance -= n
            return True
        return False

    def settle(self, reserved: int, actual: int) -> None:
        """Settles a reservation by refunding unused tokens or debiting overrun tokens.

        Inputs:
            reserved: Tokens originally reserved.
            actual: Actual tokens consumed.

        Key Logic:
            Refills first to account for elapsed time during generation.
            Adjusts balance by (reserved - actual). If positive overflow occurs, balance is
            capped at self.capacity. If actual > reserved, balance may become negative,
            requiring future refills before subsequent reservations can be approved.
        """
        self._refill()
        self.balance = min(self.capacity, self.balance + (reserved - actual))


class DailySpendLedger:
    """Tracks cumulative daily energy spend per tenant with automatic UTC midnight reset."""

    def __init__(self, clock: Optional[Callable[[], datetime]] = None) -> None:
        """Initializes the spend ledger with an optional injectable UTC wall clock."""
        self._clock: Callable[[], datetime] = (
            clock if clock is not None else (lambda: datetime.now(timezone.utc))
        )
        self._current_date = self._clock().date()
        self._spend_by_tenant: dict[str, float] = {}

    def _rollover_if_needed(self) -> None:
        """Clears accumulated spend if the current UTC calendar date has advanced."""
        today = self._clock().date()
        if today != self._current_date:
            self._spend_by_tenant.clear()
            self._current_date = today

    def spent_today(self, tenant_id: str) -> float:
        """Returns the total energy cost accumulated today in UTC for the given tenant."""
        self._rollover_if_needed()
        return self._spend_by_tenant.get(tenant_id, 0.0)

    def add_spend(self, tenant_id: str, amount: float) -> None:
        """Adds spend to today's accumulator for the given tenant."""
        if amount <= 0.0:
            return
        self._rollover_if_needed()
        self._spend_by_tenant[tenant_id] = self._spend_by_tenant.get(tenant_id, 0.0) + amount


@dataclass
class QuotaDecision:
    """Outcome of a quota evaluation and token reservation attempt."""

    allowed: bool
    reason: Optional[Literal["quota_tokens", "quota_spend"]]
    reservation: Optional[Reservation]
    requested_tokens: int
    available_tokens: float
    spent_today: float
    spend_cap: Optional[float]


class QuotaManager:
    """Coordinates rate-limiting token buckets and daily spend caps across tenants."""

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        """Initializes the QuotaManager with monotonic and wall clocks for deterministic testing."""
        self.clock: Callable[[], float] = clock
        self.spend_ledger: DailySpendLedger = DailySpendLedger(clock=wall_clock)
        self._buckets: dict[str, TokenBucket] = {}

    def _get_or_create_bucket(self, tenant: Tenant) -> TokenBucket:
        """Retrieves or instantiates the TokenBucket for a tenant."""
        capacity = float(tenant.tokens_per_minute)  # type: ignore[arg-type]
        bucket = self._buckets.get(tenant.tenant_id)
        if bucket is None or bucket.capacity != capacity:
            bucket = TokenBucket(capacity=capacity, clock=self.clock)
            self._buckets[tenant.tenant_id] = bucket
        return bucket

    def check_and_reserve(
        self, tenant: Tenant, prompt_tokens: int, max_tokens: int
    ) -> QuotaDecision:
        """Evaluates daily spend cap and token bucket limits, reserving tokens if permitted.

        Inputs:
            tenant: Tenant configuration model.
            prompt_tokens: Estimated prompt token count.
            max_tokens: Requested or default maximum output token count.

        Outputs:
            QuotaDecision: Decision record detailing admission status and reservation details.

        Key Logic:
            1. Check daily spend cap: if spent_today >= spend_cap_per_day, reject with
               reason 'quota_spend'. (Soft ceiling: evaluated on entry; does not reserve).
            2. Check token bucket: if tokens_per_minute is configured, attempt reservation of
               (prompt_tokens + max_tokens). If insufficient balance or request > capacity,
               reject with reason 'quota_tokens'.
            3. If limits are absent or satisfied, approve and return provisional Reservation.
        """
        requested_tokens = prompt_tokens + max_tokens
        spent_today = self.spend_ledger.spent_today(tenant.tenant_id)
        spend_cap = tenant.spend_cap_per_day

        # 1. Soft ceiling check on daily spend cap
        if spend_cap is not None and spent_today >= spend_cap:
            bucket = self._buckets.get(tenant.tenant_id)
            available = bucket.balance if bucket is not None else float("inf")
            return QuotaDecision(
                allowed=False,
                reason="quota_spend",
                reservation=None,
                requested_tokens=requested_tokens,
                available_tokens=available,
                spent_today=spent_today,
                spend_cap=spend_cap,
            )

        # 2. Token bucket reservation check
        if tenant.tokens_per_minute is not None:
            bucket = self._get_or_create_bucket(tenant)
            reserved = bucket.try_reserve(requested_tokens)
            if not reserved:
                return QuotaDecision(
                    allowed=False,
                    reason="quota_tokens",
                    reservation=None,
                    requested_tokens=requested_tokens,
                    available_tokens=bucket.balance,
                    spent_today=spent_today,
                    spend_cap=spend_cap,
                )
            reservation = Reservation(
                reservation_id=str(uuid.uuid4()),
                tenant_id=tenant.tenant_id,
                reserved_tokens=requested_tokens,
            )
            return QuotaDecision(
                allowed=True,
                reason=None,
                reservation=reservation,
                requested_tokens=requested_tokens,
                available_tokens=bucket.balance,
                spent_today=spent_today,
                spend_cap=spend_cap,
            )

        # Neither limit configured, or spend cap is set and not exceeded with no token limit
        return QuotaDecision(
            allowed=True,
            reason=None,
            reservation=None,
            requested_tokens=requested_tokens,
            available_tokens=float("inf"),
            spent_today=spent_today,
            spend_cap=spend_cap,
        )

    def settle(
        self,
        tenant: Tenant,
        reservation: Optional[Reservation],
        actual_tokens: int,
        energy_cost: Optional[float],
    ) -> None:
        """Settles token reservation and records actual energy cost.

        Inputs:
            tenant: Tenant configuration model.
            reservation: Optional Reservation returned by check_and_reserve.
            actual_tokens: Actual total tokens (prompt + output) consumed.
            energy_cost: Actual electricity cost in currency units (or None if unpriced/unmeasured).

        Errors:
            ValueError: If reservation is non-None and has already been settled (idempotency guard).
        """
        if reservation is not None:
            if reservation.settled:
                raise ValueError(
                    f"Reservation '{reservation.reservation_id}' has already been settled"
                )
            reservation.settled = True
            bucket = self._buckets.get(tenant.tenant_id)
            if bucket is not None:
                bucket.settle(reserved=reservation.reserved_tokens, actual=actual_tokens)

        if energy_cost is not None and energy_cost > 0.0:
            self.spend_ledger.add_spend(tenant.tenant_id, energy_cost)
