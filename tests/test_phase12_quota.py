"""Unit tests for token bucket rate limiting, daily spend cap tracking, and quota management."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest

from quota import (
    DailySpendLedger,
    QuotaManager,
    TokenBucket,
    estimate_prompt_tokens,
)
from tenants import Tenant


class FakeMonotonicClock:
    """Controllable monotonic clock for deterministic time progression."""

    def __init__(self, initial: float = 1000.0) -> None:
        self.current: float = initial

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


class FakeWallClock:
    """Controllable wall clock returning UTC datetimes for daily rollover tests."""

    def __init__(self, initial: datetime) -> None:
        self.current: datetime = initial

    def __call__(self) -> datetime:
        return self.current

    def advance(self, td: timedelta) -> None:
        self.current += td


def test_estimate_prompt_tokens_basic_and_edge_cases() -> None:
    """Verify prompt token estimation calculates ceil(chars/4) with min 1."""
    # Empty messages list -> minimum 1 token
    assert estimate_prompt_tokens([]) == 1

    # Empty content and role -> minimum 1 token
    assert estimate_prompt_tokens([{"role": "", "content": ""}]) == 1

    # 4 chars -> 1 token
    assert estimate_prompt_tokens([{"role": "user", "content": ""}]) == 1  # 4 chars -> 1

    # 5 chars -> ceil(5/4) = 2 tokens
    assert estimate_prompt_tokens([{"role": "user", "content": "!"}]) == 2  # 5 chars -> 2

    # Multiple messages
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},  # 6 + 28 = 34
        {"role": "user", "content": "Hello!"},  # 4 + 6 = 10 -> total 44 chars -> 11 tokens
    ]
    assert estimate_prompt_tokens(msgs) == 11


def test_token_bucket_initial_balance_and_reservation() -> None:
    """Verify token bucket starts full, debits on reserve, and enforces capacity bounds."""
    clock = FakeMonotonicClock(100.0)
    bucket = TokenBucket(capacity=600.0, clock=clock)  # 600 tokens/min = 10 tokens/sec

    assert bucket.balance == 600.0

    # Reserve 200 tokens -> balance 400.0
    assert bucket.try_reserve(200) is True
    assert bucket.balance == 400.0

    # Reserve 300 tokens -> balance 100.0
    assert bucket.try_reserve(300) is True
    assert bucket.balance == 100.0

    # Try reserving 150 tokens when only 100 available -> False, undebited
    assert bucket.try_reserve(150) is False
    assert bucket.balance == 100.0


def test_token_bucket_over_capacity_rejection() -> None:
    """Verify a request exceeding whole bucket capacity is rejected immediately."""
    clock = FakeMonotonicClock(100.0)
    bucket = TokenBucket(capacity=100.0, clock=clock)

    assert bucket.try_reserve(101) is False
    assert bucket.balance == 100.0


def test_token_bucket_refill_and_capacity_cap() -> None:
    """Verify continuous refill restores tokens up to, but never exceeding, capacity."""
    clock = FakeMonotonicClock(100.0)
    bucket = TokenBucket(capacity=600.0, refill_per_second=10.0, clock=clock)

    assert bucket.try_reserve(600) is True
    assert bucket.balance == 0.0

    # Advance 10 seconds -> 100 tokens refilled
    clock.advance(10.0)
    assert bucket.try_reserve(100) is True
    assert bucket.balance == 0.0

    # Advance 120 seconds -> 1200 tokens generated, capped at capacity 600
    clock.advance(120.0)
    assert bucket.try_reserve(50) is True
    # Balance was 600, debited 50 -> 550
    assert bucket.balance == 550.0


def test_token_bucket_settle_refund() -> None:
    """Verify unused reserved tokens are refunded back to the bucket balance."""
    clock = FakeMonotonicClock(100.0)
    bucket = TokenBucket(capacity=1000.0, refill_per_second=10.0, clock=clock)

    assert bucket.try_reserve(100) is True
    assert bucket.balance == 900.0

    # Only 20 tokens actually used -> refund 80 tokens
    bucket.settle(reserved=100, actual=20)
    assert bucket.balance == 980.0


def test_token_bucket_settle_overrun_and_recovery() -> None:
    """Verify token consumption exceeding reservation drives balance negative, recovering over time."""
    clock = FakeMonotonicClock(100.0)
    bucket = TokenBucket(capacity=100.0, refill_per_second=10.0, clock=clock)

    assert bucket.try_reserve(50) is True
    assert bucket.balance == 50.0

    # Actual consumption was 80 (30 more than reserved)
    bucket.settle(reserved=50, actual=80)
    # 50 - 30 = 20.0
    assert bucket.balance == 20.0

    # Now drain remaining
    assert bucket.try_reserve(20) is True
    assert bucket.balance == 0.0

    # Overrun by 25 tokens
    bucket.settle(reserved=0, actual=25)
    assert bucket.balance == -25.0

    # Further reservations rejected while negative
    assert bucket.try_reserve(1) is False

    # Advance 3 seconds (30 tokens refilled): -25 + 30 = 5.0
    clock.advance(3.0)
    assert bucket.try_reserve(5) is True
    assert bucket.balance == 0.0


def test_daily_spend_ledger_accumulation_and_utc_rollover() -> None:
    """Verify daily spend accumulates and resets cleanly at 00:00:00 UTC."""
    initial_time = datetime(2026, 11, 19, 23, 50, 0, tzinfo=timezone.utc)
    wall_clock = FakeWallClock(initial_time)
    ledger = DailySpendLedger(clock=wall_clock)

    assert ledger.spent_today("tenant_a") == 0.0

    ledger.add_spend("tenant_a", 1.25)
    ledger.add_spend("tenant_a", 0.75)
    ledger.add_spend("tenant_b", 3.00)

    assert ledger.spent_today("tenant_a") == 2.00
    assert ledger.spent_today("tenant_b") == 3.00

    # Advance 15 minutes past UTC midnight (into 2026-11-20)
    wall_clock.advance(timedelta(minutes=15))

    assert ledger.spent_today("tenant_a") == 0.0
    assert ledger.spent_today("tenant_b") == 0.0

    # Accumulate on the new day
    ledger.add_spend("tenant_a", 0.50)
    assert ledger.spent_today("tenant_a") == 0.50


def test_quota_manager_unlimited_tenant() -> None:
    """Verify tenant without configured limits bypasses token and spend checks."""
    qm = QuotaManager()
    tenant = Tenant(
        tenant_id="unlimited_user",
        tier="premium",
        key_sha256="a" * 64,
        tokens_per_minute=None,
        spend_cap_per_day=None,
    )

    decision = qm.check_and_reserve(tenant, prompt_tokens=50, max_tokens=200)
    assert decision.allowed is True
    assert decision.reason is None
    assert decision.reservation is None
    assert decision.requested_tokens == 250
    assert decision.available_tokens == float("inf")

    # Settling with no reservation does not raise error
    qm.settle(tenant, reservation=None, actual_tokens=120, energy_cost=0.15)


def test_quota_manager_spend_cap_enforcement() -> None:
    """Verify daily spend cap acts as soft ceiling evaluated on arrival."""
    wall_clock = FakeWallClock(datetime(2026, 11, 19, 12, 0, 0, tzinfo=timezone.utc))
    qm = QuotaManager(wall_clock=wall_clock)
    tenant = Tenant(
        tenant_id="budget_tenant",
        tier="free",
        key_sha256="b" * 64,
        tokens_per_minute=None,
        spend_cap_per_day=5.00,
    )

    # Initially allowed
    d1 = qm.check_and_reserve(tenant, prompt_tokens=10, max_tokens=100)
    assert d1.allowed is True
    assert d1.reservation is None

    # Simulate request completing and adding 4.90 spend (below cap)
    qm.settle(tenant, reservation=None, actual_tokens=80, energy_cost=4.90)

    # Next request arrival: spent_today (4.90) < cap (5.00), so allowed!
    d2 = qm.check_and_reserve(tenant, prompt_tokens=10, max_tokens=100)
    assert d2.allowed is True

    # That request costs 0.20, pushing total to 5.10 (over cap)
    qm.settle(tenant, reservation=None, actual_tokens=90, energy_cost=0.20)

    # Subsequent request arrival: spent_today (5.10) >= cap (5.00) -> rejected!
    d3 = qm.check_and_reserve(tenant, prompt_tokens=10, max_tokens=100)
    assert d3.allowed is False
    assert d3.reason == "quota_spend"
    assert d3.spent_today == pytest.approx(5.10)
    assert d3.spend_cap == 5.00

    # Advance to next UTC day
    wall_clock.advance(timedelta(days=1))
    d4 = qm.check_and_reserve(tenant, prompt_tokens=10, max_tokens=100)
    assert d4.allowed is True
    assert d4.spent_today == 0.0


def test_quota_manager_token_limit_and_reservation_lifecycle() -> None:
    """Verify token bucket check, reservation debit, and settle refund in QuotaManager."""
    mono_clock = FakeMonotonicClock(100.0)
    qm = QuotaManager(clock=mono_clock)
    tenant = Tenant(
        tenant_id="rate_limited_tenant",
        tier="free",
        key_sha256="c" * 64,
        tokens_per_minute=500,
        spend_cap_per_day=None,
    )

    # Request 100 prompt + 200 max = 300 tokens
    d1 = qm.check_and_reserve(tenant, prompt_tokens=100, max_tokens=200)
    assert d1.allowed is True
    assert d1.reason is None
    assert d1.reservation is not None
    assert d1.requested_tokens == 300
    assert d1.available_tokens == 200.0  # 500 - 300

    # Try requesting 250 tokens -> exceeds 200 remaining balance -> rejected
    d2 = qm.check_and_reserve(tenant, prompt_tokens=50, max_tokens=200)
    assert d2.allowed is False
    assert d2.reason == "quota_tokens"
    assert d2.reservation is None

    # Request 1 completes: used only 80 actual tokens -> refunds 300 - 80 = 220
    qm.settle(tenant, reservation=d1.reservation, actual_tokens=80, energy_cost=None)

    # Balance was 200 + 220 = 420.0
    # Now the 250 request can succeed
    d3 = qm.check_and_reserve(tenant, prompt_tokens=50, max_tokens=200)
    assert d3.allowed is True


def test_quota_manager_double_settle_raises() -> None:
    """Verify attempting to settle the same reservation twice raises ValueError."""
    qm = QuotaManager()
    tenant = Tenant(
        tenant_id="t1",
        tier="premium",
        key_sha256="d" * 64,
        tokens_per_minute=1000,
    )

    decision = qm.check_and_reserve(tenant, prompt_tokens=10, max_tokens=50)
    assert decision.reservation is not None

    qm.settle(tenant, reservation=decision.reservation, actual_tokens=40, energy_cost=None)

    with pytest.raises(ValueError, match="already been settled"):
        qm.settle(tenant, reservation=decision.reservation, actual_tokens=40, energy_cost=None)


def test_tenant_model_validation_rejects_negative_or_zero_limits() -> None:
    """Verify Tenant pydantic model rejects non-positive tokens_per_minute and spend_cap_per_day."""
    with pytest.raises(ValueError, match="tokens_per_minute must be > 0"):
        Tenant(
            tenant_id="t_bad",
            tier="free",
            key_sha256="e" * 64,
            tokens_per_minute=0,
        )

    with pytest.raises(ValueError, match="spend_cap_per_day must be > 0"):
        Tenant(
            tenant_id="t_bad",
            tier="free",
            key_sha256="e" * 64,
            spend_cap_per_day=-1.0,
        )
