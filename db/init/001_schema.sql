-- Schema initialization for Landauer's Limit production telemetry and tenants.
-- Note: DOUBLE PRECISION is used for all floating-point columns (FLOAT in the specification).

CREATE TABLE IF NOT EXISTS telemetry_events (
    id SERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    engine TEXT NOT NULL,
    concurrency INT NOT NULL,
    ttft_ms DOUBLE PRECISION,
    itl_ms DOUBLE PRECISION,
    tokens_per_sec DOUBLE PRECISION,
    cache_hit BOOLEAN NOT NULL,
    agent_enabled BOOLEAN NOT NULL,
    tenant_id TEXT,
    power_draw_w DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_telemetry_events_timestamp ON telemetry_events (timestamp);
CREATE INDEX IF NOT EXISTS idx_telemetry_events_engine_timestamp ON telemetry_events (engine, timestamp);

CREATE TABLE IF NOT EXISTS tenants (
    tenant_id TEXT PRIMARY KEY,
    tier TEXT NOT NULL CHECK (tier IN ('free', 'premium')),
    key_sha256 CHAR(64) NOT NULL UNIQUE
);
