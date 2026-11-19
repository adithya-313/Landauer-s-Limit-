"""Unit tests for schema dump normalization and parity comparison."""

import hashlib
import pytest
from scripts.check_schema_parity import (
    compare_schema_dumps,
    normalize_schema_dump,
    require_tables,
)


def test_normalize_schema_dump_strips_comments_and_directives():
    """Verify that comment lines, blank lines, and psql directives are removed."""
    raw = (
        "-- PostgreSQL database dump\n"
        "-- Dumped by pg_dump version 15.10\n"
        "\n"
        "\\restrict some_setting\n"
        "\\unrestrict other_setting\n"
        "CREATE TABLE test (\n"
        "    id integer NOT NULL,\n"
        "    -- column comment\n"
        "    name text\n"
        ");\n"
        "\n"
        "-- End of dump\n"
    )
    expected = (
        "CREATE TABLE test (\n"
        "id integer NOT NULL,\n"
        "name text\n"
        ");\n"
    )
    normalized = normalize_schema_dump(raw)
    assert normalized == expected


def test_compare_schema_dumps_matching():
    """Verify identical schema dumps return True with correct SHA-256."""
    dump_a = (
        "-- Header\n"
        "CREATE TABLE tenants (\n"
        "    tenant_id text NOT NULL\n"
        ");\n"
    )
    dump_b = (
        "\\restrict\n"
        "CREATE TABLE tenants (\n"
        "    tenant_id text NOT NULL\n"
        ");\n"
    )
    is_match, digest, diff = compare_schema_dumps(dump_a, dump_b)
    assert is_match is True
    assert diff == ""
    expected_norm = "CREATE TABLE tenants (\ntenant_id text NOT NULL\n);\n"
    assert digest == hashlib.sha256(expected_norm.encode("utf-8")).hexdigest()


def test_compare_schema_dumps_detects_one_column_difference():
    """Verify that a single column difference causes failure and produces a unified diff."""
    dump_init = (
        "CREATE TABLE tenants (\n"
        "    tenant_id text NOT NULL,\n"
        "    tier text NOT NULL\n"
        ");\n"
    )
    dump_fresh = (
        "CREATE TABLE tenants (\n"
        "    tenant_id text NOT NULL,\n"
        "    tier text NOT NULL,\n"
        "    extra_col text\n"
        ");\n"
    )
    is_match, digest, diff = compare_schema_dumps(dump_init, dump_fresh)
    assert is_match is False
    assert digest == ""
    assert "+extra_col text" in diff
    assert "--- parity_init" in diff
    assert "+++ parity_fresh" in diff


def test_require_tables_passes_when_all_tables_present():
    """Verify require_tables succeeds when all 4 expected tables are in dump."""
    dump = (
        "CREATE TABLE public.telemetry_events (\n"
        "    id integer NOT NULL\n"
        ");\n"
        "CREATE TABLE public.tenants (\n"
        "    tenant_id text NOT NULL\n"
        ");\n"
        "CREATE TABLE public.run_manifests (\n"
        "    run_id text NOT NULL\n"
        ");\n"
        "CREATE TABLE public.alembic_version (\n"
        "    version_num character varying(32) NOT NULL\n"
        ");\n"
        "CREATE TABLE public.usage_events (\n"
        "    id bigint NOT NULL\n"
        ");\n"
    )
    # Should complete without raising SystemExit
    require_tables(dump, "test_db")


def test_require_tables_fails_on_empty_dumps(capsys):
    """Verify require_tables exits 1 on empty schema dump."""
    with pytest.raises(SystemExit) as exc_info:
        require_tables("", "parity_init")
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "SCHEMA_PARITY: FAIL (missing table public.telemetry_events in parity_init)" in captured.err

    with pytest.raises(SystemExit) as exc_info2:
        require_tables("", "parity_fresh")
    assert exc_info2.value.code == 1
    captured2 = capsys.readouterr()
    assert "SCHEMA_PARITY: FAIL (missing table public.telemetry_events in parity_fresh)" in captured2.err


def test_require_tables_fails_when_run_manifests_missing(capsys):
    """Verify require_tables exits 1 and names missing run_manifests table."""
    dump_missing_run_manifests = (
        "CREATE TABLE public.telemetry_events (\n"
        "    id integer NOT NULL\n"
        ");\n"
        "CREATE TABLE public.tenants (\n"
        "    tenant_id text NOT NULL\n"
        ");\n"
        "CREATE TABLE public.alembic_version (\n"
        "    version_num character varying(32) NOT NULL\n"
        ");\n"
    )
    with pytest.raises(SystemExit) as exc_info:
        require_tables(dump_missing_run_manifests, "parity_fresh")
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "SCHEMA_PARITY: FAIL (missing table public.run_manifests in parity_fresh)" in captured.err


def test_require_tables_fails_when_usage_events_missing(capsys):
    """Verify require_tables exits 1 and names missing usage_events table."""
    dump_missing_usage_events = (
        "CREATE TABLE public.telemetry_events (\n"
        "    id integer NOT NULL\n"
        ");\n"
        "CREATE TABLE public.tenants (\n"
        "    tenant_id text NOT NULL\n"
        ");\n"
        "CREATE TABLE public.run_manifests (\n"
        "    run_id text NOT NULL\n"
        ");\n"
        "CREATE TABLE public.alembic_version (\n"
        "    version_num character varying(32) NOT NULL\n"
        ");\n"
    )
    with pytest.raises(SystemExit) as exc_info:
        require_tables(dump_missing_usage_events, "parity_fresh")
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "SCHEMA_PARITY: FAIL (missing table public.usage_events in parity_fresh)" in captured.err

