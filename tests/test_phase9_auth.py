from __future__ import annotations

import hashlib
from pathlib import Path
import re

import pytest
import yaml

from tenants import Tenant, TenantStore, default_tenants_path, load_tenants


def test_tenants_valid_load(tmp_path: Path) -> None:
    """Verifies that a well-formed tenants YAML file loads successfully."""
    config_path = tmp_path / "valid_tenants.yaml"
    content = {
        "tenants": [
            {
                "tenant_id": "tenant_a",
                "tier": "free",
                "key_sha256": "a" * 64,
            },
            {
                "tenant_id": "tenant_b",
                "tier": "premium",
                "key_sha256": "b" * 64,
            },
        ]
    }
    config_path.write_text(yaml.safe_dump(content), encoding="utf-8")

    store = load_tenants(config_path)
    assert len(store.tenants) == 2
    assert store.tenants[0].tenant_id == "tenant_a"
    assert store.tenants[0].tier == "free"
    assert store.tenants[1].tenant_id == "tenant_b"
    assert store.tenants[1].tier == "premium"


def test_tenants_missing_file_raises(tmp_path: Path) -> None:
    """Verifies that attempting to load a nonexistent file raises ValueError."""
    nonexistent = tmp_path / "does_not_exist.yaml"
    with pytest.raises(ValueError, match="does not exist"):
        load_tenants(nonexistent)


def test_tenants_duplicate_id_raises(tmp_path: Path) -> None:
    """Verifies that duplicate tenant_id values raise ValueError."""
    config_path = tmp_path / "dup_id.yaml"
    content = {
        "tenants": [
            {
                "tenant_id": "same_id",
                "tier": "free",
                "key_sha256": "a" * 64,
            },
            {
                "tenant_id": "same_id",
                "tier": "premium",
                "key_sha256": "b" * 64,
            },
        ]
    }
    config_path.write_text(yaml.safe_dump(content), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate tenant_id 'same_id'"):
        load_tenants(config_path)


def test_tenants_duplicate_hash_raises(tmp_path: Path) -> None:
    """Verifies that duplicate key_sha256 hashes raise ValueError."""
    config_path = tmp_path / "dup_hash.yaml"
    content = {
        "tenants": [
            {
                "tenant_id": "tenant_one",
                "tier": "free",
                "key_sha256": "c" * 64,
            },
            {
                "tenant_id": "tenant_two",
                "tier": "premium",
                "key_sha256": "c" * 64,
            },
        ]
    }
    config_path.write_text(yaml.safe_dump(content), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate key_sha256"):
        load_tenants(config_path)


def test_tenants_bad_hash_length_and_uppercase_raises(tmp_path: Path) -> None:
    """Verifies that invalid hash lengths and uppercase hex characters raise ValueError."""
    # Bad length
    bad_len_path = tmp_path / "bad_len.yaml"
    bad_len_path.write_text(
        yaml.safe_dump({"tenants": [{"tenant_id": "t1", "tier": "free", "key_sha256": "1234"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid"):
        load_tenants(bad_len_path)

    # Uppercase hex chars
    upper_path = tmp_path / "upper.yaml"
    upper_path.write_text(
        yaml.safe_dump(
            {"tenants": [{"tenant_id": "t2", "tier": "free", "key_sha256": ("A" * 64)}]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid"):
        load_tenants(upper_path)


def test_tenant_store_lookup_success_and_failure() -> None:
    """Verifies lookup behavior with valid keys, wrong keys, empty strings, and None."""
    raw_secret = "my_secret_token_12345"
    secret_hash = hashlib.sha256(raw_secret.encode("utf-8")).hexdigest()

    tenant_record = Tenant(
        tenant_id="dev_user",
        tier="premium",
        key_sha256=secret_hash,
    )
    store = TenantStore([tenant_record])

    # Valid key -> returns Tenant
    found = store.lookup(raw_secret)
    assert found is not None
    assert found.tenant_id == "dev_user"
    assert found.tier == "premium"

    # Wrong key -> returns None
    assert store.lookup("completely_incorrect_key") is None

    # Empty string and None -> returns None
    assert store.lookup("") is None
    assert store.lookup(None) is None


def test_committed_tenants_yaml_contains_no_plaintext_keys() -> None:
    """
    Verifies that the committed tenants.yaml loads cleanly and contains
    no string longer than 64 characters or non-hash key values.
    """
    tenants_path = default_tenants_path()
    assert tenants_path.is_file(), f"Expected {tenants_path} to exist"

    raw_text = tenants_path.read_text(encoding="utf-8")
    store = load_tenants(tenants_path)
    assert len(store.tenants) >= 1

    # Check each token and string in the file to guarantee no plaintext key leaked
    hex_64_re = re.compile(r"^[a-f0-9]{64}$")
    for tenant in store.tenants:
        assert hex_64_re.match(tenant.key_sha256), f"Invalid hash: {tenant.key_sha256}"

    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # If line has a scalar string value longer than 64 chars, it is suspicious
        parts = line.split(":", 1)
        if len(parts) == 2:
            val = parts[1].strip().strip('"').strip("'")
            if len(val) > 64:
                pytest.fail(f"Found scalar value longer than 64 characters in tenants.yaml: {val}")
