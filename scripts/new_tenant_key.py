"""
scripts/new_tenant_key.py
Generates a cryptographically secure API key and prints its SHA-256 hash
formatted for insertion into tenants.yaml for either tenants or admins.
"""

from __future__ import annotations

import argparse
import hashlib
import secrets


def main() -> None:
    """
    Parses CLI arguments, generates an API key and outputs the YAML snippet.
    """
    parser = argparse.ArgumentParser(
        description="Generate API key and SHA-256 hash for tenants or admins."
    )
    parser.add_argument(
        "--admin",
        action="store_true",
        help="Generate an administrator key instead of a tenant key.",
    )
    parser.add_argument(
        "identifier",
        nargs="?",
        default=None,
        help="Identifier for the tenant (tenant_id) or admin (admin_id).",
    )
    parser.add_argument(
        "tier",
        nargs="?",
        default="premium",
        help="Tier for tenant mode ('free' or 'premium'). Defaults to 'premium'.",
    )

    args = parser.parse_args()

    plaintext_key = secrets.token_urlsafe(32)
    key_sha256 = hashlib.sha256(plaintext_key.encode("utf-8")).hexdigest()

    if args.admin:
        admin_id = args.identifier if args.identifier is not None else "local_admin"
        print("=== Generated Admin Key ===")
        print(f"Plaintext Key (save securely; will not be shown again):\n{plaintext_key}\n")
        print("YAML entry for tenants.yaml:")
        print("admins:")
        print(f"  - admin_id: {admin_id}")
        print(f"    key_sha256: {key_sha256}")
    else:
        tenant_id = args.identifier if args.identifier is not None else "local_dev"
        tier = args.tier
        print("=== Generated Tenant Key ===")
        print(f"Plaintext Key (save securely; will not be shown again):\n{plaintext_key}\n")
        print("YAML entry for tenants.yaml:")
        print(f"  - tenant_id: {tenant_id}")
        print(f"    tier: {tier}")
        print(f"    key_sha256: {key_sha256}")


if __name__ == "__main__":
    main()

