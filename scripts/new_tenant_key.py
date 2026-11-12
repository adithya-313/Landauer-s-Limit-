import hashlib
import secrets
import sys


def main() -> None:
    tenant_id = sys.argv[1] if len(sys.argv) > 1 else "local_dev"
    tier = sys.argv[2] if len(sys.argv) > 2 else "premium"

    plaintext_key = secrets.token_urlsafe(32)
    key_sha256 = hashlib.sha256(plaintext_key.encode("utf-8")).hexdigest()

    print("=== Generated Tenant Key ===")
    print(f"Plaintext Key (save securely; will not be shown again):\n{plaintext_key}\n")
    print("YAML entry for tenants.yaml:")
    print(f"  - tenant_id: {tenant_id}")
    print(f"    tier: {tier}")
    print(f"    key_sha256: {key_sha256}")


if __name__ == "__main__":
    main()
