"""
scripts/ci_smoke.py
===================
PHASE 10 — CI smoke check for containerized stack with CPU llama.cpp engine.

Validates end-to-end functionality of the running stack:
1. POST a chat completion to the gateway with API key, stream SSE tokens, and
   verify non-empty, error-free output.
2. GET Grafana /api/health and verify the database status is "ok".

Inputs:
- Gateway and Grafana URLs via CLI flags (--gateway-url, --grafana-url)
  or environment variables (LANDAUER_GATEWAY_URL, GRAFANA_URL, etc.).
- LANDAUER_API_KEY from environment.
- GF_SECURITY_ADMIN_PASSWORD from environment.

Outputs:
- Exactly one PASS/FAIL line per check.
- Exits 0 if all checks pass, 1 if any check fails.
"""

import argparse
import json
import os
import sys
from typing import List

import httpx


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for gateway and Grafana service URLs.

    Returns:
    - Parsed Namespace containing gateway_url and grafana_url.
    """
    default_gateway_port = os.environ.get("GATEWAY_HOST_PORT", "8080")
    default_gateway_url = os.environ.get(
        "LANDAUER_GATEWAY_URL", f"http://127.0.0.1:{default_gateway_port}"
    )
    default_grafana_port = os.environ.get("GRAFANA_HOST_PORT", "3000")
    default_grafana_url = os.environ.get(
        "GRAFANA_URL", f"http://127.0.0.1:{default_grafana_port}"
    )

    parser = argparse.ArgumentParser(description="CI Smoke Test for Landauer Stack")
    parser.add_argument(
        "--gateway-url",
        default=default_gateway_url,
        help="Base URL for the Landauer gateway (e.g. http://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--grafana-url",
        default=default_grafana_url,
        help="Base URL for Grafana (e.g. http://127.0.0.1:3000)",
    )
    return parser.parse_args()


def test_gateway_chat_completion(gateway_url: str, api_key: str) -> bool:
    """
    Sends a chat completion request to the gateway and validates the SSE stream.

    Purpose:
    - Confirms that the gateway accepts authenticated requests, applies guardrails,
      and successfully dispatches streaming generation to the backend engine.

    Inputs:
    - gateway_url: Base URL of the gateway service.
    - api_key: Plaintext API key for tenant authentication.

    Returns:
    - True if the completion stream is HTTP 200, non-empty, and free of engine errors.
    - False otherwise.
    """
    if gateway_url.endswith("/v1/chat/completions"):
        endpoint = gateway_url
    else:
        endpoint = f"{gateway_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
    }
    payload = {
        "model": "qwen2.5-0.5b-instruct",
        "messages": [
            {"role": "user", "content": "What is 2+2? Answer in one word."}
        ],
        "tier": "premium",
        "stream": True,
    }

    collected_tokens: List[str] = []
    try:
        with httpx.Client(timeout=60.0) as client:
            with client.stream("POST", endpoint, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    error_body = response.read().decode("utf-8", errors="replace")
                    print(
                        f"FAIL: gateway status {response.status_code} != 200 (response: {error_body})"
                    )
                    return False

                for line in response.iter_lines():
                    trimmed = line.strip()
                    if not trimmed or trimmed == "data: [DONE]":
                        continue
                    if trimmed.startswith("data: "):
                        json_str = trimmed[6:]
                        try:
                            event_data = json.loads(json_str)
                            choices = event_data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    collected_tokens.append(content)
                        except json.JSONDecodeError:
                            if "[Engine error" in json_str:
                                print(f"FAIL: gateway stream contained engine error: {json_str}")
                                return False

        full_text = "".join(collected_tokens).strip()
        if not full_text:
            print("FAIL: gateway chat completion stream returned empty text")
            return False

        if "[Engine error" in full_text:
            print(f"FAIL: gateway chat completion contained engine error: {full_text}")
            return False

        print(f"PASS: gateway chat completion stream returned: '{full_text}'")
        return True

    except Exception as exc:
        print(f"FAIL: gateway request failed with exception: {exc}")
        return False


def test_grafana_health(grafana_url: str) -> bool:
    """
    Checks Grafana /api/health endpoint to ensure the database connection is healthy.

    Purpose:
    - Confirms that Grafana is up and its internal metadata database initialized properly.

    Inputs:
    - grafana_url: Base URL of the Grafana service.

    Returns:
    - True if HTTP 200 and database == 'ok'.
    - False otherwise.
    """
    endpoint = f"{grafana_url.rstrip('/')}/api/health"
    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.get(endpoint)
            if response.status_code != 200:
                print(f"FAIL: grafana health status {response.status_code} != 200")
                return False

            data = response.json()
            db_status = data.get("database")
            if db_status != "ok":
                print(f"FAIL: grafana health database status '{db_status}' != 'ok'")
                return False

            print(f"PASS: grafana health status database == '{db_status}'")
            return True

    except Exception as exc:
        print(f"FAIL: grafana health request failed with exception: {exc}")
        return False


def load_env_file() -> None:
    """
    Load environment variables from a local .env file if present.
    Does not overwrite existing environment variables.
    """
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, val = stripped.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            if key not in os.environ:
                os.environ[key] = val


def main() -> None:
    """
    Main entrypoint for CI smoke script.
    """
    load_env_file()
    args = parse_arguments()

    api_key = os.environ.get("LANDAUER_API_KEY") or os.environ.get("API_KEY")
    if not api_key:
        print("FAIL: LANDAUER_API_KEY environment variable is not set")
        sys.exit(1)

    gateway_ok = test_gateway_chat_completion(args.gateway_url, api_key)
    grafana_ok = test_grafana_health(args.grafana_url)

    if gateway_ok and grafana_ok:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
