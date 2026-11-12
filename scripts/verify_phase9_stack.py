#!/usr/bin/env python3
"""
Phase 9 Production Stack End-to-End Verification Script.

Tests the full Docker Compose stack:
- Multi-container startup and healthcheck orchestration
- Authenticated requests and dual telemetry recording (JSONL + Postgres)
- Prometheus scraping and Grafana dashboard provisioning
- Resilient recovery from database failure
- Dynamic engine configuration reload
- Byte-for-byte 401 unauthorized response parity
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import httpx

def load_env() -> dict[str, str]:
    """Loads environment variables from os.environ and the local .env file."""
    env: dict[str, str] = dict(os.environ)
    env_file = Path(".env")
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k not in env or not env[k]:
                    env[k] = v
    return env


_ENV = load_env()
GATEWAY_URL = f"http://127.0.0.1:{_ENV.get('GATEWAY_HOST_PORT', '8080').strip() or '8080'}"
GRAFANA_URL = f"http://127.0.0.1:{_ENV.get('GRAFANA_HOST_PORT', '3000').strip() or '3000'}"
CONFIG_FILE = Path("deploy/gateway-config/config.yaml")
EVENTS_FILE = Path("telemetry/gateway_events.jsonl")


def main() -> None:
    global GATEWAY_URL, GRAFANA_URL
    env = load_env()
    GATEWAY_URL = f"http://127.0.0.1:{env.get('GATEWAY_HOST_PORT', '8080').strip() or '8080'}"
    GRAFANA_URL = f"http://127.0.0.1:{env.get('GRAFANA_HOST_PORT', '3000').strip() or '3000'}"
    api_key = env.get("LANDAUER_API_KEY", "").strip()
    gf_password = env.get("GF_SECURITY_ADMIN_PASSWORD", "").strip()

    if not api_key:
        print("[FAIL] LANDAUER_API_KEY not found in environment or .env", file=sys.stderr)
        sys.exit(1)
    if not gf_password:
        print("[FAIL] GF_SECURITY_ADMIN_PASSWORD not found in environment or .env", file=sys.stderr)
        sys.exit(1)

    all_pass = True

    # -------------------------------------------------------------------------
    # STEP 1: docker compose down -v, time docker compose up -d --wait
    # -------------------------------------------------------------------------
    print("[WARNING] Executing 'docker compose down -v' which deletes the named pgdata volume", file=sys.stderr)
    subprocess.run(["docker", "compose", "down", "-v"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    t0 = time.monotonic()
    up_proc = subprocess.run(["docker", "compose", "up", "-d", "--wait"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    startup_duration = round(time.monotonic() - t0, 2)
    step1_ok = (up_proc.returncode == 0) and (startup_duration <= 90.0)
    if not step1_ok:
        all_pass = False
    print(f"1 stack_startup {startup_duration}s {'PASS' if step1_ok else 'FAIL'}")

    # -------------------------------------------------------------------------
    # STEP 2: 10 POST /v1/chat/completions with key (llamacpp_local)
    # -------------------------------------------------------------------------
    http_200_count = 0
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "ping"}],
        "engine": "llamacpp_local",
        "stream": True,
    }
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=10.0) as client:
        for _ in range(10):
            try:
                with client.stream("POST", f"{GATEWAY_URL}/v1/chat/completions", json=payload, headers=headers) as resp:
                    if resp.status_code == 200:
                        http_200_count += 1
                        for _ in resp.iter_bytes():
                            pass
            except Exception as exc:
                print(f"[debug step 2 request error]: {exc}", file=sys.stderr)

    step2_ok = (http_200_count == 10)
    if not step2_ok:
        all_pass = False
    print(f"2 requests_http_200 {http_200_count}/10 {'PASS' if step2_ok else 'FAIL'}")

    # -------------------------------------------------------------------------
    # STEP 3: Wait up to 30s polling Grafana's Prometheus proxy
    # -------------------------------------------------------------------------
    prom_val = 0.0
    waited_s = 0.0
    step3_ok = False
    t_start_g = time.monotonic()

    with httpx.Client(timeout=5.0) as client:
        while time.monotonic() - t_start_g < 30.0:
            try:
                resp = client.get(
                    f"{GRAFANA_URL}/api/datasources/proxy/uid/prometheus/api/v1/query",
                    params={"query": "sum(landauer_requests_total)"},
                    auth=("admin", gf_password),
                )
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("data", {}).get("result", [])
                    if results:
                        prom_val = float(results[0]["value"][1])
                        if prom_val >= 10.0:
                            waited_s = round(time.monotonic() - t_start_g, 2)
                            step3_ok = True
                            break
            except Exception:
                pass
            time.sleep(1.0)

    if not step3_ok:
        waited_s = round(time.monotonic() - t_start_g, 2)
        all_pass = False
    print(f"3 grafana_prometheus_proxy_query {waited_s}s_val_{int(prom_val)} {'PASS' if step3_ok else 'FAIL'}")

    # -------------------------------------------------------------------------
    # STEP 4: Grafana /api/search finds both dashboard UIDs
    # -------------------------------------------------------------------------
    found_uids = 0
    with httpx.Client(timeout=5.0) as client:
        try:
            resp = client.get(f"{GRAFANA_URL}/api/search", auth=("admin", gf_password))
            if resp.status_code == 200:
                dashboards = resp.json()
                uids = {d.get("uid") for d in dashboards}
                if "landauer-perf" in uids:
                    found_uids += 1
                if "landauer-sla" in uids:
                    found_uids += 1
        except Exception as exc:
            print(f"[debug step 4 error]: {exc}", file=sys.stderr)

    step4_ok = (found_uids == 2)
    if not step4_ok:
        all_pass = False
    print(f"4 grafana_dashboards_found {found_uids} {'PASS' if step4_ok else 'FAIL'}")

    # -------------------------------------------------------------------------
    # STEP 5: Postgres row count in telemetry_events >= 10
    # -------------------------------------------------------------------------
    pg_count_step5 = 0
    try:
        out = subprocess.check_output(
            ["docker", "compose", "exec", "-T", "postgres", "psql", "-U", "landauer", "-d", "landauer", "-tAc", "select count(*) from telemetry_events"],
            text=True,
        ).strip()
        pg_count_step5 = int(out)
    except Exception as exc:
        print(f"[debug step 5 error]: {exc}", file=sys.stderr)

    step5_ok = (pg_count_step5 >= 10)
    if not step5_ok:
        all_pass = False
    print(f"5 postgres_events_count {pg_count_step5} {'PASS' if step5_ok else 'FAIL'}")

    # -------------------------------------------------------------------------
    # STEP 6: Stop Postgres, send 5 requests -> 5 HTTP 200, JSONL delta 5
    # -------------------------------------------------------------------------
    subprocess.run(["docker", "compose", "stop", "postgres"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    lines_before = 0
    if EVENTS_FILE.is_file():
        lines_before = len(EVENTS_FILE.read_text(encoding="utf-8").strip().splitlines())

    step6_200 = 0
    with httpx.Client(timeout=10.0) as client:
        for _ in range(5):
            try:
                with client.stream("POST", f"{GATEWAY_URL}/v1/chat/completions", json=payload, headers=headers) as resp:
                    if resp.status_code == 200:
                        step6_200 += 1
                        for _ in resp.iter_bytes():
                            pass
            except Exception as exc:
                print(f"[debug step 6 error]: {exc}", file=sys.stderr)

    time.sleep(1.0)
    lines_after = 0
    if EVENTS_FILE.is_file():
        lines_after = len(EVENTS_FILE.read_text(encoding="utf-8").strip().splitlines())
    jsonl_delta = lines_after - lines_before

    step6_ok = (step6_200 == 5) and (jsonl_delta == 5)
    if not step6_ok:
        all_pass = False
    print(f"6 postgres_down_dual_write 5_200_delta_{jsonl_delta} {'PASS' if step6_ok else 'FAIL'}")

    # -------------------------------------------------------------------------
    # STEP 7: Start Postgres, wait healthy, sleep 6, 3 requests -> delta 3, restart count unchanged
    # -------------------------------------------------------------------------
    subprocess.run(["docker", "compose", "start", "postgres"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait for postgres to be healthy
    for _ in range(30):
        try:
            chk = subprocess.check_output(
                ["docker", "compose", "exec", "-T", "postgres", "pg_isready", "-U", "landauer", "-d", "landauer"],
                text=True,
            )
            if "accepting connections" in chk:
                break
        except Exception:
            pass
        time.sleep(1.0)

    time.sleep(6.0)

    gateway_cid = subprocess.check_output(["docker", "compose", "ps", "-q", "gateway"], text=True).strip()
    restarts_before = subprocess.check_output(["docker", "inspect", "--format", "{{.RestartCount}}", gateway_cid], text=True).strip()

    with httpx.Client(timeout=10.0) as client:
        for _ in range(3):
            try:
                with client.stream("POST", f"{GATEWAY_URL}/v1/chat/completions", json=payload, headers=headers) as resp:
                    for _ in resp.iter_bytes():
                        pass
            except Exception as exc:
                print(f"[debug step 7 error]: {exc}", file=sys.stderr)

    time.sleep(1.0)
    pg_count_step7 = 0
    try:
        out = subprocess.check_output(
            ["docker", "compose", "exec", "-T", "postgres", "psql", "-U", "landauer", "-d", "landauer", "-tAc", "select count(*) from telemetry_events"],
            text=True,
        ).strip()
        pg_count_step7 = int(out)
    except Exception as exc:
        print(f"[debug step 7 count error]: {exc}", file=sys.stderr)

    pg_delta_step7 = pg_count_step7 - pg_count_step5
    restarts_after = subprocess.check_output(["docker", "inspect", "--format", "{{.RestartCount}}", gateway_cid], text=True).strip()

    step7_ok = (pg_delta_step7 == 3) and (restarts_before == restarts_after)
    if not step7_ok:
        all_pass = False
    print(f"7 postgres_recovered_row_delta {pg_delta_step7}_restarts_{restarts_after} {'PASS' if step7_ok else 'FAIL'}")

    # -------------------------------------------------------------------------
    # STEP 8: Dynamic engine configuration reload via file mount
    # -------------------------------------------------------------------------
    orig_config_bytes = CONFIG_FILE.read_bytes()
    orig_sha256 = hashlib.sha256(orig_config_bytes).hexdigest()

    probe_snippet = (
        "  - id: phase9_probe\n"
        "    type: llamacpp\n"
        "    url: http://host.docker.internal:11434\n"
        "    tier: free\n"
    )
    CONFIG_FILE.write_text(CONFIG_FILE.read_text(encoding="utf-8") + probe_snippet, encoding="utf-8")

    subprocess.run(["docker", "compose", "restart", "gateway"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait for gateway to become healthy
    probe_found = False
    with httpx.Client(timeout=5.0) as client:
        for _ in range(30):
            try:
                resp = client.get(f"{GATEWAY_URL}/admin/engines", headers={"X-API-Key": api_key})
                if resp.status_code == 200:
                    engine_list = resp.json().get("engines", [])
                    engine_ids = [e.get("id") for e in engine_list]
                    if "phase9_probe" in engine_ids:
                        probe_found = True
                        break
            except Exception:
                pass
            time.sleep(1.0)

    # Restore original file byte-for-byte
    CONFIG_FILE.write_bytes(orig_config_bytes)
    restored_sha256 = hashlib.sha256(CONFIG_FILE.read_bytes()).hexdigest()
    subprocess.run(["docker", "compose", "restart", "gateway"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait healthy after second restart
    with httpx.Client(timeout=5.0) as client:
        for _ in range(30):
            try:
                resp = client.get(f"{GATEWAY_URL}/health")
                if resp.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(1.0)

    step8_ok = probe_found and (orig_sha256 == restored_sha256)
    if not step8_ok:
        all_pass = False
    print(f"8 dynamic_engine_reload probe_{probe_found}_sha_{restored_sha256[:8]} {'PASS' if step8_ok else 'FAIL'}")

    # -------------------------------------------------------------------------
    # STEP 9: 401 Parity for missing vs invalid key
    # -------------------------------------------------------------------------
    status_no_key = 0
    body_no_key = b""
    status_wrong_key = 0
    body_wrong_key = b""

    with httpx.Client(timeout=5.0) as client:
        resp_no = client.post(f"{GATEWAY_URL}/v1/chat/completions", json=payload)
        status_no_key = resp_no.status_code
        body_no_key = resp_no.content

        resp_wrong = client.post(f"{GATEWAY_URL}/v1/chat/completions", json=payload, headers={"X-API-Key": "invalid_wrong_key"})
        status_wrong_key = resp_wrong.status_code
        body_wrong_key = resp_wrong.content

    sha_no = hashlib.sha256(body_no_key).hexdigest()
    sha_wrong = hashlib.sha256(body_wrong_key).hexdigest()
    bodies_equal = (sha_no == sha_wrong)

    step9_ok = (status_no_key == 401) and (status_wrong_key == 401) and bodies_equal
    if not step9_ok:
        all_pass = False
    parity_str = f"{status_no_key}_{status_wrong_key}_{'equal' if bodies_equal else 'different'}"
    print(f"9 unauth_401_parity {parity_str} {'PASS' if step9_ok else 'FAIL'}")

    # -------------------------------------------------------------------------
    # STEP 10: Last JSONL line tenant_id == local_dev
    # -------------------------------------------------------------------------
    last_tenant_id = ""
    if EVENTS_FILE.is_file():
        lines = [line_str for line_str in EVENTS_FILE.read_text(encoding="utf-8").splitlines() if line_str.strip()]
        if lines:
            try:
                last_event = json.loads(lines[-1])
                last_tenant_id = str(last_event.get("tenant_id", ""))
            except Exception as exc:
                print(f"[debug step 10 error]: {exc}", file=sys.stderr)

    step10_ok = (last_tenant_id == "local_dev")
    if not step10_ok:
        all_pass = False
    print(f"10 jsonl_tenant_id {last_tenant_id} {'PASS' if step10_ok else 'FAIL'}")

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
