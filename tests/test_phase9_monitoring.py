from __future__ import annotations

import json
from pathlib import Path
import re

import yaml

from engine_config import load_engine_specs
from telemetry import metrics_registry


def test_grafana_dashboards_parse_and_validate_structure() -> None:
    """Verifies dashboard JSON structure, UIDs, 30s refresh, and datasource UIDs."""
    dashboards_dir = Path("monitoring/grafana/dashboards")
    dashboard_files = {
        "landauer-perf": dashboards_dir / "dashboard_a_performance.json",
        "landauer-sla": dashboards_dir / "dashboard_b_sla.json",
    }

    for expected_uid, path in dashboard_files.items():
        assert path.exists(), f"Dashboard file {path} missing"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data.get("uid") == expected_uid
        assert data.get("refresh") == "30s"

        panels = data.get("panels", [])
        assert len(panels) >= 4, f"Dashboard {expected_uid} should have at least 4 panels"

        for panel in panels:
            ds = panel.get("datasource", {})
            assert ds.get("uid") == "prometheus", (
                f"Panel '{panel.get('title')}' in {expected_uid} has datasource {ds.get('uid')}, expected 'prometheus'"
            )


def test_dashboard_promql_metrics_exist_in_telemetry_registry() -> None:
    """Verifies all metric names referenced in dashboard PromQL expressions exist in telemetry registry."""
    registered_names: set[str] = set()
    for metric in metrics_registry.collect():
        registered_names.add(metric.name)
        registered_names.add(f"{metric.name}_total")
        registered_names.add(f"{metric.name}_bucket")
        registered_names.add(f"{metric.name}_count")
        registered_names.add(f"{metric.name}_sum")
        for sample in metric.samples:
            registered_names.add(sample.name)

    dashboards_dir = Path("monitoring/grafana/dashboards")
    for json_file in dashboards_dir.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for panel in data.get("panels", []):
            for target in panel.get("targets", []):
                expr = target.get("expr", "")
                metric_names = re.findall(r"\b(landauer_[a-zA-Z0-9_]+)\b", expr)
                assert metric_names, f"No landauer metric found in expr: {expr}"
                for m in metric_names:
                    assert m in registered_names, (
                        f"Metric '{m}' from dashboard {json_file.name} not in telemetry registry"
                    )


def test_prometheus_config_valid() -> None:
    """Verifies prometheus.yml parses with target gateway:8080 and 15s scrape interval."""
    prom_path = Path("monitoring/prometheus/prometheus.yml")
    assert prom_path.exists()

    with open(prom_path, "r", encoding="utf-8") as f:
        prom_cfg = yaml.safe_load(f)

    scrape_interval = prom_cfg.get("global", {}).get("scrape_interval")
    assert scrape_interval == "15s"

    scrape_configs = prom_cfg.get("scrape_configs", [])
    gateway_job = next((job for job in scrape_configs if job.get("job_name") == "gateway"), None)
    assert gateway_job is not None, "Scrape job 'gateway' not found"
    assert gateway_job.get("metrics_path") == "/metrics"

    targets = [
        target
        for sc in gateway_job.get("static_configs", [])
        for target in sc.get("targets", [])
    ]
    assert "gateway:8080" in targets


def test_docker_compose_config_valid() -> None:
    """Verifies compose file parses, non-profiled services have healthchecks, and ports bind to 127.0.0.1."""
    compose_path = Path("docker-compose.yml")
    assert compose_path.exists()

    with open(compose_path, "r", encoding="utf-8") as f:
        compose_cfg = yaml.safe_load(f)

    assert compose_cfg.get("name") == "landauer"
    services = compose_cfg.get("services", {})

    for name, srv in services.items():
        if "profiles" in srv:
            continue

        # Check healthcheck
        assert "healthcheck" in srv, f"Active service '{name}' must have a healthcheck"
        hc = srv["healthcheck"]
        assert "test" in hc, f"Service '{name}' healthcheck missing 'test'"

        # Check port binding
        for port in srv.get("ports", []):
            port_str = str(port)
            assert port_str.startswith("127.0.0.1:"), (
                f"Service '{name}' published port '{port_str}' does not start with '127.0.0.1:'"
            )


def test_deploy_engine_config_valid() -> None:
    """Verifies deploy config loads, omits custom_runtime, and routes via host.docker.internal."""
    deploy_cfg_path = Path("deploy/gateway-config/config.yaml")
    assert deploy_cfg_path.exists()

    specs = load_engine_specs(deploy_cfg_path)
    engine_ids = [spec.id for spec in specs]

    assert "custom_runtime" not in engine_ids
    assert len(specs) >= 3

    for spec in specs:
        assert spec.url is not None
        assert "host.docker.internal" in spec.url, (
            f"Engine '{spec.id}' url '{spec.url}' does not contain host.docker.internal"
        )
