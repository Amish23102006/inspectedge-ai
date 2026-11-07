"""Prometheus metrics and health state for the API (used by Kubernetes probes and HPA demos).

Metrics use route *templates* (``/inspections/{inspection_id}``), never raw paths, so label
cardinality stays bounded. One uvicorn worker per pod is assumed (no multiprocess mode);
scale out with more pods, not more workers.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Inference on CPU takes tens to hundreds of ms; health checks take < 5 ms.
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


class Metrics:
    """Per-app registry so tests can build many apps without duplicate-metric errors."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        self.requests = Counter(
            "inspectedge_http_requests_total",
            "HTTP requests by method, route template and status code.",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.errors = Counter(
            "inspectedge_http_errors_total",
            "HTTP responses with status >= 500, plus unhandled exceptions.",
            ["method", "route"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "inspectedge_http_request_duration_seconds",
            "End-to-end request latency inside the app.",
            ["method", "route"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.in_flight = Gauge(
            "inspectedge_http_requests_in_flight",
            "Requests currently being processed.",
            registry=self.registry,
        )
        self.inference_latency = Histogram(
            "inspectedge_inference_duration_seconds",
            "Time inside InspectionService.inspect (decode, quality gate, model, heatmap).",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.decisions = Counter(
            "inspectedge_decisions_total",
            "Inspection decisions returned by /inspect.",
            ["decision"],
            registry=self.registry,
        )
        self.model_loaded = Gauge(
            "inspectedge_model_loaded",
            "1 if a model artefact is loaded in this pod, else 0.",
            registry=self.registry,
        )
        self.model_info = Gauge(
            "inspectedge_model_info",
            "Constant 1, labelled with the served model version and category.",
            ["model_version", "category"],
            registry=self.registry,
        )

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST


class HealthState:
    """Inputs to /health/ready. Liveness never depends on any of this."""

    def __init__(self, require_model: bool, drain_file: str | Path) -> None:
        self.require_model = require_model
        self.drain_file = Path(drain_file)
        self.started_at = time.time()

    def draining(self) -> bool:
        # Touching this file inside a pod takes it out of Service endpoints without
        # restarting it (used by scripts/k8s/recovery-test.sh).
        return self.drain_file.exists()


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)
