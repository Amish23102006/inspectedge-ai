"""Health, readiness and metrics endpoints used by Kubernetes probes and Prometheus."""

from fastapi.testclient import TestClient

from api.main import create_app
from inspectedge.audit.store import AuditStore
from inspectedge.inference.service import InspectionService
from tests.conftest import FakeModel, noise_image, png_bytes


def _client(cfg, thresholds, monkeypatch, tmp_path, with_model=True, require_model=True):
    monkeypatch.setenv("INSPECTEDGE_REQUIRE_MODEL", "true" if require_model else "false")
    monkeypatch.setenv("INSPECTEDGE_DRAIN_FILE", str(tmp_path / "drain"))
    svc = None
    if with_model:
        svc = InspectionService(
            FakeModel(0.5), thresholds, "fake-bottle-test", "bottle", cfg, tmp_path / "hm"
        )
    return TestClient(create_app(service=svc, store=AuditStore(cfg.paths.db), cfg=cfg))


def test_live_and_ready_with_model(cfg, thresholds, monkeypatch, tmp_path):
    c = _client(cfg, thresholds, monkeypatch, tmp_path)
    assert c.get("/health/live").json() == {"status": "alive"}
    r = c.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["checks"] == {"model": True, "audit_db": True, "not_draining": True}


def test_not_ready_without_model_when_required(cfg, thresholds, monkeypatch, tmp_path):
    c = _client(cfg, thresholds, monkeypatch, tmp_path, with_model=False)
    assert c.get("/health/live").status_code == 200  # liveness ignores the model
    r = c.get("/health/ready")
    assert r.status_code == 503 and r.json()["checks"]["model"] is False


def test_ready_without_model_when_not_required(cfg, thresholds, monkeypatch, tmp_path):
    c = _client(cfg, thresholds, monkeypatch, tmp_path, with_model=False, require_model=False)
    assert c.get("/health/ready").status_code == 200


def test_drain_file_makes_pod_unready_then_ready_again(cfg, thresholds, monkeypatch, tmp_path):
    c = _client(cfg, thresholds, monkeypatch, tmp_path)
    (tmp_path / "drain").touch()
    r = c.get("/health/ready")
    assert r.status_code == 503 and r.json()["checks"]["not_draining"] is False
    assert c.get("/health/live").status_code == 200
    (tmp_path / "drain").unlink()
    assert c.get("/health/ready").status_code == 200


def test_metrics_count_requests_errors_and_decisions(cfg, thresholds, monkeypatch, tmp_path):
    c = _client(cfg, thresholds, monkeypatch, tmp_path)
    files = {"file": ("x.png", png_bytes(noise_image()), "image/png")}
    assert c.post("/inspect", files=files).status_code == 200
    iid = "00000000-0000-0000-0000-000000000000"
    assert c.get(f"/inspections/{iid}").status_code == 404

    r = c.get("/metrics")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    text = r.text
    assert (
        'inspectedge_http_requests_total{method="POST",route="/inspect",status="200"} 1.0' in text
    )
    # Route templates, not raw IDs, keep label cardinality bounded.
    assert 'route="/inspections/{inspection_id}",status="404"' in text
    assert iid not in text
    assert 'inspectedge_decisions_total{decision="PASS"} 1.0' in text
    assert "inspectedge_inference_duration_seconds_count 1.0" in text
    assert "inspectedge_model_loaded 1.0" in text
    assert 'model_version="fake-bottle-test"' in text


def test_metrics_count_5xx_as_errors(cfg, thresholds, monkeypatch, tmp_path):
    c = _client(cfg, thresholds, monkeypatch, tmp_path, with_model=False, require_model=False)
    files = {"file": ("x.png", png_bytes(noise_image()), "image/png")}
    assert c.post("/inspect", files=files).status_code == 503
    text = c.get("/metrics").text
    assert 'inspectedge_http_errors_total{method="POST",route="/inspect"} 1.0' in text
    assert "inspectedge_model_loaded 0.0" in text


def test_startup_delay_hook(cfg, thresholds, monkeypatch, tmp_path):
    slept = []
    monkeypatch.setenv("INSPECTEDGE_STARTUP_DELAY_SECONDS", "7")
    monkeypatch.setattr("api.main.time.sleep", slept.append)
    _client(cfg, thresholds, monkeypatch, tmp_path)
    assert slept == [7.0]


class SlowModel(FakeModel):
    def predict(self, images):
        import time

        time.sleep(0.8)  # stands in for CPU-bound inference
        return super().predict(images)


def test_liveness_answers_while_inference_runs(cfg, thresholds, monkeypatch, tmp_path):
    import asyncio
    import time

    import httpx

    monkeypatch.setenv("INSPECTEDGE_DRAIN_FILE", str(tmp_path / "drain"))
    svc = InspectionService(SlowModel(0.5), thresholds, "slow", "bottle", cfg, tmp_path / "hm")
    app = create_app(service=svc, store=AuditStore(cfg.paths.db), cfg=cfg)

    async def run() -> tuple[int, float, int]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            files = {"file": ("x.png", png_bytes(noise_image()), "image/png")}
            inspect = asyncio.create_task(c.post("/inspect", files=files))
            await asyncio.sleep(0.2)  # inference is now running
            t0 = time.perf_counter()
            live = await c.get("/health/live")
            live_s = time.perf_counter() - t0
            return live.status_code, live_s, (await inspect).status_code

    live_status, live_s, inspect_status = asyncio.run(run())
    assert live_status == 200 and inspect_status == 200
    assert live_s < 0.4, f"/health/live blocked for {live_s:.2f}s behind inference"
