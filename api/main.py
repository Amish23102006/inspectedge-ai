"""FastAPI service. Start with:  uvicorn api.main:create_app --factory --port 8000

MVP note: no authentication yet; reviewer_id is self-declared. See README > Limitations.

Operational endpoints (used by Kubernetes, see k8s/ and docs/kubernetes.md):
  /health/live   process is up and the event loop answers (liveness + startup probe)
  /health/ready  pod can serve /inspect right now (readiness probe)
  /metrics       Prometheus metrics
"""

import logging
import os
import time
import uuid

import anyio
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response

from inspectedge.audit.store import AuditStore
from inspectedge.config import AppConfig, load_config
from inspectedge.inference.service import InspectionService, load_service
from inspectedge.inference.uploads import UploadError
from inspectedge.observability import HealthState, Metrics, env_flag, env_float
from inspectedge.schemas import (
    DISCLAIMER,
    AuditEvent,
    HumanReview,
    HumanReviewCreate,
    InspectionRecord,
    InspectionResponse,
    utcnow,
)

log = logging.getLogger("inspectedge.api")


def create_app(
    service: InspectionService | None = None,
    store: AuditStore | None = None,
    cfg: AppConfig | None = None,
) -> FastAPI:
    cfg = cfg or load_config(os.environ.get("INSPECTEDGE_CONFIG", "configs/default.yaml"))
    store = store or AuditStore(cfg.paths.db)
    health_state = HealthState(
        require_model=env_flag("INSPECTEDGE_REQUIRE_MODEL", default=False),
        drain_file=os.environ.get("INSPECTEDGE_DRAIN_FILE", "/tmp/inspectedge.drain"),
    )
    # Test hook for Kubernetes startup probes: simulates a slow model load.
    if (delay := env_float("INSPECTEDGE_STARTUP_DELAY_SECONDS", 0.0)) > 0:
        log.warning("INSPECTEDGE_STARTUP_DELAY_SECONDS=%s: delaying startup", delay)
        time.sleep(delay)
    if service is None and (art := os.environ.get("INSPECTEDGE_ARTEFACT_DIR")):
        try:
            service = load_service(art, cfg)
        except Exception:  # keep API up; /inspect returns 503 and /health reports it
            log.exception("failed to load model artefact from %s", art)

    # Each kNN inference can briefly allocate a few hundred MB (patch-to-memory-bank distances),
    # so cap concurrent inferences per pod; extra requests wait in the pod instead of OOM-killing
    # it. Throughput comes from more pods (HPA), not more threads.
    inference_limiter = anyio.CapacityLimiter(
        max(1, int(env_float("INSPECTEDGE_MAX_CONCURRENT_INFERENCES", 2)))
    )
    metrics = Metrics()
    metrics.model_loaded.set(1 if service is not None else 0)
    if service is not None:
        metrics.model_info.labels(service.model_version, service.category).set(1)

    app = FastAPI(title="InspectEdge AI", version="0.2.0", description=DISCLAIMER)
    app.state.metrics = metrics

    @app.middleware("http")
    async def request_id_and_metrics(request: Request, call_next):
        request.state.request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        method = request.method
        t0 = time.perf_counter()
        metrics.in_flight.inc()
        try:
            response = await call_next(request)
        except Exception:
            route = _route_template(request)
            metrics.errors.labels(method, route).inc()
            metrics.requests.labels(method, route, "500").inc()
            raise
        finally:
            metrics.in_flight.dec()
        route = _route_template(request)
        metrics.latency.labels(method, route).observe(time.perf_counter() - t0)
        metrics.requests.labels(method, route, str(response.status_code)).inc()
        if response.status_code >= 500:
            metrics.errors.labels(method, route).inc()
        response.headers["x-request-id"] = request.state.request_id
        return response

    @app.get("/health/live")
    def live() -> dict:
        # Must stay cheap and independent of the model/DB: failing it restarts the container.
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready() -> JSONResponse:
        checks = {
            "model": service is not None or not health_state.require_model,
            "audit_db": store.ping(),
            "not_draining": not health_state.draining(),
        }
        ok = all(checks.values())
        return JSONResponse(
            {"status": "ready" if ok else "not_ready", "checks": checks},
            status_code=200 if ok else 503,
        )

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        body, content_type = metrics.render()
        return Response(body, media_type=content_type)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model_loaded": service is not None,
            "model_version": service.model_version if service else None,
            "category": service.category if service else None,
            "disclaimer": DISCLAIMER,
        }

    @app.post("/inspect", response_model=InspectionResponse)
    async def inspect(
        request: Request,
        file: UploadFile = File(...),
        category: str | None = Form(None),
    ) -> InspectionResponse:
        if service is None:
            raise HTTPException(503, "no registered model loaded; see README 'Serve a model'")
        if category and category != service.category:
            raise HTTPException(
                422, f"model serves category {service.category!r}, not {category!r}"
            )
        data = await file.read(cfg.upload.max_bytes + 1)
        t0 = time.perf_counter()
        try:
            # CPU-bound: run off the event loop so /health/live and /metrics keep answering
            # while a pod is saturated (otherwise liveness probes fail and pods restart).
            result = await anyio.to_thread.run_sync(
                service.inspect, data, file.content_type, limiter=inference_limiter
            )
        except UploadError as e:
            raise HTTPException(e.status_code, str(e)) from e
        metrics.inference_latency.observe(time.perf_counter() - t0)
        metrics.decisions.labels(result.decision.value).inc()
        event = AuditEvent(
            actor="api",
            action="inspect",
            object_id=result.inspection_id,
            request_id=request.state.request_id,
            result=result.decision.value,
        )
        await run_in_threadpool(_persist, store, result, event)
        return result

    @app.post("/reviews", response_model=HumanReview)
    def create_review(body: HumanReviewCreate, request: Request) -> HumanReview:
        if store.get_inspection(body.inspection_id) is None:
            raise HTTPException(404, "inspection not found")
        review = HumanReview(**body.model_dump(), reviewed_at=utcnow())
        store.save_review(review)
        store.log_event(
            AuditEvent(
                actor=f"reviewer:{review.reviewer_id}",
                action="review",
                object_id=review.inspection_id,
                request_id=request.state.request_id,
                result=f"{review.disposition.value}:{review.reason_code}",
            )
        )
        return review

    @app.get("/inspections/{inspection_id}", response_model=InspectionRecord)
    def get_inspection(inspection_id: uuid.UUID) -> InspectionRecord:
        iid = str(inspection_id)
        insp = store.get_inspection(iid)
        if insp is None:
            raise HTTPException(404, "inspection not found")
        return InspectionRecord(
            inspection=insp, reviews=store.list_reviews(iid), audit=store.list_events(iid)
        )

    @app.get("/heatmaps/{inspection_id}")
    def get_heatmap(inspection_id: uuid.UUID) -> FileResponse:
        if service is None:
            raise HTTPException(503, "no model loaded")
        path = service.heatmap_path(str(inspection_id))
        if not path.is_file():
            raise HTTPException(404, "heatmap not found")
        return FileResponse(path, media_type="image/png")

    return app


def _persist(store: AuditStore, result: InspectionResponse, event: AuditEvent) -> None:
    store.save_inspection(result)
    store.log_event(event)


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", None) or "unmatched"
