"""FastAPI service. Start with:  uvicorn api.main:create_app --factory --port 8000

MVP note: no authentication yet; reviewer_id is self-declared. See README > Limitations.
"""

import logging
import os
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from inspectedge.audit.store import AuditStore
from inspectedge.config import AppConfig, load_config
from inspectedge.inference.service import InspectionService, load_service
from inspectedge.inference.uploads import UploadError
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
    if service is None and (art := os.environ.get("INSPECTEDGE_ARTEFACT_DIR")):
        try:
            service = load_service(art, cfg)
        except Exception:  # keep API up; /inspect returns 503 and /health reports it
            log.exception("failed to load model artefact from %s", art)

    app = FastAPI(title="InspectEdge AI", version="0.1.0", description=DISCLAIMER)

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        request.state.request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["x-request-id"] = request.state.request_id
        return response

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
        try:
            result = service.inspect(data, file.content_type)
        except UploadError as e:
            raise HTTPException(e.status_code, str(e)) from e
        store.save_inspection(result)
        store.log_event(
            AuditEvent(
                actor="api",
                action="inspect",
                object_id=result.inspection_id,
                request_id=request.state.request_id,
                result=result.decision.value,
            )
        )
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
