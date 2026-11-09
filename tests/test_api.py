import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from inspectedge.audit.store import AuditStore
from inspectedge.inference.service import InspectionService
from inspectedge.quality.scope import ScopeGate
from tests.conftest import FakeModel, flat_image, noise_image, png_bytes
from tests.test_scope_gate import grey_parts, rgb_mean_extractor


def make_client(cfg, thresholds, score=0.5, with_model=True, scope_gate=None):
    svc = None
    model = FakeModel(score)
    if with_model:
        svc = InspectionService(
            model,
            thresholds,
            "fake-bottle-test",
            "bottle",
            cfg,
            cfg.paths.artefacts / "heatmaps",
            scope_gate=scope_gate,
        )
    app = create_app(service=svc, store=AuditStore(cfg.paths.db), cfg=cfg)
    return TestClient(app), model


def upload(client, img_bytes, ctype="image/png", **form):
    return client.post("/inspect", files={"file": ("x.png", img_bytes, ctype)}, data=form)


@pytest.mark.parametrize(
    ("score", "decision"), [(0.5, "PASS"), (1.5, "MANUAL_REVIEW"), (3.0, "LIKELY_ANOMALY")]
)
def test_inspect_decisions_and_heatmap(cfg, thresholds, score, decision):
    client, _ = make_client(cfg, thresholds, score)
    r = upload(client, png_bytes(noise_image()))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decision"] == decision
    assert body["model_version"] == "fake-bottle-test"
    assert body["human_review_required"] is True
    assert "x-request-id" in r.headers
    hm = client.get(body["heatmap_uri"])
    assert hm.status_code == 200 and hm.headers["content-type"] == "image/png"


def test_blurry_input_is_recapture_and_model_not_called(cfg, thresholds):
    client, model = make_client(cfg, thresholds, score=99.0)
    body = upload(client, png_bytes(flat_image())).json()
    assert body["decision"] == "RECAPTURE"
    assert body["anomaly_score"] is None and body["heatmap_uri"] is None
    assert model.calls == 0


def test_rejects_bad_mime_and_category(cfg, thresholds):
    client, _ = make_client(cfg, thresholds)
    assert upload(client, b"text", ctype="text/plain").status_code == 415
    assert upload(client, png_bytes(noise_image()), category="screw").status_code == 422


def test_no_model_returns_503(cfg, thresholds):
    client, _ = make_client(cfg, thresholds, with_model=False)
    assert client.get("/health").json()["model_loaded"] is False
    assert upload(client, png_bytes(noise_image())).status_code == 503


def test_review_and_audit_trail(cfg, thresholds):
    client, _ = make_client(cfg, thresholds, score=1.5)
    iid = upload(client, png_bytes(noise_image())).json()["inspection_id"]
    review = {
        "inspection_id": iid,
        "reviewer_id": "amish",
        "disposition": "FALSE_ALARM",
        "reason_code": "REFLECTION",
        "comment": "glare on shoulder",
    }
    assert client.post("/reviews", json=review).status_code == 200

    rec = client.get(f"/inspections/{iid}").json()
    assert rec["inspection"]["decision"] == "MANUAL_REVIEW"
    assert rec["reviews"][0]["disposition"] == "FALSE_ALARM"
    assert [e["action"] for e in rec["audit"]] == ["inspect", "review"]


def test_review_unknown_inspection_404(cfg, thresholds):
    client, _ = make_client(cfg, thresholds)
    body = {
        "inspection_id": "00000000-0000-0000-0000-000000000000",
        "reviewer_id": "a",
        "disposition": "ACCEPT",
        "reason_code": "OK_PART",
    }
    assert client.post("/reviews", json=body).status_code == 404


def test_heatmap_path_traversal_rejected(cfg, thresholds):
    client, _ = make_client(cfg, thresholds)
    assert client.get("/heatmaps/..%2F..%2Fetc%2Fpasswd").status_code in (404, 422)
    assert client.get("/heatmaps/not-a-uuid").status_code == 422


def _grey_gate():
    return ScopeGate(rgb_mean_extractor).fit(grey_parts(10), grey_parts(5, seed=1), 1.0, 1.5)


def test_out_of_scope_image_is_not_scored(cfg, thresholds):
    client, model = make_client(cfg, thresholds, score=99.0, scope_gate=_grey_gate())
    img = noise_image()
    img[..., 1:] //= 4  # strongly red-tinted: unlike the grey training "setup"
    body = upload(client, png_bytes(img)).json()
    assert body["decision"] == "OUT_OF_SCOPE"
    assert body["anomaly_score"] is None and body["heatmap_uri"] is None
    assert body["scope"]["in_scope"] is False
    assert body["scope"]["distance"] > body["scope"]["threshold"]
    assert model.calls == 0


def test_in_scope_image_is_scored(cfg, thresholds):
    client, model = make_client(cfg, thresholds, score=0.5, scope_gate=_grey_gate())
    img = np.clip(np.random.default_rng(3).normal(120, 30, (256, 256, 3)), 0, 255)
    body = upload(client, png_bytes(img.astype(np.uint8))).json()
    assert body["decision"] == "PASS" and body["scope"]["in_scope"] is True
    assert model.calls == 1
