"""Streamlit reviewer dashboard. Run: streamlit run dashboard/app.py (API must be running)."""

import os
from pathlib import Path

import requests
import streamlit as st

API = os.environ.get("INSPECTEDGE_API_URL", "http://localhost:8000")
# Optional folder of demo images (used by the hosted demo, streamlit_app.py).
SAMPLES_DIR = os.environ.get("INSPECTEDGE_SAMPLES_DIR")
DECISION_STYLE = {
    "PASS": ("✅", "success"),
    "MANUAL_REVIEW": ("🟡", "warning"),
    "LIKELY_ANOMALY": ("🔴", "error"),
    "RECAPTURE": ("📷", "info"),
    "OUT_OF_SCOPE": ("🚫", "warning"),
}
DECISION_NOTE = {
    "RECAPTURE": "image quality too poor to assess; retake the photo",
    "OUT_OF_SCOPE": "not assessed: this image does not look like the inspection setup the model "
    "was trained on",
}

st.set_page_config(page_title="InspectEdge AI", page_icon="🔍", layout="wide")
st.title("InspectEdge AI — inspection review")

try:
    health = requests.get(f"{API}/health", timeout=5).json()
except requests.RequestException:
    st.error(f"API not reachable at {API}. Start it with the '3. API' launch config.")
    st.stop()

st.caption(health["disclaimer"])
if not health["model_loaded"]:
    st.warning(
        "API is up but no model is loaded. Run evaluation, then set INSPECTEDGE_ARTEFACT_DIR."
    )
else:
    st.caption(f"Model: `{health['model_version']}` · category: `{health['category']}`")

source = None
samples = sorted(Path(SAMPLES_DIR).glob("*.png")) if SAMPLES_DIR else []
if samples:
    choice = st.selectbox("Try a sample image", ["(none)"] + [p.name for p in samples])
    if choice != "(none)":
        source = (choice, (Path(SAMPLES_DIR) / choice).read_bytes(), "image/png")
uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded:
    source = (uploaded.name, uploaded.getvalue(), uploaded.type)
if source and st.button("Inspect", type="primary", disabled=not health["model_loaded"]):
    name, data, content_type = source
    r = requests.post(f"{API}/inspect", files={"file": (name, data, content_type)}, timeout=120)
    if r.ok:
        st.session_state.update(result=r.json(), image=data)
    else:
        st.error(f"{r.status_code}: {r.json().get('detail')}")

res = st.session_state.get("result")
if res:
    icon, kind = DECISION_STYLE[res["decision"]]
    note = DECISION_NOTE.get(res["decision"], "human review required for final disposition")
    getattr(st, kind)(f"{icon} **{res['decision']}** — {note}")

    c1, c2, c3 = st.columns([1, 1, 1])
    c1.image(st.session_state["image"], caption="Original", width="stretch")
    if res["heatmap_uri"]:
        c2.image(
            requests.get(API + res["heatmap_uri"], timeout=10).content,
            caption="Anomaly heatmap (fixed scale)",
            width="stretch",
        )
    elif res["decision"] == "OUT_OF_SCOPE":
        c2.info(
            f"No heatmap: the model only knows top-down photos of `{res['category']}` parts "
            "from its training setup, so it does not score other images."
        )
    else:
        c2.info("No heatmap: input failed the quality gate.")
    t = res["thresholds"]
    score = res["anomaly_score"]
    c3.metric("Anomaly score", "NOT ASSESSED" if score is None else f"{score:.4f}")
    c3.write(f"Pass < `{t['pass_threshold']:.4f}` ≤ review < `{t['likely_anomaly_threshold']:.4f}`")
    q = res["quality"]
    c3.write(f"Quality: {'passed' if q['passed'] else 'FAILED'}")
    if res.get("scope"):
        sc = res["scope"]
        c3.write(
            f"Scope: {'in scope' if sc['in_scope'] else 'OUT OF SCOPE'} "
            f"(distance `{sc['distance']:.4f}` vs limit `{sc['threshold']:.4f}`)"
        )
    for w in q["warnings"]:
        c3.write(f"- {w}")
    c3.caption(f"Latency {res['latency_ms']:.0f} ms · id `{res['inspection_id']}`")

    st.subheader("Reviewer disposition")
    with st.form("review"):
        reviewer = st.text_input("Reviewer ID", value="reviewer_01")
        disposition = st.selectbox(
            "Disposition",
            ["ACCEPT", "CONFIRMED_DEFECT", "FALSE_ALARM", "RECAPTURE_REQUESTED", "ESCALATE"],
        )
        reason = st.text_input("Reason code (UPPER_SNAKE)", value="VISUAL_CHECK")
        comment = st.text_area("Comment", max_chars=1000)
        if st.form_submit_button("Submit review"):
            r = requests.post(
                f"{API}/reviews",
                json={
                    "inspection_id": res["inspection_id"],
                    "reviewer_id": reviewer,
                    "disposition": disposition,
                    "reason_code": reason,
                    "comment": comment,
                },
                timeout=10,
            )
            st.success("Review saved") if r.ok else st.error(r.text)

    with st.expander("Audit trail"):
        st.json(requests.get(f"{API}/inspections/{res['inspection_id']}", timeout=10).json())
