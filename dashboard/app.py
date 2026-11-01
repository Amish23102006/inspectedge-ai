"""Streamlit reviewer dashboard. Run: streamlit run dashboard/app.py (API must be running)."""

import os

import requests
import streamlit as st

API = os.environ.get("INSPECTEDGE_API_URL", "http://localhost:8000")
DECISION_STYLE = {
    "PASS": ("✅", "success"),
    "MANUAL_REVIEW": ("🟡", "warning"),
    "LIKELY_ANOMALY": ("🔴", "error"),
    "RECAPTURE": ("📷", "info"),
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

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded and st.button("Inspect", type="primary", disabled=not health["model_loaded"]):
    data = uploaded.getvalue()
    r = requests.post(
        f"{API}/inspect", files={"file": (uploaded.name, data, uploaded.type)}, timeout=120
    )
    if r.ok:
        st.session_state.update(result=r.json(), image=data)
    else:
        st.error(f"{r.status_code}: {r.json().get('detail')}")

res = st.session_state.get("result")
if res:
    icon, kind = DECISION_STYLE[res["decision"]]
    getattr(st, kind)(f"{icon} **{res['decision']}** — human review required for final disposition")

    c1, c2, c3 = st.columns([1, 1, 1])
    c1.image(st.session_state["image"], caption="Original", use_container_width=True)
    if res["heatmap_uri"]:
        c2.image(
            requests.get(API + res["heatmap_uri"], timeout=10).content,
            caption="Anomaly heatmap (fixed scale)",
            use_container_width=True,
        )
    else:
        c2.info("No heatmap: input failed the quality gate.")
    t = res["thresholds"]
    score = res["anomaly_score"]
    c3.metric("Anomaly score", "NOT AVAILABLE" if score is None else f"{score:.4f}")
    c3.write(f"Pass < `{t['pass_threshold']:.4f}` ≤ review < `{t['likely_anomaly_threshold']:.4f}`")
    q = res["quality"]
    c3.write(f"Quality: {'passed' if q['passed'] else 'FAILED'}")
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
