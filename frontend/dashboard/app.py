"""
GridlockVision AI — Streamlit Dashboard.

Two modes (sidebar):
  - "Judge Demo Mode": upload image/video, watch the pipeline run step
    by step, see evidence generated live.
  - "Analytics Dashboard": totals, trends, heatmap, search, AI insights,
    hotspot prediction.

Run with:
    streamlit run frontend/dashboard/app.py

Talks to the FastAPI backend over HTTP — set API_BASE_URL if not running
on the default localhost:8000.
"""
import os
import time

import requests
import streamlit as st
import pandas as pd
import plotly.express as px

API_BASE_URL = os.getenv("GRIDLOCK_API_URL", "http://localhost:8000")

st.set_page_config(page_title="GridlockVision AI", page_icon="🚦", layout="wide")

st.sidebar.title("🚦 GridlockVision AI")
st.sidebar.caption("Flipkart Gridlock Hackathon 2.0 — Theme 3")
mode = st.sidebar.radio("Mode", ["Judge Demo Mode", "Analytics Dashboard"])


# ---------------------------------------------------------------------------
# Judge Demo Mode
# ---------------------------------------------------------------------------
def render_demo_mode():
    st.title("🎬 Judge Demo Mode")
    st.write(
        "Upload a traffic image or short video clip. The system runs the "
        "full pipeline live: preprocessing → detection → violation rules → "
        "plate OCR → evidence generation → database update."
    )

    col1, col2 = st.columns(2)
    with col1:
        camera_id = st.text_input("Camera ID", value="CAM-DEMO-01")
    with col2:
        st.caption("GPS (demo defaults to New Delhi if left as-is)")
        lat = st.number_input("Latitude", value=28.6139, format="%.4f")
        lon = st.number_input("Longitude", value=77.2090, format="%.4f")

    tab1, tab2 = st.tabs(["📷 Image", "🎥 Video"])

    with tab1:
        img_file = st.file_uploader("Upload traffic image", type=["jpg", "jpeg", "png"], key="img")
        if img_file and st.button("Run Pipeline on Image", type="primary"):
            with st.spinner("Running detection + violation pipeline..."):
                resp = requests.post(
                    f"{API_BASE_URL}/demo/image",
                    files={"file": (img_file.name, img_file.getvalue())},
                    data={"camera_id": camera_id, "gps_lat": lat, "gps_lon": lon},
                )
            if resp.status_code != 200:
                st.error(f"API error: {resp.text}")
            else:
                render_pipeline_result(resp.json())

    with tab2:
        vid_file = st.file_uploader("Upload traffic video clip", type=["mp4", "avi", "mov"], key="vid")
        sample_n = st.slider("Sample every N frames", 1, 20, 5)
        max_frames = st.slider("Max frames to process", 10, 300, 100)
        if vid_file and st.button("Run Pipeline on Video", type="primary"):
            with st.spinner("Processing video... this may take a minute."):
                resp = requests.post(
                    f"{API_BASE_URL}/demo/video",
                    files={"file": (vid_file.name, vid_file.getvalue())},
                    data={
                        "camera_id": camera_id, "gps_lat": lat, "gps_lon": lon,
                        "sample_every_n_frames": sample_n, "max_frames": max_frames,
                    },
                )
            if resp.status_code != 200:
                st.error(f"API error: {resp.text}")
            else:
                data = resp.json()
                st.success(f"Processed {data['total_frames_sampled']} sampled frames — "
                           f"{data['total_violations']} violations found.")
                render_violations_table(data["violations"])


def render_pipeline_result(result: dict):
    steps = result["steps"]

    st.success(f"Pipeline complete — {result['total_detections']} objects detected, "
               f"{result['total_violations']} violations found.")

    with st.expander("Step 1: Preprocessing", expanded=True):
        st.json(steps["1_preprocessing"])

    with st.expander("Step 2: Detection + Tracking"):
        if steps["2_detection"]:
            st.dataframe(pd.DataFrame(steps["2_detection"]))
        else:
            st.info("No road users detected in this frame.")

    with st.expander("Step 3: Signal State (Red Light Logic)"):
        st.write(f"Detected signal state: **{steps['3_signal_state']}**")

    with st.expander("Step 4-6: Violations, Plate OCR & Evidence", expanded=True):
        render_violations_table(result["violations"])


def render_violations_table(violations: list):
    if not violations:
        st.info("No violations detected.")
        return

    for v in violations:
        with st.container(border=True):
            cols = st.columns([1, 2, 2])
            with cols[0]:
                if v.get("evidence_image_path"):
                    try:
                        st.image(v["evidence_image_path"], use_container_width=True)
                    except Exception:
                        st.caption("Evidence image not available for preview in this environment.")
            with cols[1]:
                st.markdown(f"**{v['violation_type'].replace('_', ' ').title()}**")
                st.write(f"Vehicle: {v['vehicle_type']}")
                plate = v.get("plate_number") or "Not readable"
                st.write(f"Plate: `{plate}`")
                st.write(f"Confidence: {v['confidence']:.2f}")
                if v.get("stub_mode"):
                    st.warning("⚠️ This module is in STUB MODE (model not yet trained) — status is UNKNOWN, not a real detection.")
            with cols[2]:
                st.write(f"Risk Score: **{v.get('risk_score', '—')}**")
                st.write(f"Risk Category: {v.get('risk_category', '—')}")
                st.write(f"Enforcement Priority: {v.get('enforcement_priority', '—')}")
                st.caption(f"Violation ID: {v.get('violation_id', '—')}")


# ---------------------------------------------------------------------------
# Analytics Dashboard
# ---------------------------------------------------------------------------
def render_analytics():
    st.title("📊 Analytics Dashboard")

    try:
        summary = requests.get(f"{API_BASE_URL}/analytics/summary").json()
    except requests.exceptions.ConnectionError:
        st.error(f"Could not reach backend API at {API_BASE_URL}. Is FastAPI running?")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Violations", summary["total_violations"])
    c2.metric("Distinct Violation Types", len(summary["violations_by_type"]))
    c3.metric("Cameras Reporting", len(summary["violations_by_camera"]))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Violations by Type")
        if summary["violations_by_type"]:
            df = pd.DataFrame(
                list(summary["violations_by_type"].items()), columns=["type", "count"]
            )
            st.plotly_chart(px.bar(df, x="type", y="count"), use_container_width=True)
        else:
            st.info("No data yet — run Demo Mode to populate the database.")

    with col2:
        st.subheader("Top Offending Vehicles")
        if summary["top_offending_vehicles"]:
            st.dataframe(pd.DataFrame(summary["top_offending_vehicles"]))
        else:
            st.info("No repeat offenders yet.")

    st.subheader("Daily Trend")
    daily = requests.get(f"{API_BASE_URL}/analytics/trends/daily").json()
    if daily:
        df = pd.DataFrame(list(daily.items()), columns=["date", "count"])
        st.plotly_chart(px.line(df, x="date", y="count", markers=True), use_container_width=True)
    else:
        st.info("No daily trend data yet.")

    st.subheader("Hourly Trend")
    hourly = requests.get(f"{API_BASE_URL}/analytics/trends/hourly").json()
    df_h = pd.DataFrame(list(hourly.items()), columns=["hour", "count"])
    st.plotly_chart(px.bar(df_h, x="hour", y="count"), use_container_width=True)

    st.subheader("🗺️ Violation Heatmap")
    heat = requests.get(f"{API_BASE_URL}/analytics/heatmap").json()
    if heat:
        df_heat = pd.DataFrame(heat)
        st.plotly_chart(
            px.density_mapbox(
                df_heat, lat="lat", lon="lon", radius=25, zoom=10,
                mapbox_style="open-street-map",
            ),
            use_container_width=True,
        )
    else:
        st.info("No GPS-tagged violations yet.")

    st.subheader("🤖 AI Traffic Insights")
    insights = requests.get(f"{API_BASE_URL}/analytics/insights").json()["insights"]
    for line in insights:
        st.info(line)

    st.subheader("🔮 Predicted Hotspots")
    hotspots = requests.get(f"{API_BASE_URL}/analytics/hotspots").json()["hotspots"]
    if hotspots:
        st.dataframe(pd.DataFrame(hotspots))
        st.caption("Method: frequency-based heuristic on historical camera-level density. "
                   "Not a trained time-series forecast — see docs for the v2 upgrade path.")
    else:
        st.info("Not enough historical data yet to rank hotspots.")

    st.divider()
    st.subheader("🔍 Search Violations")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        search_plate = st.text_input("Plate number contains")
    with sc2:
        search_type = st.text_input("Violation type (exact, e.g. HELMET_VIOLATION)")
    with sc3:
        search_limit = st.number_input("Max results", 10, 1000, 50)

    if st.button("Search"):
        params = {"limit": search_limit}
        if search_plate:
            params["plate_number"] = search_plate
        if search_type:
            params["violation_type"] = search_type
        results = requests.get(f"{API_BASE_URL}/violations/search", params=params).json()
        st.dataframe(pd.DataFrame(results))


if mode == "Judge Demo Mode":
    render_demo_mode()
else:
    render_analytics()
