from pathlib import Path
import streamlit as st
import cv2
import tempfile
import os
from collections import deque
from ultralytics import YOLO
import pandas as pd
import plotly.graph_objects as go
import numpy as np



# --- Streamlit page config ---
st.set_page_config(layout="wide")
st.title("🛠️ CNC Process Monitoring with Object Detection")

# --- Initialize session state ---
if "frame_buffer" not in st.session_state:
    st.session_state.frame_buffer = deque(maxlen=10)
if "stop_video_flag" not in st.session_state:
    st.session_state.stop_video_flag = False
if "history_data" not in st.session_state:
    st.session_state.history_data = []


# --- Kamera-Suche Funktion mit erklärenden Labels ---
@st.cache_resource
def find_available_cameras(max_index=5):
    labels = {
        0: "0 (Default / Built-in Webcam)",
        1: "1 (First External / USB Camera)",
        2: "2 (Second External / USB Camera)",
    }
    available = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            label = labels.get(i, f"{i} (Camera)")
            available.append(label)
        cap.release()
    return available

# --- Tabs ---
tabs = st.tabs(["Analysis", "Historical data", "KPI analysis"])

with tabs[0]:
    st.sidebar.header("🎥 Video setup")
    input_mode = st.sidebar.radio("Choose your source:", ["Upload Video", "USB Camera"])
    frame_skip = st.sidebar.slider("Frame Skip", 1, 100, 20)
    save_video = st.sidebar.checkbox("Save annotated video")
    rotate_view = st.sidebar.selectbox("Rotate view", ["None", "90° Clockwise", "90° Counterclockwise"])
    quality_factor = st.sidebar.slider(
        "🔬 Quality Factor (0.0 – 1.0)",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.01,
        format="%.2f",
        help="Value between 0 (completely defective) and 1 (perfect quality)"
    )

    quality_reason = st.sidebar.selectbox(
        "📋 Reason for Quality Loss (optional)",
        options=[
            "None",
            "Dimensional deviation",
            "Rework required",
            "Wrong tool used",
            "Damaged part",
            "Other"
        ],
        index=0
    )



    video_file = None
    camera_index = 0
    if input_mode == "Upload Video":
        video_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    elif input_mode == "USB Camera":
        available_cameras = find_available_cameras()
        if available_cameras:
            camera_label = st.sidebar.selectbox("Available Cameras", available_cameras)
            camera_index = int(camera_label.split(" ")[0])  # Extract index from "0 (...)", "1 (...)" etc.
        else:
            st.sidebar.error("❌ No available camera detected.")
            st.stop()

    model = YOLO("v2.pt")

    # Updated tool definitions
    setup_tools = {"torx", "caliper", "setter_tool"}  # tip_tool still in logic but not displayed
    maintenance_unplanned_tools = {"hook", "plier"}
    maintenance_planned_tools = {"operator", "rough"}  # 'rough' will display as Profilometer
    intervention_objects = {"chips", "open_door"}

    # === Tool/group definitions (dashboard + logic) ===
    # Dashboard (what you want to SEE)
    setup_tools = {"torx", "caliper", "setter_tool"}   # tip_tool stays in logic but is hidden here
    maintenance_unplanned_tools = {"hook", "plier"}
    maintenance_planned_tools = {"operator", "rough"}  # 'rough' will display as "Profilometer"
    intervention_objects = {"chips", "open_door"}

    # Logic (what the state resolver USES)
    SETUP_TOOLS = {"torx", "caliper", "setter_tool", "tip_tool"}  # includes tip_tool
    UNPLANNED_MAINT = maintenance_unplanned_tools
    PLANNED_MAINT = maintenance_planned_tools

    # === Helpers (must be defined before process_video) ===
    def calculate_visibility(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(100, max(0, int((variance / 80) * 100)))

    def format_mm_ss_sss(seconds: float) -> str:
        total_ms = int(round(seconds * 1000))
        mm = (total_ms // 60000) % 60
        ss = (total_ms // 1000) % 60
        sss = total_ms % 1000
        return f"{mm:02d}:{ss:02d}:{sss:03d}"

    def determine_machine_state(detected, timestamp):
        # normalize labels to avoid trailing spaces / case issues
        s = {str(lbl).strip().lower() for lbl in detected}
        if not s:
            return "Unknown", s

        # 1) Machining — highest priority (coolant-on alone counts)
        if "on_coolant" in s:
            return "Machining", s

       # 2) Setup (planned): any setup tool (tip_tool included).
        # Coolant is already handled above as Machining.
        if SETUP_TOOLS & s:
            return "Setup", s

        # 3) Maintenance (planned)
        if "t0" in s and (PLANNED_MAINT & s):
            return "Maintenance (Planned)", s
        if "t1" in s and "open_door" in s and (PLANNED_MAINT & s):
            return "Maintenance (Planned)", s

        # 4) Inspection (planned): t1 + open_door, coolant off, no unplanned maint tools
        if ("t1" in s) and ("open_door" in s) and ("on_coolant" not in s) and not (UNPLANNED_MAINT & s):
            return "Inspection (Planned)", s

        # 5) Inspection (unplanned): t1 + {hook, plier}
        if ("t1" in s) and (UNPLANNED_MAINT & s):
            return "Inspection (Unplanned)", s

        # 6) Maintenance (unplanned): t0 + {hook, plier}, coolant off
        if ("t0" in s) and (UNPLANNED_MAINT & s) and ("on_coolant" not in s):
            return "Maintenance (Unplanned)", s

        # 7) Idle
        return "Idle", s



    def draw_boxes(img, results):
        return results[0].plot()

    def render_status(label, detected, display_name=None):
        name = display_name if display_name else label.replace('_', ' ').title()
        return f"**{name}**<br>{'✅' if label in detected else '❌'}"

    def display_dashboard(detected, state, visibility):
        st.markdown("### ⚙️ Machine State")
        st.success(state)

        st.markdown("### 🔍 Visibility")
        st.metric(label="Frame Visibility", value=f"{visibility}%")

        # Setup
        st.markdown("### 🧰 Setup Tools")
        setup_cols = st.columns(len(setup_tools))
        for i, tool in enumerate(sorted(setup_tools)):
            setup_cols[i].markdown(render_status(tool, detected), unsafe_allow_html=True)

        # Maintenance (Unplanned)
        st.markdown("### 🔧 Maintenance (Unplanned) Tools")
        maint_u_cols = st.columns(len(maintenance_unplanned_tools))
        for i, tool in enumerate(sorted(maintenance_unplanned_tools)):
            maint_u_cols[i].markdown(render_status(tool, detected), unsafe_allow_html=True)

        # Maintenance (Planned)
        st.markdown("### 🛠️ Maintenance (Planned) Tools")
        maint_p_cols = st.columns(len(maintenance_planned_tools))
        for i, tool in enumerate(sorted(maintenance_planned_tools)):
            display_name = "Profilometer" if tool == "rough" else None
            maint_p_cols[i].markdown(render_status(tool, detected, display_name), unsafe_allow_html=True)

        # Intervention
        st.markdown("### 👋 Intervention")
        intv_cols = st.columns(len(intervention_objects))
        for i, tool in enumerate(sorted(intervention_objects)):
            intv_cols[i].markdown(render_status(tool, detected), unsafe_allow_html=True)



    def process_video(path_or_index):
        cap = cv2.VideoCapture(path_or_index)
        # Detect and store FPS for video-relative timestamps
        processed_idx = 0  # counts only the sampled frames we append
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > 240:
            fps = 30.0
        st.session_state["detected_fps"] = float(fps)


        # Clear previous data when a new video or camera stream is processed
        st.session_state.history_data = []


        if not cap.isOpened():
            st.error(f"❌ Camera at index {path_or_index} could not be opened.")
            return

        frame_count = 0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None


        col1, col2 = st.columns([3, 2])
        video_placeholder = col1.empty()
        dashboard_placeholder = col2.empty()

        stop_button = st.sidebar.button("🚩 Stop Video", key="stop_button")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # fallback if detection fails


        while cap.isOpened():
            if stop_button:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            if rotate_view == "90° Clockwise":
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotate_view == "90° Counterclockwise":
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            results = model.predict(frame, verbose=False)
            labels = [model.names[int(b.cls)] for r in results for b in r.boxes]

            state, label_set = determine_machine_state(labels, frame_count)
            annotated = draw_boxes(frame.copy(), results)
            visibility_score = calculate_visibility(frame)

        
            # Video-relative time based on processed samples so first row is 00:00:000
            fps = float(st.session_state.get("detected_fps", 30.0))
            elapsed_seconds = (processed_idx * frame_skip) / fps
            video_time_str = format_mm_ss_sss(elapsed_seconds)

            # Compute video-relative timestamp (mm:ss:SSS)
            video_time_str = format_mm_ss_sss(frame_count / float(fps))

            tracked = setup_tools | maintenance_unplanned_tools | maintenance_planned_tools | intervention_objects | {"open_door", "t1", "t0", "on_coolant"}
            st.session_state.history_data.append({
                "Timestamp": video_time_str,
                "Frame": frame_count,
                "Machine State": state,
                "Visibility %": visibility_score,
                **{label: (label in label_set) for label in sorted(tracked)}
            })




            processed_idx += 1





            video_placeholder.image(annotated, channels="BGR", use_container_width=True)
            with dashboard_placeholder.container():
                display_dashboard(label_set, state, visibility_score)

            if save_video:
                if out is None:
                    h, w, _ = annotated.shape
                    filename = f"{Path(path_or_index).stem}_annotated.mp4" if isinstance(path_or_index, str) else "camera_annotated.mp4"
                    out_path = os.path.join(os.path.dirname(str(path_or_index)), filename)
                    out = cv2.VideoWriter(out_path, fourcc, 10, (w, h))
                out.write(annotated)

        cap.release()
        if out:
            out.release()
            st.sidebar.success(f"Video saved to: {out_path}")

    if video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.read())
            temp_path = tmp.name
        process_video(temp_path)
        try:
            os.unlink(temp_path)
        except PermissionError:
            pass
    elif input_mode == "USB Camera":
        process_video(int(camera_index))



with tabs[1]:
    st.markdown("## 📊 Historical Data")
    if st.session_state.history_data:
        df = pd.DataFrame(st.session_state.history_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No data recorded yet.")

with tabs[2]:
    st.markdown("## 📈 KPI Analysis")

    if not st.session_state.history_data:
        st.info("No data recorded yet.")
    else:
        if "df_enriched" in st.session_state:
            df = st.session_state["df_enriched"].copy()
        else:
            df = pd.DataFrame(st.session_state.history_data)

        # Optional: put Timestamp first
        cols = df.columns.tolist()
        if "Timestamp" in cols:
            cols = ["Timestamp"] + [c for c in cols if c != "Timestamp"]
        df = df[cols]

        st.dataframe(df, use_container_width=True)


        # --- Step 1: Final Cycle Detection (auto FPS + auto gap tolerance) ---

        # 1) Get FPS (prefer stored value from processing; else 30)
        def _get_fps_fallback():
            try:
                return float(st.session_state.get("detected_fps", 0)) or 30.0
            except Exception:
                return 30.0

        fps = _get_fps_fallback()

        # 2) Analyse idle gaps to estimate a good gap tolerance (in steps)
        mergeable = {"Idle", "Maintenance", "Unknown"}

        idle_gaps = []
        gap_len = 0
        for state in df["Machine State"]:
            if state in mergeable:
                gap_len += 1
            else:
                if gap_len > 0:
                    idle_gaps.append(gap_len)
                gap_len = 0
        if gap_len > 0:
            idle_gaps.append(gap_len)

        if idle_gaps:
            # Use the 90th percentile of idle gaps as tolerance base (in steps)
            gap_tol = max(1, int(np.percentile(idle_gaps, 90)))
            approx_tol_seconds = (gap_tol * frame_skip) / float(fps)
        else:
            # Fallback: ~8s real-time tolerance scaled by frame_skip
            default_max_idle_seconds = 8.0
            gap_tol = max(1, int((default_max_idle_seconds * float(fps)) / float(frame_skip)))
            approx_tol_seconds = default_max_idle_seconds

        st.caption(
            f"Auto gap tolerance: {gap_tol} steps  ≈ {approx_tol_seconds:.1f}s (FPS={fps:.1f}, frame_skip={frame_skip})"
        )

        # 3) Cycle detection with merged gaps (gaps <= gap_tol are counted as machining)
        final_cycles = []
        i = 0
        min_duration_steps = 3  # ignore tiny blips

        while i < len(df):
            if df.loc[i, "Machine State"] == "Machining":
                start_idx = i
                j = i + 1
                duration_steps = 1  # count first machining row

                while j < len(df):
                    state = df.loc[j, "Machine State"]
                    if state == "Machining":
                        duration_steps += 1
                        j += 1
                        continue

                    # Measure consecutive non-machining run
                    k = j
                    ok_merge = True
                    while k < len(df) and df.loc[k, "Machine State"] != "Machining":
                        if df.loc[k, "Machine State"] not in mergeable:
                            ok_merge = False
                            break
                        k += 1

                    gap_len = k - j  # in steps

                    # End on non-mergeable OR zero-length (safety) OR long gap
                    if not ok_merge or gap_len == 0:
                        break

                    if gap_len <= gap_tol:
                        # Treat gap as machining (include in duration) and continue
                        duration_steps += gap_len
                        j = k
                        continue
                    else:
                        # Long gap ends the cycle
                        break

                end_idx = j - 1  # last included row
                start_frame = int(df.loc[start_idx, "Frame"])
                end_frame   = int(df.loc[end_idx,   "Frame"])

                # real frame span across the cycle (independent of frame_skip)
                frame_span = max(0, end_frame - start_frame)

                # seconds from actual span (no off-by-one, no frame_skip error)
                duration_seconds = frame_span / float(fps)

                # video-relative timestamps
                start_time_str = format_mm_ss_sss(start_frame / float(fps))
                end_time_str   = format_mm_ss_sss(end_frame   / float(fps))

                final_cycles.append({
                    "Cycle #": len(final_cycles) + 1,
                    "Start Time": start_time_str,
                    "End Time": end_time_str,
                    "Duration (s)": round(duration_seconds, 2),
                    "Start Frame": start_frame,
                    "End Frame": end_frame,
                    "Duration (frames)": int(duration_steps),  # sampled rows count (info only)
                })



                i = j
            else:
                i += 1






       # --- Step 2: KPI Calculation (ISO 22400, SECONDS-BASED) ---

        # Build cycle_df if needed
        cycle_df = pd.DataFrame(final_cycles) if len(final_cycles) > 0 else pd.DataFrame(
            columns=["Cycle #","Start Frame","End Frame","Duration (frames)","Duration (s)"]
        )

        fps = float(st.session_state.get("detected_fps", 30.0))
        cycle_apt_sec = float(cycle_df["Duration (s)"].sum()) if not cycle_df.empty else 0.0


        # Sort and compute per-row seconds
        df = df.sort_values("Frame").reset_index(drop=True)
        next_frame = df["Frame"].shift(-1).fillna(df["Frame"])
        delta_sec = (next_frame - df["Frame"]).clip(lower=0) / fps

        # --- ISO buckets (now including Inspection (Planned) in PDT) ---
        POT_sec = float(delta_sec.sum())

        # --- Seconds by state (needed for KPIs and charts) ---
        state_sec = (
            df.groupby("Machine State")
            .apply(lambda g: float(delta_sec.loc[g.index].sum()))
            .to_dict()
        )
        sec = lambda s: state_sec.get(s, 0.0)

        # --- ISO buckets for KPI math (use cycle-based APT) ---
        POT_sec = float(delta_sec.sum())
        PDT_sec = sec("Setup") + sec("Maintenance (Planned)") + sec("Inspection (Planned)")
        PBT_sec = max(0.0, POT_sec - PDT_sec)

        # IMPORTANT: use cycle APT for KPI math
        APT_sec = cycle_apt_sec

        UDT_sec = sec("Maintenance (Unplanned)") + sec("Inspection (Unplanned)") + sec("Idle") + sec("Unknown")

        availability = (APT_sec / PBT_sec) if PBT_sec > 0 else 0.0

        num_cycles = int(cycle_df.shape[0]) if not cycle_df.empty else 0
        # --- Ideal Cycle Time policy ---
        ct_series = cycle_df["Duration (s)"]

        ict_mode = st.selectbox(
            "Ideal Cycle Time (ICT) source",
            ["Fastest cycle", "P10 (fast 10%)", "P20 (fast 20%)", "Median", "Manual"],
            index=0  # default to Fastest
        )

        if ict_mode == "Fastest cycle":
            ideal_ct_sec = float(ct_series.min())
        elif ict_mode == "P10 (fast 10%)":
            ideal_ct_sec = float(ct_series.quantile(0.10))
        elif ict_mode == "P20 (fast 20%)":
            ideal_ct_sec = float(ct_series.quantile(0.20))
        elif ict_mode == "Median":
            ideal_ct_sec = float(ct_series.median())
        else:  # Manual
            ideal_ct_sec = st.number_input("Set Ideal CT [s]", value=float(ct_series.min()), min_value=0.0)

        # Performance (still capped at 1.0)
        performance = min((num_cycles * ideal_ct_sec) / APT_sec, 1.0) if APT_sec > 0 else 0.0

        st.caption(f"ICT = {ideal_ct_sec:.2f}s | APT(cycles) = {APT_sec:.1f}s")


        # --- Quality & OEE ---
        quality = float(quality_factor)
        oee = availability * performance * quality

        # --- Debug caption ---
        st.caption(
            f"POT={POT_sec:.1f}s | PDT={PDT_sec:.1f}s | PBT={PBT_sec:.1f}s | "
            f"APT={APT_sec:.1f}s | UDT={UDT_sec:.1f}s | Cycles={num_cycles} | ICT={ideal_ct_sec:.1f}s"
        )

        # --- Store KPI time components for display ---
        apt_sec = APT_sec  # Actual Production Time
        pbt_sec = PBT_sec  # Planned Busy Time

        # --- KPI Display with formulas ---
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Availability", f"{availability * 100:.1f}%")
            st.markdown(
                f"""<div style="font-size: 0.9em; color: gray;">
                = APT / PBT<br>
                = {apt_sec:.1f}s / {pbt_sec:.1f}s
                </div>""",
                unsafe_allow_html=True
            )

        with col2:
            st.metric("Performance", f"{performance * 100:.1f}%")
            st.markdown(
                f"""<div style="font-size: 0.9em; color: gray;">
                = (Cycles × Ideal CT) / APT<br>
                = ({num_cycles} × {ideal_ct_sec:.1f}s) / {apt_sec:.1f}s
                </div>""",
                unsafe_allow_html=True
            )

        with col3:
            st.metric("Quality", f"{quality * 100:.1f}%")

        with col4:
            st.metric("OEE", f"{oee * 100:.1f}%")


        # --- Step 3: Cycle Table + Chart ---
        st.markdown("### 📊 Cycle Overview")

        cycle_df = pd.DataFrame(final_cycles)

        if cycle_df.empty or "Cycle #" not in cycle_df.columns:
            st.warning("⚠️ Keine gültigen Zyklen erkannt – KPI-Visualisierung übersprungen.")
        else:
            # Tabelle anzeigen
            st.dataframe(cycle_df, use_container_width=True)

            # Balkendiagramm: Dauer pro Zyklus
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=cycle_df["Cycle #"],
                y=cycle_df["Duration (s)"],
                name="Machining Time",
                marker_color='skyblue'
            ))

            avg = cycle_df["Duration (s)"].mean()
            fig.add_trace(go.Scatter(
                x=cycle_df["Cycle #"],
                y=[avg] * len(cycle_df),
                mode="lines",
                name=f"Avg. = {avg:.1f}s",
                line=dict(color="red", dash="dash")
            ))

            fig.update_layout(
                title="Machining Time per Cycle",
                xaxis_title="Cycle #",
                yaxis_title="Time (s)",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # ✔️ Nur wenn gültige Zyklen vorhanden sind, fortfahren mit erweiterten KPIs
            # (z. B. Sichtbarkeit, Timeline, Combined Chart, Summary Report)
            # Den kompletten Erweiterungsblock hier wie zuvor ergänzt lassen


       # --- Updated Tool Sets ---
        setup_tools = {"torx", "caliper", "setter_tool", "tip_tool"}
        maintenance_unplanned_tools = {"hook", "plier"}
        maintenance_planned_tools = {"operator", "rough"}

        def _count_true(df, col):
            # returns how many rows are True in df[col]; 0 if the column doesn't exist
            s = df[col] if col in df.columns else pd.Series(False, index=df.index)
            return int((s == True).sum())
        # --- Tool Counts ---
        setup_tool_counts = {
            tool.replace("_", " ").title(): _count_true(df, tool)
            for tool in setup_tools
        }

        maintenance_planned_counts = {
            ("Profilometer" if tool == "rough" else tool.replace("_", " ").title()): _count_true(df, tool)
            for tool in maintenance_planned_tools
        }


        maintenance_unplanned_counts = {
            tool.replace("_", " ").title(): _count_true(df, tool)
            for tool in maintenance_unplanned_tools
        }



        # --- Step 4: Pie Charts (ISO-aligned, seconds-based) ---
        st.markdown("### 🧭 Machine State Summary (Seconds)")

        # Seconds by state (using delta_sec)
        state_sec = (
            df.groupby("Machine State")
            .apply(lambda g: float(delta_sec.loc[g.index].sum()))
            .to_dict()
        )
        sec = lambda s: state_sec.get(s, 0.0)

        # Exact ISO buckets
        PDT_sec = sec("Setup") + sec("Maintenance (Planned)") + sec("Inspection (Planned)")
        APT_sec = sec("Machining")
        UDT_sec = sec("Maintenance (Unplanned)") + sec("Inspection (Unplanned)") + sec("Idle") + sec("Unknown")
        POT_sec = float(delta_sec.sum())
        PBT_sec = max(0.0, POT_sec - PDT_sec)

        # High-level pie (APT vs PDT vs UDT)
        grouped_secs = {
            "APT (Machining)": APT_sec,
            "PDT (Planned)": PDT_sec,
            "UDT (Unplanned)": UDT_sec,
        }

        # Productive split (for transparency; note Setup/Planned Maint are in PDT)
        productive_split_secs = {
            "Setup (Planned)": sec("Setup"),
            "Machining (APT)": APT_sec,
            "Maintenance (Planned)": sec("Maintenance (Planned)"),
            "Inspection (Planned)": sec("Inspection (Planned)"),
        }

        # Downtime split (unplanned only)
        downtime_secs = {
            "Idle": sec("Idle"),
            "Inspection (Unplanned)": sec("Inspection (Unplanned)"),
            "Maintenance (Unplanned)": sec("Maintenance (Unplanned)"),
            "Unknown": sec("Unknown"),
        }

        # (Plot code stays the same as you just added; feed these dicts into the pies.)

        # Optional caption
        st.caption(
            f"POT={POT_sec:.1f}s | PDT={PDT_sec:.1f}s | PBT={PBT_sec:.1f}s | APT={APT_sec:.1f}s | UDT={UDT_sec:.1f}s"
        )


        setup_tool_counts = {
            tool.replace("_", " ").title(): _count_true(df, tool)
            for tool in setup_tools
        }

        maintenance_unplanned_counts = {
            tool.replace("_", " ").title(): _count_true(df, tool)
            for tool in maintenance_unplanned_tools
        }

        maintenance_planned_counts = {
            ("Profilometer" if tool == "rough" else tool.replace("_", " ").title()): _count_true(df, tool)
            for tool in maintenance_planned_tools
        }



        # --- Plots ---
        col1, col2 = st.columns(2)
        with col1:
            fig0 = go.Figure(go.Pie(
                labels=list(grouped_secs.keys()),
                values=list(grouped_secs.values()),
                hole=0.3
            ))
            fig0.update_layout(title="ISO Buckets (APT vs PDT vs UDT) – Seconds", height=350)
            st.plotly_chart(fig0, use_container_width=True)

        with col2:
            fig_prod = go.Figure(go.Pie(
                labels=list(productive_split_secs.keys()),
                values=list(productive_split_secs.values()),
                hole=0.3
            ))
            fig_prod.update_layout(title="Productive Split (Setup, Machining, Planned Maint.) – Seconds", height=350)
            st.plotly_chart(fig_prod, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig_dt = go.Figure(go.Pie(
                labels=list(downtime_secs.keys()),
                values=list(downtime_secs.values()),
                hole=0.3
            ))
            fig_dt.update_layout(title="Downtime & Inspection Breakdown – Seconds", height=350)
            st.plotly_chart(fig_dt, use_container_width=True)

        with col4:
            fig_setup = go.Figure(go.Pie(
                labels=list(setup_tool_counts.keys()),
                values=list(setup_tool_counts.values()),
                hole=0.3
            ))
            fig_setup.update_layout(title="Setup Tools Usage (frame occurrences)", height=350)
            st.plotly_chart(fig_setup, use_container_width=True)

        col5, col6 = st.columns(2)
        with col5:
            fig_maint_u = go.Figure(go.Pie(
                labels=list(maintenance_unplanned_counts.keys()),
                values=list(maintenance_unplanned_counts.values()),
                hole=0.3
            ))
            fig_maint_u.update_layout(title="Maintenance (Unplanned) Tools Usage (frame occurrences)", height=350)
            st.plotly_chart(fig_maint_u, use_container_width=True)

        with col6:
            fig_maint_p = go.Figure(go.Pie(
                labels=list(maintenance_planned_counts.keys()),
                values=list(maintenance_planned_counts.values()),
                hole=0.3
            ))
            fig_maint_p.update_layout(title="Maintenance (Planned) Tools Usage (frame occurrences)", height=350)
            st.plotly_chart(fig_maint_p, use_container_width=True)


        # Optional caption to confirm alignment with KPI math
        st.caption(
            f"POT={POT_sec:.1f}s | PBT={PBT_sec:.1f}s | APT={APT_sec:.1f}s | PDT={PDT_sec:.1f}s | UDT={UDT_sec:.1f}s"
        )



  # --- Erweiterte Visualisierungen & Bericht ---
        st.markdown("## 🔍 Visibility & Object Behavior")

        if cycle_df.empty:
            st.warning("⚠️ Keine gültigen Zyklen erkannt. Erweiterte Auswertung wird übersprungen.")
        else:
            # --- Frame → Cycle mapping (for per-cycle charts) ---
            df = df.copy()
            df["Cycle #"] = None
            for _, r in cycle_df.iterrows():
                mask = (df["Frame"] >= r["Start Frame"]) & (df["Frame"] <= r["End Frame"])
                df.loc[mask, "Cycle #"] = r["Cycle #"]
            df["Cycle #"] = df["Cycle #"].fillna(method="ffill")

            # Seconds per (Cycle, State) for the stacked % chart
            cycle_state_sec = (
                df.groupby(["Cycle #", "Machine State"])
                .apply(lambda g: float(delta_sec.loc[g.index].sum()))
                .unstack(fill_value=0)
            )
            total_sec_per_cycle = cycle_state_sec.sum(axis=1)
            normalized_sec = (cycle_state_sec.div(total_sec_per_cycle, axis=0) * 100.0)
            normalized_sec = normalized_sec.reindex(sorted(normalized_sec.index))

            # Visibility per cycle for the overlaid line
            vis_by_cycle = (
                df.groupby("Cycle #")["Visibility %"].mean()
                .reindex(normalized_sec.index)
                .reset_index()
            )

            # --- Top row: stacked chart (left) + synced summary (right) ---
            colA, colB = st.columns([2, 1])

            with colA:
                st.markdown("### 📊 Machine State Distribution + Visibility (%)")
                state_order = [
                    "Setup",
                    "Machining",
                    "Inspection (Planned)",
                    "Inspection (Unplanned)",
                    "Maintenance (Planned)",
                    "Maintenance (Unplanned)",
                    "Idle",
                    "Unknown",
                ]

                fig_comb = go.Figure()
                for state in state_order:
                    if state in normalized_sec.columns:
                        fig_comb.add_trace(
                            go.Bar(
                                x=normalized_sec.index,
                                y=normalized_sec[state],
                                name=state,
                            )
                        )

                fig_comb.add_trace(
                    go.Scatter(
                        x=vis_by_cycle["Cycle #"],
                        y=vis_by_cycle["Visibility %"],
                        mode="lines+markers",
                        name="Visibility",
                        yaxis="y2",
                    )
                )

                fig_comb.update_layout(
                    barmode="stack",
                    height=360,
                    xaxis_title="Machining Cycle",
                    yaxis=dict(title="Machine State Share (%)", range=[0, 100]),
                    yaxis2=dict(
                        title="Visibility (%)",
                        overlaying="y",
                        side="right",
                        range=[0, 100],
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=20, r=20, t=30, b=10),
                    plot_bgcolor="white",
                )
                st.plotly_chart(fig_comb, use_container_width=True)

                st.caption(
                    f"POT={POT_sec:.1f}s | PBT={PBT_sec:.1f}s | APT={APT_sec:.1f}s | PDT={PDT_sec:.1f}s | UDT={UDT_sec:.1f}s"
                )

            with colB:
                st.markdown("### 🧾 Summary")

                # Recovery time (frames until visibility ≥95% right after each cycle ends)
                recovery_times = []
                for _, row in cycle_df.iterrows():
                    segment = df[df["Frame"] > row["End Frame"]].head(30)
                    idx = segment[segment["Visibility %"] >= 95].index
                    if len(idx) > 0:
                        recovery_times.append(idx[0] - segment.index[0])
                avg_recovery_time = float(np.mean(recovery_times)) if recovery_times else 0.0

                # False operator triggers while coolant is ON
                coolant_on = df["on_coolant"] if "on_coolant" in df.columns else pd.Series([False]*len(df), index=df.index)
                false_triggers = int(((df.get("operator", False) == True) & (coolant_on == True)).sum())

                avg_visibility = float(df["Visibility %"].mean())
                num_cycles = int(cycle_df.shape[0]) if not cycle_df.empty else 0
                ideal_ct_sec = float(cycle_df["Duration (s)"].median()) if not cycle_df.empty else 0.0  # already used in KPIs

                # IMPORTANT: Use the SAME KPI values you computed above
                st.markdown(
                    f"""
        - **Cycles**: **{num_cycles}**  
        - **Availability**: **{availability*100:.1f}%** &nbsp; (= APT / PBT = {APT_sec:.1f}s / {PBT_sec:.1f}s)  
        - **Performance**: **{performance*100:.1f}%** &nbsp; (= Cycles × Ideal CT / APT = {num_cycles} × {ideal_ct_sec:.1f}s / {APT_sec:.1f}s)  
        - **Quality**: **{quality*100:.1f}%**  
        - **OEE**: **{oee*100:.1f}%**  

        - **POT/PDT/PBT/UDT**: {POT_sec:.1f}s / {PDT_sec:.1f}s / {PBT_sec:.1f}s / {UDT_sec:.1f}s  
        - **Avg. Visibility**: {avg_visibility:.1f}%  
        - **Avg. Recovery**: {avg_recovery_time:.2f} frames  
        - **False operator while coolant-on**: {false_triggers}
        """
                )

                # Downloadable report (synced to the same numbers)
                summary_text = f"""
        CNC Process Summary Report
        ==========================

        Cycles: {num_cycles}
        Avg. Visibility: {avg_visibility:.1f} %
        Avg. Recovery (frames): {avg_recovery_time:.2f}
        False Operator while Coolant-On: {false_triggers}

        KPI Overview (seconds-based, ISO-aligned)
        -----------------------------------------
        POT = {POT_sec:.1f}s
        PDT = {PDT_sec:.1f}s
        PBT = {PBT_sec:.1f}s
        APT = {APT_sec:.1f}s
        UDT = {UDT_sec:.1f}s

        Availability = APT / PBT = {availability:.3f} ({availability*100:.1f}%)
        Performance  = (Cycles × Ideal CT) / APT = {performance:.3f} ({performance*100:.1f}%)
        Quality      = {quality:.3f} ({quality*100:.1f}%)
        OEE          = {oee:.3f} ({oee*100:.1f}%)
        """
                st.download_button(
                    label="📥 Download Summary Report",
                    data=summary_text,
                    file_name="cnc_kpi_summary.txt",
                    mime="text/plain",
                )

            # --- Second row: optional detail views (keep neat) ---
            colL, colR = st.columns(2)

            # Visibility line by cycle
            with colL:
                visibility_by_cycle = df.groupby("Cycle #")["Visibility %"].mean().reset_index()
                fig_vis = go.Figure()
                fig_vis.add_trace(
                    go.Scatter(
                        x=visibility_by_cycle["Cycle #"],
                        y=visibility_by_cycle["Visibility %"],
                        mode="lines+markers",
                        name="Visibility %"
                    )
                )
                fig_vis.update_layout(
                    xaxis_title="Machining Cycle",
                    yaxis_title="Visibility (%)",
                    yaxis=dict(range=[0, 100]),
                    height=280,
                    margin=dict(l=10, r=10, t=30, b=30),
                )
                st.plotly_chart(fig_vis, use_container_width=True)

            # Timeline of a few key detections
            with colR:
                fig_tl = go.Figure()
                for label in ["chips", "open_door", "operator"]:
                    if label not in df.columns:
                        df[label] = False
                    frames = df[df[label] == True]["Cycle #"]
                    fig_tl.add_trace(
                        go.Scatter(
                            x=frames,
                            y=[label] * len(frames),
                            mode="markers",
                            name=label,
                        )
                    )
                fig_tl.update_layout(
                    xaxis_title="Machining Cycle",
                    yaxis_title="Detected Object",
                    height=280,
                    margin=dict(l=10, r=10, t=30, b=30),
                )
                st.plotly_chart(fig_tl, use_container_width=True)
