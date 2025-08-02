from pathlib import Path
import streamlit as st
import cv2
import tempfile
import os
from collections import deque
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go


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

    model = YOLO("v6.pt")

    # Updated tool definitions
    setup_tools = {"torx", "setter_tool", "tip_tool"}
    inspection_tools = {"caliper", "rough"}
    maintenance_tools = {"hook", "plier"}
    intervention_objects = {"operator", "chips", "open_door"}


    def calculate_visibility(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(100, max(0, int((variance / 80) * 100)))

    def determine_machine_state(detected, timestamp):
        detected_set = set(detected)

        if not detected_set:
            return "Unknown", detected_set

        if setup_tools & detected_set:
            return "Setup", detected_set

        if "on_coolant" in detected_set:
            return "Machining", detected_set

        if "t1" in detected_set and "open_door" in detected_set:
            return "Inspection (Planned)", detected_set

        if "t1" in detected_set and maintenance_tools & detected_set:
            return "Inspection (Unplanned)", detected_set

        if maintenance_tools & detected_set:
            return "Maintenance", detected_set

        return "Idle", detected_set

    def draw_boxes(img, results):
        return results[0].plot()

    def render_status(label, detected):
        return "✅" if label in detected else "❌"

    def display_dashboard(detected, state, visibility):
        st.markdown("### ⚙️ Machine State")
        st.success(state)

        st.markdown("### 🔍 Visibility")
        st.metric(label="Frame Visibility", value=f"{visibility}%")           

        st.markdown("### 🧰 Setup Tools")
        setup_cols = st.columns(len(setup_tools))
        for i, tool in enumerate(sorted(setup_tools)):
            setup_cols[i].markdown(f"**{tool.replace('_', ' ').title()}**<br>{render_status(tool, detected)}", unsafe_allow_html=True)

        st.markdown("### 📏 Inspection Tools")
        insp_cols = st.columns(len(inspection_tools))
        for i, tool in enumerate(sorted(inspection_tools)):
            insp_cols[i].markdown(f"**{tool.replace('_', ' ').title()}**<br>{render_status(tool, detected)}", unsafe_allow_html=True)

        st.markdown("### 🔧 Maintenance Tools")
        maint_cols = st.columns(len(maintenance_tools))
        for i, tool in enumerate(sorted(maintenance_tools)):
            maint_cols[i].markdown(f"**{tool.replace('_', ' ').title()}**<br>{render_status(tool, detected)}", unsafe_allow_html=True)

        st.markdown("### 👋 Intervention")
        intv_cols = st.columns(len(intervention_objects))
        for i, tool in enumerate(sorted(intervention_objects)):
            intv_cols[i].markdown(f"**{tool.replace('_', ' ').title()}**<br>{render_status(tool, detected)}", unsafe_allow_html=True)


    def process_video(path_or_index):
        cap = cv2.VideoCapture(path_or_index)

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

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            state, label_set = determine_machine_state(labels, frame_count)
            annotated = draw_boxes(frame.copy(), results)
            visibility_score = calculate_visibility(frame)

            st.session_state.history_data.append({
                "Timestamp": timestamp,
                "Frame": frame_count,
                "Machine State": state,
                "Visibility %": visibility_score,
                **{label: (label in label_set) for label in sorted(setup_tools | maintenance_tools | intervention_objects | {"open_door", "t1", "rough"})}
            })

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
        df = pd.DataFrame(st.session_state.history_data)

        # --- Step 1: Final Cycle Detection (fehlertolerant)
        final_cycles = []
        i = 0
        min_duration = 10  # Mindestdauer für einen vollständigen Zyklus (in Frames)

        while i < len(df):
            if df.loc[i, "Machine State"] == "Machining":
                start_idx = i
                j = i + 1
                non_machining_streak = 0
                max_tolerated_idle = 10  # z. B. max. 5 Frames Idle/Maintenance dazwischen erlaubt

                while j < len(df):
                    state = df.loc[j, "Machine State"]

                    if state == "Machining":
                        non_machining_streak = 0  # zurücksetzen
                    elif state in {"Idle", "Maintenance", "Unknown"}:
                        non_machining_streak += 1
                        if non_machining_streak > max_tolerated_idle:
                            break
                    else:
                        break

                    j += 1

                end_idx = j - 1
                duration = end_idx - start_idx + 1

                if duration >= min_duration:
                    final_cycles.append({
                        "Cycle #": len(final_cycles) + 1,
                        "Start Frame": df.loc[start_idx, "Frame"],
                        "End Frame": df.loc[end_idx, "Frame"],
                        "Duration (s)": duration
                    })

                i = j
            else:
                i += 1


        # --- Step 2: Robust KPI Calculation ---
        total_time = len(df)
        machining_time = df[df["Machine State"] == "Machining"].shape[0]
        setup_time = df[df["Machine State"] == "Setup"].shape[0]
        maintenance_time = df[df["Machine State"] == "Maintenance"].shape[0]

        # Fallback-Regel, falls Setup oder Maintenance im Videoschnitt fehlen
        min_operating_time = machining_time + max(setup_time, 1) + maintenance_time
        operating_time = min(total_time, min_operating_time)

        availability = operating_time / total_time if total_time else 0
        performance = machining_time / operating_time if operating_time else 0
        quality = 1.0  # angenommen

        oee = availability * performance * quality

        # KPI Display mit Formeln
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Availability", f"{availability * 100:.1f}%")
            st.markdown(
                f"""<div style="font-size: 0.9em; color: gray;">
                = Operating Time / Total Time  
                = {operating_time} / {total_time}
                </div>""",
                unsafe_allow_html=True
            )

        with col2:
            st.metric("Performance", f"{performance * 100:.1f}%")
            st.markdown(
                f"""<div style="font-size: 0.9em; color: gray;">
                = Machining Time / Operating Time  
                = {machining_time} / {operating_time}
                </div>""",
                unsafe_allow_html=True
            )

        with col3:
            st.metric("Quality", f"{quality * 100:.1f}%")
            st.markdown(
                f"""<div style="font-size: 0.9em; color: gray;">
                (Assumed constant)
                </div>""",
                unsafe_allow_html=True
            )

        with col4:
            st.metric("OEE", f"{oee * 100:.1f}%")
            st.markdown(
                f"""<div style="font-size: 0.9em; color: gray;">
                = A × P × Q  
                = {availability:.2f} × {performance:.2f} × {quality:.2f}
                </div>""",
                unsafe_allow_html=True
            )


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
                yaxis_title="Time (frames)",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # ✔️ Nur wenn gültige Zyklen vorhanden sind, fortfahren mit erweiterten KPIs
            # (z. B. Sichtbarkeit, Timeline, Combined Chart, Summary Report)
            # Den kompletten Erweiterungsblock hier wie zuvor ergänzt lassen


       # --- Step 4: Pie Charts ---
        st.markdown("### 🧭 Machine State Summary")
        state_counts = df["Machine State"].value_counts()

    # Grouped machine states for high-level overview
        grouped_counts = {
            "Productive": state_counts.get("Setup", 0) + state_counts.get("Machining", 0),
            "Maintenance": state_counts.get("Maintenance", 0),
            "Downtime": state_counts.get("Idle", 0)
                    + state_counts.get("Inspection (Planned)", 0)
                    + state_counts.get("Inspection (Unplanned)", 0)
                    + state_counts.get("Unknown", 0)
        }   

        productive_split = {
            "Setup": state_counts.get("Setup", 0),
            "Machining": state_counts.get("Machining", 0)
        }

        downtime_counts = {
            "Idle": state_counts.get("Idle", 0),
            "Inspection (Planned)": df[df["open_door"] == True].shape[0] + df[df.get("rough", pd.Series([False]*len(df))) == True].shape[0],
            "Inspection (Unplanned)": state_counts.get("Inspection (Unplanned)", 0),
            "Unknown": state_counts.get("Unknown", 0)
        }

        inspection_split = {
            "Door is Open": df[df["open_door"] == True].shape[0],
            "Surface Roughness Tester": df[df.get("rough", pd.Series([False]*len(df))) == True].shape[0]
        }

        # Row 1: High-level overview (Grouped + Productive breakdown)
        col1, col2 = st.columns(2)
        with col1:
            fig0 = go.Figure(go.Pie(
                labels=list(grouped_counts.keys()),
                values=list(grouped_counts.values()),
                hole=0.3
            ))
            fig0.update_layout(title="All Machine States (Grouped)", height=350)
            st.plotly_chart(fig0, use_container_width=True)

        with col2:
            fig_prod = go.Figure(go.Pie(
                labels=list(productive_split.keys()),
                values=list(productive_split.values()),
                hole=0.3
            ))
            fig_prod.update_layout(title="Productive Time Breakdown", height=350)
            st.plotly_chart(fig_prod, use_container_width=True)

        # Row 2: Downtime + Inspection planned breakdown
        col3, col4 = st.columns(2)
        with col3:
            fig1 = go.Figure(go.Pie(
                labels=list(downtime_counts.keys()),
                values=list(downtime_counts.values()),
                hole=0.3
            ))
            fig1.update_layout(title="Downtime Breakdown", height=350)
            st.plotly_chart(fig1, use_container_width=True)

        with col4:
            fig2 = go.Figure(go.Pie(
                labels=list(inspection_split.keys()),
                values=list(inspection_split.values()),
                pull=[0.1, 0.1],
                hole=0.3
            ))
            fig2.update_layout(title="Inspection (Planned) Details", height=350)
            st.plotly_chart(fig2, use_container_width=True)
                # Row 3: Setup Tools and Maintenance Tools Usage
        col5, col6 = st.columns(2)

        # Setup Tools Breakdown
        with col5:
            setup_tool_counts = {
                tool.replace("_", " ").title(): df[df.get(tool, False) == True].shape[0]
                for tool in setup_tools
            }
            fig_setup = go.Figure(go.Pie(
                labels=list(setup_tool_counts.keys()),
                values=list(setup_tool_counts.values()),
                hole=0.3
            ))
            fig_setup.update_layout(title="Setup Tools Usage", height=350)
            st.plotly_chart(fig_setup, use_container_width=True)

        # Maintenance Tools Breakdown
        with col6:
            maintenance_tool_counts = {
                tool.replace("_", " ").title(): df[df.get(tool, False) == True].shape[0]
                for tool in maintenance_tools
            }
            fig_maint = go.Figure(go.Pie(
                labels=list(maintenance_tool_counts.keys()),
                values=list(maintenance_tool_counts.values()),
                hole=0.3
            ))
            fig_maint.update_layout(title="Maintenance Tools Usage", height=350)
            st.plotly_chart(fig_maint, use_container_width=True)

 # --- Erweiterte Visualisierungen & Bericht ---
            st.markdown("## 🔍 Visibility & Object Behavior")

            if cycle_df.empty:
                st.warning("⚠️ Keine gültigen Zyklen erkannt. Erweiterte Auswertung wird übersprungen.")
            else:
                # --- Frame → Cycle Mapping ---
                df["Cycle #"] = None
                for _, row in cycle_df.iterrows():
                    mask = (df["Frame"] >= row["Start Frame"]) & (df["Frame"] <= row["End Frame"])
                    df.loc[mask, "Cycle #"] = row["Cycle #"]
                df["Cycle #"] = df["Cycle #"].fillna(method="ffill")

                if df["Cycle #"].isnull().all():
                    st.warning("⚠️ Zyklen konnten nicht eindeutig zugewiesen werden.")
                else:
                    # Spalten vorbereiten
                    for label in ["chips", "open_door", "operator"]:
                        if label not in df.columns:
                            df[label] = False

                    # --- Sichtbarkeit ---
                    visibility_by_cycle = df.groupby("Cycle #")["Visibility %"].mean().reset_index()
                    fig_vis = go.Figure()
                    fig_vis.add_trace(go.Scatter(
                        x=visibility_by_cycle["Cycle #"],
                        y=visibility_by_cycle["Visibility %"],
                        mode="lines+markers",
                        name="Visibility %",
                        line=dict(color="royalblue")
                    ))
                    fig_vis.update_layout(
                        xaxis_title="Machining Cycle",
                        yaxis_title="Visibility (%)",
                        yaxis=dict(range=[0, 100]),
                        height=300,
                        margin=dict(l=10, r=10, t=30, b=30)
                    )

                    # --- Timeline ---
                    fig_classes = go.Figure()
                    for label in ["chips", "open_door", "operator"]:
                        frames = df[df[label] == True]["Cycle #"]
                        fig_classes.add_trace(go.Scatter(
                            x=frames,
                            y=[label] * len(frames),
                            mode="markers",
                            name=label,
                            marker=dict(size=6)
                        ))
                    fig_classes.update_layout(
                        xaxis_title="Machining Cycle",
                        yaxis_title="Detected Object",
                        height=300,
                        margin=dict(l=10, r=10, t=30, b=30)
                    )

                    # Zwei Diagramme nebeneinander anzeigen
                    with col5:
                        st.markdown("### 📊 Machine State + Visibility Combined")

                        # Farben für Machine States
                        state_colors = {
                            "Setup": "lightgray", "Machining": "steelblue",
                            "Inspection (Planned)": "lightgreen", "Inspection (Unplanned)": "salmon",
                            "Maintenance": "orange", "Idle": "lightyellow", "Unknown": "lightpink"
                        }

                        # Aggregation vorbereiten
                        cycle_visibility = df.groupby("Cycle #")["Visibility %"].mean().reset_index()
                        cycle_states = df.groupby("Cycle #")["Machine State"].first().reset_index()

                        # Neue Figure mit Balken
                        fig_comb = go.Figure()

                        for state, color in state_colors.items():
                            mask = cycle_states["Machine State"] == state
                            fig_comb.add_trace(go.Bar(
                                x=cycle_states[mask]["Cycle #"],
                                y=[5] * mask.sum(),  # Dummyhöhe für Visualisierung
                                name=state,
                                marker_color=color,
                                opacity=0.8
                            ))

                        # Sichtbarkeitslinie drüberlegen
                        fig_comb.add_trace(go.Scatter(
                            x=cycle_visibility["Cycle #"],
                            y=cycle_visibility["Visibility %"],
                            mode="lines+markers",
                            name="Visibility",
                            yaxis="y2",
                            line=dict(color="black", width=2),
                            marker=dict(size=4)
                        ))

                        # Layout mit zwei Achsen
                        fig_comb.update_layout(
                            height=400,
                            xaxis_title="Machining Cycle",
                            yaxis=dict(title="Machine State", showticklabels=False),
                            yaxis2=dict(title="Visibility (%)", overlaying="y", side="right", range=[0, 100]),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            margin=dict(l=30, r=30, t=30, b=30),
                            plot_bgcolor="white"
                        )

                        st.plotly_chart(fig_comb, use_container_width=True)


                    # --- Recovery Time & Summary Report ---
                    with col6:
                        st.markdown("### 🧾 Summary Report")

                        # Recovery Time Berechnung
                        recovery_times = []
                        for _, row in cycle_df.iterrows():
                            segment = df[df["Frame"] > row["End Frame"]].head(30)
                            recovery_index = segment[segment["Visibility %"] >= 95].index
                            if not recovery_index.empty:
                                frames_needed = recovery_index[0] - segment.index[0]
                                recovery_times.append(frames_needed)

                        avg_visibility = df["Visibility %"].mean()
                        avg_recovery_time = sum(recovery_times) / len(recovery_times) if recovery_times else 0

                        # False Operator Triggers
                        coolant_on = df["on_coolant"] if "on_coolant" in df.columns else pd.Series([False] * len(df))
                        false_triggers = df[(df["operator"] == True) & (coolant_on == True)].shape[0]

                        # Textuelle Zusammenfassung
                        st.markdown(f"""
                        - **Total Machining Cycles**: {len(cycle_df)}  
                        - **Average Visibility**: **{avg_visibility:.1f} %**  
                        - **Average Recovery Time after Machining**: **{avg_recovery_time:.2f} Frames**  
                        - **False Operator Triggers during Coolant-On**: **{false_triggers}**
                        """)

                        # Downloadbarer Bericht
                        summary_text = f"""
                    CNC Process Summary Report
                    ==========================

                    Total Machining Cycles: {len(cycle_df)}
                    Average Visibility: {avg_visibility:.1f} %
                    Average Recovery Time after Machining: {avg_recovery_time:.2f} Frames
                    False Operator Triggers during Coolant-On: {false_triggers}

                    KPI Overview:
                    - Availability = Operating Time / Total Time
                    = {availability:.3f} ({availability*100:.1f}%)
                    - Performance = Machining Time / Operating Time
                    = {performance:.3f} ({performance*100:.1f}%)
                    - Quality = 1.0
                    - OEE = {oee:.3f} ({oee*100:.1f}%)

                    Note:
                    - 'Recovery Time' measures how fast visibility returns to 95% after each machining block.
                    - 'False Operator Triggers' counts how often the model falsely detected operator while coolant was active.
                    """

                        st.download_button(
                            label="📥 Download Summary Report",
                            data=summary_text,
                            file_name="cnc_kpi_summary.txt",
                            mime="text/plain"
                        )
  



        

               