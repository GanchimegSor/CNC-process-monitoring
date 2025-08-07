# 🛠️ CNC Process Monitoring with Real-Time Object Detection

This project provides a real-time CNC process monitoring application using object detection (**YOLOv8**), integrated with KPI analysis and visibility diagnostics via a user-friendly **Streamlit dashboard**.

---

## 🎯 Purpose

Monitor and evaluate CNC machining operations by:

- Detecting tools, coolant, chips, and operator interactions  
- Classifying machine states (e.g., *Machining*, *Setup*, *Maintenance*, *Inspection (planned/unplanned)*, *Idle* )  
- Visualizing visibility conditions during machining  
- Calculating OEE and other KPIs (*Overall Equipment Effectiveness*, *Key Performance Indicator*)  
- Generating structured machining cycle reports  

---

## ✨ Features

- 📹 Live camera support (USB or uploaded video)  
- 🎯 YOLOv8 model integration for object detection  
- 📊 Cycle recognition with tolerance for noisy classification  
- 🧠 Automatic KPI computation: Availability, Performance, OEE  
- 📈 Visual analytics: cycle durations, visibility trends, timeline overlays  
- 🧾 Auto-generated process summary with download option  
- ⚙️ Built with Streamlit, OpenCV, Plotly, and Ultralytics YOLO  

---

## 📦 Installation

**1. Clone the repository:**
git clone https://github.com/GanchimegSor/CNC-process-monitoring.git
cd cnc-process-monitoring

**2. Create and activate a virtual environment (optional):**

python -m venv venv  
source venv/bin/activate    # On Windows: .\venv\Scripts\activate

**3. Install dependencies:**

pip install -r requirements.txt

**4. Download a YOLOv8 model (e.g., v6.pt) and place it in the project root.**  
You can also train your own model using Roboflow or Ultralytics YOLO.
[Roboflow](https://universe.roboflow.com/stillstand/cnc-process-monitoring-with-object-detection-bwyzo) or [Ultralytics YOLO](https://docs.ultralytics.com/).

---

## 🚀 Usage

**Start the Streamlit app:**

streamlit run main.py

You can now:

- Select between a video upload or USB camera
- Observe real-time detection and KPI updates
- Navigate to:
  - Historical Data tab – live detection log
  - KPI Analysis tab – availability, cycle breakdowns, visibility trends

---

## 🧠 Object Classes (Example)

The model supports the following classes (customizable):

**Setup Tools:** torx, caliper, setter_tool, tip_tool  
**Maintenance Tools:** hook, plier, operator  
**Process Signals:**  
- `t0` → Tool is at home position (outside the workpiece)  
- `t1` → Tool is in machining position (engaged with or close to the workpiece)  
- `on_coolant`, `rough`, `chips`, `open_door`

---

## 📊 KPI Logic

Machine states are derived from object detection as follows. These are primarily based on combinations of detected objects and **tool positioning signals (`t0`, `t1`)**:

| Machine State            | Detected Objects / Conditions                        |
|--------------------------|------------------------------------------------------|
| Setup                    | Setup tools (e.g., torx, caliper) while tool is at `t0` |
| Machining                | Coolant is active (`on_coolant`) while tool is at `t1` |
| Inspection (Planned)     | Door open (`open_door`), roughness inspection at `t1`  |
| Inspection (Unplanned)   | Maintenance tools (hook, plier) at `t1`              |
| Maintenance              | Maintenance tools (hook, plier) at `t0`              |
| Idle / Unknown           | No relevant detections or closed door               |

> `t0` and `t1` are position markers derived from tool detection:
> - **`t0`**: tool is fully retracted or idle (outside workpiece)  
> - **`t1`**: tool is actively positioned for machining or engaged in cutting  

The system tolerates up to **5 consecutive Idle/Unknown frames** to prevent fragmentation of cycles.

## 📁 Project Structure


cnc-process-monitoring/
├── main.py            # Main Python code for Streamlit app
├── v1.pt              # YOLOv8 model weights
├── v2.pt              # YOLOv8 model weights
├── v3.pt              # YOLOv8 model weights
├── v4.pt              # YOLOv8 model weights
├── requirements.txt   # Project dependencies
└── README.md
```


## 🧩 Dependencies

streamlit  
opencv-python  
ultralytics  
pandas  
plotly

Install with:

pip install -r requirements.txt

---

## 📃 License

MIT License – Free to use, modify, and distribute.

---

## 🙌 Acknowledgements

- Ultralytics YOLOv8  
- Streamlit  
- Roboflow  
- Lehrstuhl für Ressourcen- und Energieeffiziente Produktionsmaschinen Prof. Dr. Nico Hanenkamp  (CNC turning machine used for development and evaluation of vision based monitoring system)





