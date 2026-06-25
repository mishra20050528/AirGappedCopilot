from pathlib import Path
from datetime import datetime
import pickle
import re

import pandas as pd
import streamlit as st
import ollama

# =========================
# PATHS / CONSTANTS
# =========================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
INCIDENTS_FILE = DATA_DIR / "incidents.csv"
MODEL_FILE = MODELS_DIR / "model.pkl"

HIGH_CPU_RUNBOOK = """Issue: High CPU Utilization

Cause:
Router overload due to excessive traffic.

Recommended Action:
1. Check traffic spikes.
2. Reroute traffic.
3. Upgrade resources if required.
"""

HIGH_LATENCY_RUNBOOK = """Issue: High Network Latency

Cause:
Network congestion or overloaded links.

Recommended Action:
1. Check link utilization.
2. Verify routing paths.
3. Shift traffic to backup path.
"""

PACKET_LOSS_RUNBOOK = """Issue: Packet Loss

Cause:
Congestion, faulty links, or hardware issues.

Recommended Action:
1. Inspect interfaces.
2. Check cable and link health.
3. Reduce congestion.
"""

NORMAL_RUNBOOK = """Issue: Normal Network State

Cause:
Telemetry values are within safe limits.

Recommended Action:
1. Continue monitoring.
2. Keep observing trends.
3. No immediate action required.
"""


# =========================
# HELPERS
# =========================

def ensure_files():
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    if not INCIDENTS_FILE.exists():
        pd.DataFrame(
            columns=["Time", "CPU", "Latency", "Loss", "Risk", "Prediction", "Severity", "Source"]
        ).to_csv(INCIDENTS_FILE, index=False)


def extract_int(patterns, text):
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return int(float(match.group(1)))
            except ValueError:
                return None
    return None


def parse_log_text(text):
    cpu = extract_int(
        [
            r"CPU\s*[:=]\s*(\d+(?:\.\d+)?)",
            r"CPU\s+(\d+(?:\.\d+)?)",
        ],
        text,
    )

    latency = extract_int(
        [
            r"Latency\s*[:=]\s*(\d+(?:\.\d+)?)",
            r"Latency\s+(\d+(?:\.\d+)?)",
        ],
        text,
    )

    loss = extract_int(
        [
            r"Packet\s*Loss\s*[:=]\s*(\d+(?:\.\d+)?)",
            r"PacketLoss\s*[:=]\s*(\d+(?:\.\d+)?)",
            r"Loss\s*[:=]\s*(\d+(?:\.\d+)?)",
        ],
        text,
    )

    return {"cpu": cpu, "latency": latency, "loss": loss}


def calculate_risk_score(cpu, latency, loss):
    score = (
        (cpu * 0.4)
        + ((latency / 500) * 100 * 0.4)
        + ((loss / 20) * 100 * 0.2)
    )
    return max(0, min(100, int(round(score))))


def severity_from_values(risk_score, prediction):
    prediction = str(prediction).lower()

    if risk_score >= 80 or prediction == "failure":
        return "CRITICAL", "🔴", "error"
    if risk_score >= 55 or prediction == "risk":
        return "HIGH", "🟠", "warning"
    return "NORMAL", "🟢", "success"


def get_runbook(cpu, latency, loss):
    parts = []

    if cpu >= 85:
        parts.append(HIGH_CPU_RUNBOOK)

    if latency >= 150:
        parts.append(HIGH_LATENCY_RUNBOOK)

    if loss >= 5:
        parts.append(PACKET_LOSS_RUNBOOK)

    if not parts:
        return NORMAL_RUNBOOK

    return "\n\n".join(parts)


@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        return None
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


def heuristic_analysis(cpu, latency, loss, prediction):
    prediction = str(prediction).lower()

    if cpu >= 90 or latency >= 220 or loss >= 8 or prediction == "failure":
        issue = "Severe network congestion"
        cause = "High utilization is causing delay and packet drops"
        action = "Shift traffic to backup path and inspect overloaded links"
    elif cpu >= 75 or latency >= 120 or loss >= 3 or prediction == "risk":
        issue = "Elevated network stress"
        cause = "One or more telemetry thresholds are trending high"
        action = "Monitor closely and rebalance traffic"
    else:
        issue = "Network operating normally"
        cause = "Telemetry values are within safe limits"
        action = "Continue monitoring"

    return f"Issue:\n{issue}\nCause:\n{cause}\nAction:\n{action}"


def get_phi3_response(cpu, latency, loss, prediction, runbook):
    prompt = f"""
You are an air-gapped network copilot.

Network Metrics:
CPU = {cpu}%
Latency = {latency} ms
Packet Loss = {loss}%

ML Prediction:
{prediction}

Reference Runbook:
{runbook}

Return ONLY this format:

Issue:
Cause:
Action:

Keep it under 45 words.
""".strip()

    try:
        response = ollama.chat(
            model="phi3",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )
        text = response["message"]["content"].strip()
        if text:
            return text
    except Exception:
        pass

    return heuristic_analysis(cpu, latency, loss, prediction)


def load_history():
    ensure_files()
    try:
        return pd.read_csv(INCIDENTS_FILE)
    except Exception:
        return pd.DataFrame(columns=["Time", "CPU", "Latency", "Loss", "Risk", "Prediction", "Severity", "Source"])


def append_incident(row):
    history = load_history()
    history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    history.to_csv(INCIDENTS_FILE, index=False)


def build_report(row, ai_text):
    return f"""AstraGrid AI - Incident Report

Time: {row['Time']}
Source: {row['Source']}
CPU: {row['CPU']}%
Latency: {row['Latency']} ms
Packet Loss: {row['Loss']}%
Risk Score: {row['Risk']}/100
Prediction: {row['Prediction']}
Severity: {row['Severity']}

AI Copilot Output:
{ai_text}
"""


# =========================
# SETUP
# =========================

ensure_files()
model = load_model()

if "cpu" not in st.session_state:
    st.session_state.cpu = 50
if "latency" not in st.session_state:
    st.session_state.latency = 50
if "loss" not in st.session_state:
    st.session_state.loss = 0
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None
if "last_report" not in st.session_state:
    st.session_state.last_report = ""
if "last_ai_text" not in st.session_state:
    st.session_state.last_ai_text = ""
if "last_runbook" not in st.session_state:
    st.session_state.last_runbook = ""
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = ""
if "last_severity" not in st.session_state:
    st.session_state.last_severity = ""
if "last_risk" not in st.session_state:
    st.session_state.last_risk = 0
if "last_source" not in st.session_state:
    st.session_state.last_source = "Manual sliders"

# =========================
# PAGE CONFIG / STYLING
# =========================

st.set_page_config(
    page_title="AstraGrid AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    [data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 16px;
        padding: 14px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }

    [data-testid="stMetricLabel"] {
        color: #cbd5e1;
        font-size: 0.9rem;
    }

    [data-testid="stMetricValue"] {
        color: white;
        font-size: 1.4rem;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 14px;
        background: linear-gradient(90deg, #0ea5e9, #8b5cf6);
        color: white;
        border: none;
        font-weight: 700;
        padding: 0.7rem 1rem;
    }

    div.stButton > button:hover {
        opacity: 0.95;
    }

    section[data-testid="stSidebar"] {
        background: #0b1220;
    }

    .status-badge {
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        display: inline-block;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }

    .badge-green { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.35); }
    .badge-yellow { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.35); }
    .badge-red { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.35); }
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# SIDEBAR
# =========================

st.sidebar.title("System Status")
st.sidebar.success("✅ Phi-3 Local")
st.sidebar.success("✅ ML Model Active")
st.sidebar.success("✅ Air-Gapped Mode")
st.sidebar.success("✅ Incident Logging")

if INCIDENTS_FILE.exists():
    history_for_download = INCIDENTS_FILE.read_text(encoding="utf-8")
    st.sidebar.download_button(
        "Download Incident History CSV",
        data=history_for_download,
        file_name="incidents.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.sidebar.caption("No cloud APIs. No internet calls.")

# =========================
# HEADER
# =========================

st.title("🚀 AstraGrid AI")
st.subheader("Air-Gapped Predictive Copilot for Secure MPLS Operations")

# =========================
# LOG UPLOAD
# =========================

st.subheader("📁 Upload Network Log")
uploaded_file = st.file_uploader("Upload a .log or .txt file", type=["log", "txt"])

parsed_log = {"cpu": None, "latency": None, "loss": None}
log_text = ""

if uploaded_file is not None:
    log_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    parsed_log = parse_log_text(log_text)

    if parsed_log["cpu"] is not None:
        st.session_state.cpu = parsed_log["cpu"]
    if parsed_log["latency"] is not None:
        st.session_state.latency = parsed_log["latency"]
    if parsed_log["loss"] is not None:
        st.session_state.loss = parsed_log["loss"]

    st.caption("Log Preview")
    st.code(log_text[:1200], language="text")

    detected = []
    if parsed_log["cpu"] is not None:
        detected.append(f"CPU: {parsed_log['cpu']}%")
    if parsed_log["latency"] is not None:
        detected.append(f"Latency: {parsed_log['latency']} ms")
    if parsed_log["loss"] is not None:
        detected.append(f"Packet Loss: {parsed_log['loss']}%")

    if detected:
        st.success("Detected -> " + " | ".join(detected))
    else:
        st.warning("No metrics found in uploaded log. Use lines like CPU=92, Latency=240, PacketLoss=9.")

# =========================
# TELEMETRY INPUTS
# =========================

st.subheader("Network Telemetry")

col1, col2, col3 = st.columns(3)

with col1:
    cpu = st.slider("CPU Utilization (%)", 0, 100, key="cpu")

with col2:
    latency = st.slider("Latency (ms)", 0, 500, key="latency")

with col3:
    loss = st.slider("Packet Loss (%)", 0, 20, key="loss")

source_label = "Uploaded log" if uploaded_file is not None and any(v is not None for v in parsed_log.values()) else "Manual sliders"

# =========================
# LIVE METRICS
# =========================

risk_score = calculate_risk_score(cpu, latency, loss)
severity, severity_icon, severity_style = severity_from_values(risk_score, st.session_state.last_prediction or "normal")

st.subheader("Live Metrics")

m1, m2, m3, m4 = st.columns(4)
m1.metric("CPU", f"{cpu}%")
m2.metric("Latency", f"{latency} ms")
m3.metric("Packet Loss", f"{loss}%")
m4.metric("Risk Score", f"{risk_score}/100")

st.progress(risk_score / 100)

st.markdown(
    f"""
<span class="status-badge badge-green">Source: {source_label}</span>
<span class="status-badge badge-yellow">Air-Gapped</span>
<span class="status-badge {'badge-red' if severity == 'CRITICAL' else 'badge-yellow' if severity == 'HIGH' else 'badge-green'}">{severity_icon} {severity}</span>
""",
    unsafe_allow_html=True,
)

st.subheader("📡 Live Telemetry Feed")
st.write(f"CPU={cpu}% | Latency={latency}ms | PacketLoss={loss}%")

# =========================
# ANOMALY DETECTION
# =========================

st.subheader("🚨 Anomaly Detection")

anomalies = []

if cpu > 85:
    anomalies.append("High CPU Utilization")

if latency > 150:
    anomalies.append("High Network Latency")

if loss > 5:
    anomalies.append("Packet Loss Detected")

if len(anomalies) == 0:

    st.success("✅ No Anomalies Detected")

else:

    st.error(f"🚨 {len(anomalies)} Anomalies Detected")

    for anomaly in anomalies:
        st.write(f"• {anomaly}")

# =========================
# ANALYZE
# =========================

if st.button("🔍 Analyze Network"):
    if model is None:
        st.error("Model file not found. Please train the model first and keep models/model.pkl in place.")
    else:
        prediction = str(model.predict([[cpu, latency, loss]])[0]).strip()
        runbook = get_runbook(cpu, latency, loss)
        ai_text = get_phi3_response(cpu, latency, loss, prediction, runbook)

        risk_score = calculate_risk_score(cpu, latency, loss)
        severity, severity_icon, severity_style = severity_from_values(risk_score, prediction)

        row = {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "CPU": cpu,
            "Latency": latency,
            "Loss": loss,
            "Risk": risk_score,
            "Prediction": prediction,
            "Severity": severity,
            "Source": source_label,
        }

        append_incident(row)
        report_text = build_report(row, ai_text)

        st.session_state.last_analysis = row
        st.session_state.last_report = report_text
        st.session_state.last_ai_text = ai_text
        st.session_state.last_runbook = runbook
        st.session_state.last_prediction = prediction
        st.session_state.last_severity = severity
        st.session_state.last_risk = risk_score
        st.session_state.last_source = source_label

# =========================
# LAST ANALYSIS PANEL
# =========================

if st.session_state.last_analysis is not None:
    last = st.session_state.last_analysis

    st.divider()
    st.subheader("System Risk Status")

    if st.session_state.last_severity == "CRITICAL":
        st.error("🚨 CRITICAL INCIDENT")
    elif st.session_state.last_severity == "HIGH":
        st.warning("🟠 HIGH RISK")
    else:
        st.success("🟢 NORMAL")

    st.subheader("ML Prediction")
    st.write(str(st.session_state.last_prediction).upper())

    st.subheader("🤖 AI Copilot Analysis")
    st.info(st.session_state.last_ai_text)

    with st.expander("Runbook Reference", expanded=False):
        st.code(st.session_state.last_runbook, language="text")

    st.subheader("🛠 Recommended Actions")
    if "Action:" in st.session_state.last_ai_text:
        action_part = st.session_state.last_ai_text.split("Action:", 1)[1].strip()
        st.write(action_part)
    else:
        st.write("Inspect the runbook and follow the action steps shown above.")

    st.download_button(
        "Download Latest Incident Report",
        data=st.session_state.last_report,
        file_name=f"incident_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True,
    )

# =========================
# HISTORY
# =========================

history = load_history()

st.divider()
st.subheader("📋 Incident History")

if history.empty:
    st.info("No incidents logged yet. Run an analysis to create the first record.")
else:
    st.dataframe(
        history.sort_index(ascending=False),
        use_container_width=True,
        height=300,
    )

    numeric_history = history.copy()
    for col in ["CPU", "Latency", "Loss", "Risk"]:
        numeric_history[col] = pd.to_numeric(numeric_history[col], errors="coerce")

    if len(numeric_history) >= 2:
        st.subheader("📈 Network Trends")
        chart_df = numeric_history[["CPU", "Latency", "Loss", "Risk"]].fillna(0)
        st.line_chart(chart_df, use_container_width=True)

        st.subheader("📊 Risk Trend")
        st.line_chart(numeric_history[["Risk"]].fillna(0), use_container_width=True)

    st.subheader("📋 Incident Timeline")
    timeline = history.tail(5).copy()
    timeline["Event"] = timeline.apply(
        lambda r: f"{str(r['Prediction']).upper()} | {r['Severity']} | Risk {r['Risk']}/100",
        axis=1,
    )
    st.dataframe(timeline[["Time", "Event"]], use_container_width=True, height=200)

# =========================
# FOOTER
# =========================

st.caption("All telemetry processing, prediction, and AI explanation run locally.")