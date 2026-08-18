
"""
INDUSTRIAL ML
Predictive Maintenance Engine
--------------------------------
A polished Streamlit dashboard for the AI4I 2020 predictive-maintenance
project. It preserves the existing feature-engineered XGBoost deployment
workflow while presenting the project as an industrial ML product.

Expected files beside this app:
    ai4i2020.csv
    xgboost_predictive_maintenance.pkl

Run:
    streamlit run app.py
"""

from pathlib import Path
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance Engine",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "ai4i2020.csv"
MODEL_PATH = BASE_DIR / "xgboost_predictive_maintenance.pkl"

RAW_NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

ENGINEERED_FEATURES = [
    "Heat Dissipation",
    "Power",
    "Overstrain",
]

REQUIRED_FEATURES = [
    "Type",
    *RAW_NUMERICAL_FEATURES,
]

TARGET = "Machine failure"

LEAKAGE_COLUMNS = [
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

WORKFLOW = [
    ("01", "Data Processing"),
    ("02", "Exploratory Analysis"),
    ("03", "Baseline Models"),
    ("04", "Advanced Optimization"),
    ("05", "Model Selection"),
    ("06", "Model Deployment"),
]


# ============================================================
# INDUSTRIAL UI
# ============================================================

st.markdown(
    """
    <style>
    /* ---------- GLOBAL ---------- */
    .stApp {
        background:
            radial-gradient(circle at 75% 0%, rgba(14,165,233,.07), transparent 28%),
            #0B1120;
        color: #F8FAFC;
    }

    .main .block-container {
        max-width: 1450px;
        padding: 2.0rem 3rem 4rem 3rem;
    }

    html, body, [class*="css"] {
        font-family: Inter, "Segoe UI", sans-serif;
    }

    h1, h2, h3, h4, p, label {
        color: #F8FAFC !important;
    }

    /* ---------- SIDEBAR ---------- */
    section[data-testid="stSidebar"] {
        background: #0A101D;
        border-right: 1px solid #1E293B;
    }

    section[data-testid="stSidebar"] > div {
        padding: 1.4rem 1rem;
    }

    .brand {
        padding: .3rem .35rem 1.5rem .35rem;
    }

    .brand-kicker {
        color: #38BDF8;
        font-size: .68rem;
        font-weight: 800;
        letter-spacing: .18em;
        text-transform: uppercase;
    }

    .brand-title {
        color: #F8FAFC;
        font-size: 1.23rem;
        font-weight: 800;
        margin-top: 5px;
    }

    .brand-subtitle {
        color: #94A3B8;
        font-size: .78rem;
        margin-top: 4px;
    }

    .workflow-title {
        color: #64748B;
        font-size: .68rem;
        font-weight: 800;
        letter-spacing: .15em;
        text-transform: uppercase;
        margin: .6rem .35rem .8rem;
    }

    .workflow-item {
        display: flex;
        align-items: center;
        gap: .7rem;
        padding: .62rem .65rem;
        margin: .18rem 0;
        border-radius: 8px;
        color: #94A3B8;
        font-size: .83rem;
        border: 1px solid transparent;
    }

    .workflow-item.active {
        color: #F8FAFC;
        background: rgba(14,165,233,.09);
        border-color: rgba(56,189,248,.20);
    }

    .workflow-number {
        color: #475569;
        font-size: .68rem;
        font-weight: 800;
        width: 23px;
    }

    .workflow-item.active .workflow-number {
        color: #38BDF8;
    }

    .workflow-line {
        color: #334155;
        margin-left: 1rem;
        line-height: .55;
    }

    .status-box {
        margin-top: 1.5rem;
        padding: .9rem;
        border: 1px solid #1E293B;
        border-radius: 9px;
        background: #0F172A;
    }

    .status-label {
        color: #64748B;
        font-size: .64rem;
        font-weight: 800;
        letter-spacing: .15em;
    }

    .status-text {
        color: #CBD5E1;
        font-size: .78rem;
        margin-top: .5rem;
    }

    .status-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #10B981;
        box-shadow: 0 0 8px rgba(16,185,129,.75);
        margin-right: 7px;
    }

    /* ---------- HERO ---------- */
    .hero {
        padding: .4rem 0 2rem 0;
        border-bottom: 1px solid #1E293B;
        margin-bottom: 1.8rem;
    }

    .hero-kicker {
        color: #38BDF8;
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .19em;
        text-transform: uppercase;
        margin-bottom: .35rem;
    }

    .hero-title {
        color: #F8FAFC;
        font-size: 2.65rem;
        line-height: 1.12;
        font-weight: 800;
        margin: 0;
        letter-spacing: -.04em;
    }

    .hero-copy {
        color: #94A3B8;
        max-width: 820px;
        font-size: 1rem;
        line-height: 1.65;
        margin-top: .75rem;
    }

    /* ---------- SECTION ---------- */
    .section-kicker {
        color: #38BDF8;
        font-size: .66rem;
        font-weight: 800;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: .35rem;
    }

    .section-title {
        color: #F8FAFC;
        font-size: 1.55rem;
        font-weight: 750;
        margin-bottom: .25rem;
    }

    .section-copy {
        color: #64748B;
        font-size: .88rem;
        margin-bottom: 1.2rem;
    }

    /* ---------- METRICS ---------- */
    .metric {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 1rem 1.05rem;
        min-height: 105px;
    }

    .metric-label {
        color: #64748B;
        font-size: .64rem;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
    }

    .metric-value {
        color: #F8FAFC;
        font-size: 1.7rem;
        font-weight: 800;
        margin-top: .42rem;
        letter-spacing: -.02em;
    }

    .metric-accent {
        color: #38BDF8;
    }

    /* ---------- ANALYTICS PANELS ---------- */
    .panel {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 1.05rem;
        margin-bottom: 1rem;
    }

    .panel-title {
        color: #E2E8F0;
        font-size: .82rem;
        font-weight: 700;
        margin-bottom: .75rem;
    }

    .callout {
        border-left: 2px solid #38BDF8;
        background: rgba(15,23,42,.8);
        padding: .8rem 1rem;
        color: #94A3B8;
        font-size: .82rem;
        line-height: 1.55;
        margin: .8rem 0 1rem;
    }

    .callout strong {
        color: #CBD5E1;
    }

    .callout.warning {
        border-left-color: #F59E0B;
    }

    .callout.success {
        border-left-color: #10B981;
    }

    /* ---------- PREDICTION ---------- */
    .diagnostic {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 1.5rem;
    }

    .diagnostic-label {
        color: #64748B;
        font-size: .65rem;
        font-weight: 800;
        letter-spacing: .15em;
        text-transform: uppercase;
    }

    .diagnostic-state {
        font-size: 1.65rem;
        font-weight: 800;
        margin-top: .5rem;
    }

    .healthy { color: #34D399; }
    .failure { color: #FB7185; }
    .warning-text { color: #FBBF24; }

    .probability {
        color: #F8FAFC;
        font-size: 2.7rem;
        font-weight: 800;
        letter-spacing: -.04em;
    }

    .probability-caption {
        color: #64748B;
        font-size: .72rem;
    }

    .recommendation {
        border-top: 1px solid #1E293B;
        margin-top: 1.1rem;
        padding-top: 1.1rem;
        color: #94A3B8;
        line-height: 1.55;
        font-size: .84rem;
    }

    /* ---------- BUTTONS ---------- */
    .stButton > button {
        background: #0284C7;
        border: 1px solid #0EA5E9;
        color: white;
        border-radius: 7px;
        font-weight: 750;
        min-height: 42px;
    }

    .stButton > button:hover {
        background: #0369A1;
        border-color: #38BDF8;
    }

    /* ---------- TABS ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #1E293B;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        color: #64748B;
        padding: .7rem 1rem;
        font-size: .8rem;
    }

    .stTabs [aria-selected="true"] {
        color: #F8FAFC !important;
    }

    /* ---------- STREAMLIT INPUTS ---------- */
    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        background: #0F172A;
        border-color: #334155;
        color: #F8FAFC;
    }

    .stFileUploader {
        background: #0F172A;
        border-radius: 9px;
    }

    /* ---------- TABLE ---------- */
    [data-testid="stDataFrame"] {
        border: 1px solid #1E293B;
        border-radius: 8px;
    }

    /* ---------- FOOTER ---------- */
    .footer {
        border-top: 1px solid #1E293B;
        margin-top: 3rem;
        padding-top: 1rem;
        text-align: center;
        color: #475569;
        font-size: .7rem;
        letter-spacing: .03em;
    }

    @media (max-width: 900px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .hero-title {
            font-size: 2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def inject_plotly_theme(fig, height=350):
    fig.update_layout(
        height=height,
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font=dict(color="#CBD5E1"),
        margin=dict(l=20, r=20, t=45, b=20),
        title_font=dict(size=13, color="#E2E8F0"),
        xaxis=dict(
            gridcolor="rgba(148,163,184,.08)",
            zerolinecolor="rgba(148,163,184,.08)",
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,.08)",
            zerolinecolor="rgba(148,163,184,.08)",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94A3B8"),
        ),
    )
    return fig


def metric(label, value, accent=False):
    cls = "metric-accent" if accent else ""
    st.markdown(
        f"""
        <div class="metric">
            <div class="metric-label">{label}</div>
            <div class="metric-value {cls}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(kicker, title, copy=None):
    html = f"""
    <div style="margin-top:1rem;">
        <div class="section-kicker">{kicker}</div>
        <div class="section-title">{title}</div>
    """
    if copy:
        html += f'<div class="section-copy">{copy}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def create_engineered_features(data):
    data = data.copy()

    data["Heat Dissipation"] = (
        data["Process temperature [K]"]
        - data["Air temperature [K]"]
    )

    data["Power"] = (
        2
        * np.pi
        * data["Rotational speed [rpm]"]
        * data["Torque [Nm]"]
        / 60
    )

    data["Overstrain"] = (
        data["Torque [Nm]"]
        * data["Tool wear [min]"]
    )

    return data


@st.cache_data(show_spinner=False)
def load_dataset(path):
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_model(path):
    return joblib.load(path)


def validate_input_data(data):
    return [
        col for col in REQUIRED_FEATURES
        if col not in data.columns
    ]


def prepare_prediction_data(data):
    missing = validate_input_data(data)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = data[REQUIRED_FEATURES].copy()

    for col in RAW_NUMERICAL_FEATURES:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    if result[RAW_NUMERICAL_FEATURES].isnull().any().any():
        bad = result.columns[result.isnull().any()].tolist()
        raise ValueError(
            f"Non-numeric or missing values detected in: {bad}"
        )

    result = create_engineered_features(result)
    return result


def model_predict(model, data):
    prepared = prepare_prediction_data(data)
    pred = model.predict(prepared)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(prepared)[:, 1]
    else:
        prob = np.full(len(prepared), np.nan)

    return prepared, np.asarray(pred), np.asarray(prob)


def probability_label(probability):
    if probability >= 0.70:
        return "HIGH"
    if probability >= 0.30:
        return "MEDIUM"
    return "LOW"


def recommendation(probability):
    if probability >= 0.70:
        return (
            "Immediate maintenance inspection recommended. "
            "The operating conditions indicate an elevated probability "
            "of machine failure."
        )
    if probability >= 0.30:
        return (
            "Monitor the machine closely and consider a targeted "
            "inspection before the next maintenance interval."
        )
    return (
        "Continue operation under normal monitoring. "
        "No elevated failure probability was detected."
    )


def render_prediction_card(probability, prediction):
    failed = int(prediction) == 1

    if failed:
        state = "MACHINE FAILURE RISK"
        state_class = "failure"
    else:
        state = "MACHINE HEALTHY"
        state_class = "healthy"

    risk = probability_label(probability)

    html = f"""<div class="diagnostic">
<div class="diagnostic-label">Diagnostic Result</div>
<div class="diagnostic-state {state_class}">● {state}</div>
<div style="margin-top:1.4rem;">
<div class="diagnostic-label">Failure Probability</div>
<div class="probability">{probability:.1f}%</div>
<div class="probability-caption">Model-estimated probability of the Machine Failure class</div>
</div>
<div style="margin-top:1.1rem;">
<div class="diagnostic-label">Risk Level</div>
<div style="margin-top:.4rem; font-weight:800; color:#CBD5E1;">{risk}</div>
</div>
<div class="recommendation">
<strong style="color:#CBD5E1;">Recommendation</strong><br>
{recommendation(probability)}
</div>
</div>"""

    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# LOAD DATA + MODEL
# ============================================================

dataset = None
model = None
data_error = None
model_error = None

try:
    if DATA_PATH.exists():
        dataset = load_dataset(DATA_PATH)
    else:
        data_error = f"Dataset not found: {DATA_PATH.name}"
except Exception as exc:
    data_error = str(exc)

try:
    if MODEL_PATH.exists():
        model = load_model(MODEL_PATH)
    else:
        model_error = f"Model not found: {MODEL_PATH.name}"
except Exception as exc:
    model_error = str(exc)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div class="brand">
        <div class="brand-kicker">Industrial ML</div>
        <div class="brand-title">Predictive Maintenance</div>
        <div class="brand-subtitle">Machine Failure Detection</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    '<div class="workflow-title">Machine Learning Workflow</div>',
    unsafe_allow_html=True,
)

page_names = [x[1] for x in WORKFLOW]
page = st.sidebar.radio(
    "Workflow",
    page_names,
    label_visibility="collapsed",
)

active_idx = page_names.index(page)

for i, (number, name) in enumerate(WORKFLOW):
    active = "active" if i == active_idx else ""
    st.sidebar.markdown(
        f"""
        <div class="workflow-item {active}">
            <span class="workflow-number">{number}</span>
            <span>{name}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if i < len(WORKFLOW) - 1:
        st.sidebar.markdown(
            '<div class="workflow-line">↓</div>',
            unsafe_allow_html=True,
        )

st.sidebar.markdown(
    f"""
    <div class="status-box">
        <div class="status-label">MODEL STATUS</div>
        <div class="status-text">
            <span class="status-dot"></span>
            {"Tuned XGBoost model loaded" if model is not None else "Model unavailable"}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

uploaded_dataset = st.sidebar.file_uploader(
    "Replace dataset",
    type=["csv"],
    help="Optional. Upload another AI4I-compatible dataset.",
)

if uploaded_dataset is not None:
    try:
        dataset = pd.read_csv(uploaded_dataset)
        data_error = None
    except Exception as exc:
        data_error = str(exc)


# ============================================================
# GLOBAL HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Industrial Machine Learning</div>
        <div class="hero-title">Predictive Maintenance Engine</div>
        <div class="hero-copy">
            Detect potential machine failures using a feature-engineered
            XGBoost classification model trained on industrial sensor data.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA & EDA
# ============================================================

def page_data_eda():
    section_header(
        "Data & Exploratory Analysis",
        "Dataset Overview",
        "Understand the operating data before interpreting the model.",
    )

    if dataset is None:
        st.error(data_error or "Dataset unavailable.")
        return

    n_rows, n_cols = dataset.shape
    failure_rate = (
        dataset[TARGET].mean() * 100
        if TARGET in dataset.columns
        else np.nan
    )
    missing = int(dataset.isnull().sum().sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric("Observations", f"{n_rows:,}", accent=True)
    with c2:
        metric("Features", f"{n_cols:,}")
    with c3:
        metric(
            "Failure Rate",
            f"{failure_rate:.2f}%" if not np.isnan(failure_rate) else "N/A",
        )
    with c4:
        metric("Missing Values", f"{missing:,}")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    tab_overview, tab_distributions, tab_relationships = st.tabs(
        ["Dataset Preview", "Sensor Distributions", "Failure Patterns"]
    )

    with tab_overview:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-title">Dataset Preview</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            dataset.head(12),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if TARGET in dataset.columns:
            counts = dataset[TARGET].value_counts().sort_index()
            chart = px.bar(
                x=["No Failure", "Machine Failure"],
                y=[counts.get(0, 0), counts.get(1, 0)],
                labels={"x": "", "y": "Machines"},
                text=[counts.get(0, 0), counts.get(1, 0)],
            )
            chart.update_traces(
                marker_color=["#334155", "#38BDF8"],
                textposition="outside",
            )
            inject_plotly_theme(chart, 340)
            chart.update_layout(title="Machine Failure Distribution")
            st.plotly_chart(chart, use_container_width=True)

            st.markdown(
                f"""
                <div class="callout warning">
                    <strong>Class imbalance detected.</strong><br>
                    Machine failures represent <strong>{failure_rate:.2f}%</strong>
                    of the dataset. This is why Recall, F1 and F2 are more
                    informative than Accuracy alone.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_distributions:
        available = [
            c for c in RAW_NUMERICAL_FEATURES
            if c in dataset.columns
        ]

        selected = st.selectbox(
            "Select sensor feature",
            available,
            key="distribution_feature",
        )

        fig = px.histogram(
            dataset,
            x=selected,
            nbins=35,
            marginal="box",
        )
        fig.update_traces(marker_color="#38BDF8")
        inject_plotly_theme(fig, 390)
        fig.update_layout(title=f"{selected} Distribution")
        st.plotly_chart(fig, use_container_width=True)

        cols = st.columns(2)
        for i, feature in enumerate(available[:4]):
            with cols[i % 2]:
                small = px.histogram(dataset, x=feature, nbins=25)
                small.update_traces(marker_color="#0EA5E9")
                inject_plotly_theme(small, 290)
                small.update_layout(title=feature)
                st.plotly_chart(small, use_container_width=True)

    with tab_relationships:
        if TARGET not in dataset.columns:
            st.info("Machine failure target is required for failure-pattern analysis.")
            return

        type_col, rate_col = st.columns(2)

        with type_col:
            if "Type" in dataset.columns:
                type_counts = dataset["Type"].value_counts().reset_index()
                type_counts.columns = ["Type", "Count"]
                fig = px.bar(
                    type_counts,
                    x="Type",
                    y="Count",
                    text="Count",
                )
                fig.update_traces(marker_color="#38BDF8")
                inject_plotly_theme(fig, 340)
                fig.update_layout(title="Product Type Distribution")
                st.plotly_chart(fig, use_container_width=True)

        with rate_col:
            if "Type" in dataset.columns:
                rates = (
                    dataset.groupby("Type")[TARGET]
                    .mean()
                    .mul(100)
                    .reset_index(name="Failure Rate")
                )
                fig = px.bar(
                    rates,
                    x="Type",
                    y="Failure Rate",
                    text="Failure Rate",
                )
                fig.update_traces(marker_color="#0EA5E9")
                inject_plotly_theme(fig, 340)
                fig.update_layout(title="Failure Rate by Product Type")
                st.plotly_chart(fig, use_container_width=True)

        numerical = [
            c for c in RAW_NUMERICAL_FEATURES
            if c in dataset.columns
        ]
        if len(numerical) > 1:
            corr = dataset[numerical + [TARGET]].corr()
            fig = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale=[
                    [0, "#1E3A5F"],
                    [.5, "#0F172A"],
                    [1, "#38BDF8"],
                ],
                zmin=-1,
                zmax=1,
            )
            inject_plotly_theme(fig, 500)
            fig.update_layout(title="Sensor Correlation with Machine Failure")
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# MODEL EVALUATION
# ============================================================

def page_model_evaluation():
    section_header(
        "Model Evaluation",
        "Model Development Journey",
        "The final model is the result of a staged engineering workflow.",
    )

    stages = [
        (
            "01",
            "Baseline Models",
            "Original sensor measurements establish the initial predictive baseline.",
        ),
        (
            "02",
            "Feature Engineering",
            "Heat Dissipation, Power and Overstrain expose physically meaningful operating relationships.",
        ),
        (
            "03",
            "SMOTE Evaluation",
            "Synthetic minority examples were tested to determine whether class balancing improved failure detection.",
        ),
        (
            "04",
            "Final Selection",
            "Feature-engineered XGBoost was selected for its balance of Recall and false-alarm control.",
        ),
    ]

    cols = st.columns(4)
    for col, (num, title, body) in zip(cols, stages):
        with col:
            st.markdown(
                f"""
                <div class="panel" style="min-height:190px;">
                    <div class="section-kicker">{num}</div>
                    <div style="font-size:1rem;font-weight:750;
                                color:#E2E8F0;margin-bottom:.55rem;">
                        {title}
                    </div>
                    <div style="color:#64748B;font-size:.78rem;line-height:1.55;">
                        {body}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="callout success">
            <strong>Production model:</strong>
            Feature-engineered XGBoost without SMOTE. F2-score is emphasized
            because missing a genuine machine failure is more costly than
            generating an additional inspection alert.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown(
            '<div class="panel"><div class="panel-title">Engineered Features</div>',
            unsafe_allow_html=True,
        )
        engineering = pd.DataFrame(
            {
                "Feature": ENGINEERED_FEATURES,
                "Engineering Meaning": [
                    "Process temperature − Air temperature",
                    "Mechanical power from rotational speed and torque",
                    "Torque × Tool wear",
                ],
            }
        )
        st.dataframe(
            engineering,
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div class="panel"><div class="panel-title">XGBoost Optimization</div>',
            unsafe_allow_html=True,
        )
        tuning = pd.DataFrame(
            {
                "Hyperparameter": [
                    "n_estimators",
                    "max_depth",
                    "learning_rate",
                    "subsample",
                    "colsample_bytree",
                ],
                "Search Values": [
                    "[100, 200]",
                    "[3, 5, 7]",
                    "[0.01, 0.1]",
                    "[0.8, 1.0]",
                    "[0.8, 1.0]",
                ],
            }
        )
        st.dataframe(
            tuning,
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if dataset is not None and model is not None and TARGET in dataset.columns:
        st.markdown(
            '<div class="panel"><div class="panel-title">Model Diagnostic View</div>',
            unsafe_allow_html=True,
        )

        try:
            # IMPORTANT:
            # The saved production model may have been fitted on the full
            # training dataset. Therefore these numbers are explicitly
            # labelled as diagnostic/in-sample, not held-out test metrics.
            sample = dataset[REQUIRED_FEATURES].copy()
            prepared = create_engineered_features(sample)
            pred = model.predict(prepared)

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(prepared)[:, 1]
            else:
                prob = None

            diagnostic = pd.DataFrame(
                {
                    "Metric": ["Accuracy", "Precision", "Recall", "F1", "F2"],
                    "Value": [
                        accuracy_score(dataset[TARGET], pred),
                        precision_score(dataset[TARGET], pred, zero_division=0),
                        recall_score(dataset[TARGET], pred, zero_division=0),
                        f1_score(dataset[TARGET], pred, zero_division=0),
                        fbeta_score(dataset[TARGET], pred, beta=2, zero_division=0),
                    ],
                }
            )

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            for col, row in zip(
                [mc1, mc2, mc3, mc4, mc5],
                diagnostic.itertuples(index=False),
            ):
                with col:
                    metric(row.Metric, f"{row.Value:.3f}", accent=row.Metric == "F2")

            st.markdown(
                """
                <div class="callout warning">
                    These values are an <strong>in-sample diagnostic view</strong>
                    because the deployment artifact does not expose the original
                    held-out test set. They must not be presented as independent
                    test performance.
                </div>
                """,
                unsafe_allow_html=True,
            )

            cm = confusion_matrix(dataset[TARGET], pred)
            fig = px.imshow(
                cm,
                text_auto=True,
                x=["No Failure", "Failure"],
                y=["No Failure", "Failure"],
                labels={"x": "Predicted", "y": "Actual", "color": "Machines"},
                color_continuous_scale=[
                    [0, "#0F172A"],
                    [1, "#0284C7"],
                ],
            )
            inject_plotly_theme(fig, 390)
            fig.update_layout(title="Diagnostic Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as exc:
            st.warning(f"Model diagnostic could not be generated: {exc}")

        st.markdown("</div>", unsafe_allow_html=True)

    elif model is None:
        st.warning(
            "The model artifact is not available. Place "
            "`xgboost_predictive_maintenance.pkl` beside `app.py`."
        )


# ============================================================
# DEPLOYMENT
# ============================================================

def page_deployment():
    section_header(
        "Live Prediction Engine",
        "Machine Diagnostic Console",
        "Enter operating conditions or score a batch of machines.",
    )

    if model is None:
        st.error(
            "Tuned XGBoost model is unavailable. "
            "Place `xgboost_predictive_maintenance.pkl` beside `app.py`."
        )
        return

    mode = st.radio(
        "Prediction input",
        ["Manual machine", "Batch CSV"],
        horizontal=True,
    )

    if mode == "Manual machine":
        left, right = st.columns([1.05, .95])

        with left:
            st.markdown(
                '<div class="panel"><div class="panel-title">Machine Parameters</div>',
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2)

            with c1:
                product_type = st.selectbox(
                    "Product Type",
                    ["L", "M", "H"],
                )
                air_temp = st.number_input(
                    "Air temperature [K]",
                    value=298.0,
                    step=.1,
                    format="%.1f",
                )
                rpm = st.number_input(
                    "Rotational speed [rpm]",
                    value=1500,
                    min_value=0,
                    step=10,
                )

            with c2:
                process_temp = st.number_input(
                    "Process temperature [K]",
                    value=308.0,
                    step=.1,
                    format="%.1f",
                )
                torque = st.number_input(
                    "Torque [Nm]",
                    value=40.0,
                    min_value=0.0,
                    step=.5,
                    format="%.1f",
                )
                tool_wear = st.number_input(
                    "Tool wear [min]",
                    value=100,
                    min_value=0,
                    step=1,
                )

            run = st.button(
                "RUN DIAGNOSTIC",
                use_container_width=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            if run:
                record = pd.DataFrame(
                    [
                        {
                            "Type": product_type,
                            "Air temperature [K]": air_temp,
                            "Process temperature [K]": process_temp,
                            "Rotational speed [rpm]": rpm,
                            "Torque [Nm]": torque,
                            "Tool wear [min]": tool_wear,
                        }
                    ]
                )

                try:
                    prepared, pred, prob = model_predict(model, record)
                    probability = float(prob[0] * 100)

                    render_prediction_card(
                        probability,
                        int(pred[0]),
                    )

                    st.markdown(
                        '<div class="panel"><div class="panel-title">Derived Operating Signals</div>',
                        unsafe_allow_html=True,
                    )

                    derived = prepared[ENGINEERED_FEATURES].T.reset_index()
                    derived.columns = ["Signal", "Value"]
                    derived["Value"] = derived["Value"].round(3)

                    st.dataframe(
                        derived,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.markdown("</div>", unsafe_allow_html=True)

                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
            else:
                st.markdown(
                    """
                    <div class="diagnostic">
                        <div class="diagnostic-label">Diagnostic Console</div>
                        <div style="font-size:1.2rem;font-weight:750;
                                    color:#CBD5E1;margin-top:.55rem;">
                            Awaiting machine parameters
                        </div>
                        <div style="color:#64748B;font-size:.82rem;
                                    line-height:1.55;margin-top:.6rem;">
                            Enter the operating conditions and run the diagnostic
                            to estimate the machine's probability of failure.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    else:
        st.markdown(
            """
            <div class="callout">
                Upload a CSV containing <strong>Type</strong> and the five
                original sensor measurements. Machine Failure is not required
                for deployment data.
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            "Upload machine sensor data",
            type=["csv"],
            key="deployment_csv",
        )

        if uploaded is None:
            return

        try:
            batch = pd.read_csv(io.BytesIO(uploaded.getvalue()))
            missing = validate_input_data(batch)

            if missing:
                st.error(
                    "The uploaded CSV is missing: "
                    + ", ".join(missing)
                )
                return

            prepared, pred, prob = model_predict(model, batch)

            output = batch[REQUIRED_FEATURES].copy()
            output["Prediction"] = np.where(
                pred == 1,
                "Machine Failure",
                "No Failure",
            )
            output["Failure Probability (%)"] = (
                prob * 100
            ).round(2)

            total = len(output)
            failures = int((pred == 1).sum())
            healthy = total - failures

            c1, c2, c3 = st.columns(3)
            with c1:
                metric("Machines Scored", f"{total:,}", accent=True)
            with c2:
                metric("Failure Predictions", f"{failures:,}")
            with c3:
                metric("No Failure", f"{healthy:,}")

            st.markdown(
                '<div class="panel"><div class="panel-title">Prediction Output</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                output,
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            if len(output) > 1:
                fig = px.histogram(
                    output,
                    x="Failure Probability (%)",
                    nbins=25,
                )
                fig.update_traces(marker_color="#38BDF8")
                inject_plotly_theme(fig, 340)
                fig.update_layout(
                    title="Failure Probability Distribution",
                    xaxis_title="Failure Probability (%)",
                    yaxis_title="Machines",
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            csv = output.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Prediction Output",
                data=csv,
                file_name="machine_failure_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except Exception as exc:
            st.error(f"Unable to process CSV: {exc}")


# ============================================================
# OTHER WORKFLOW PAGES
# ============================================================

def page_data_processing():
    section_header(
        "Data Processing",
        "Data Quality & Preparation",
        "The first stage ensures the model receives clean, relevant industrial signals.",
    )

    if dataset is None:
        st.error(data_error or "Dataset unavailable.")
        return

    id_columns = [c for c in ["UDI", "Product ID"] if c in dataset.columns]
    leakage = [c for c in LEAKAGE_COLUMNS if c in dataset.columns]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric("Raw Rows", f"{len(dataset):,}", accent=True)
    with c2:
        metric("Raw Columns", f"{len(dataset.columns):,}")
    with c3:
        metric("Missing Values", f"{int(dataset.isnull().sum().sum()):,}")
    with c4:
        metric("Duplicate Rows", f"{int(dataset.duplicated().sum()):,}")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown(
            '<div class="panel"><div class="panel-title">Columns Removed Before Modelling</div>',
            unsafe_allow_html=True,
        )
        removed = pd.DataFrame(
            {
                "Category": ["Identifiers", "Failure-mode leakage"],
                "Columns": [
                    ", ".join(id_columns) if id_columns else "None",
                    ", ".join(leakage) if leakage else "None",
                ],
            }
        )
        st.dataframe(
            removed,
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div class="panel"><div class="panel-title">Feature Engineering</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            **Heat Dissipation**  
            Process temperature − Air temperature

            **Power**  
            Mechanical power derived from rotational speed and torque

            **Overstrain**  
            Torque × Tool wear
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="callout">
            The failure-mode columns are removed because they describe specific
            failure mechanisms and would leak information about the target into
            a general-purpose failure detector.
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_baseline():
    section_header(
        "Baseline Models",
        "Original Sensor Measurements",
        "This stage establishes how far the raw operating signals can go before engineering new features.",
    )

    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Baseline Experiment</div>
            <div style="color:#94A3B8;font-size:.84rem;line-height:1.6;">
                Logistic Regression provides the linear baseline, while
                Decision Tree, Random Forest, Gradient Boosting and XGBoost
                test increasingly nonlinear decision boundaries.
                Because machine failure is rare, Recall and F2 are emphasized
                alongside Precision and F1.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    models = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting",
                "XGBoost",
            ],
            "Role": [
                "Linear baseline",
                "Rule-based nonlinear model",
                "Bagged ensemble",
                "Boosted ensemble",
                "Final high-performance candidate",
            ],
        }
    )

    st.dataframe(
        models,
        use_container_width=True,
        hide_index=True,
    )


def page_optimization():
    section_header(
        "Advanced Optimization",
        "Feature Engineering + SMOTE",
        "Test whether physically meaningful features and class balancing improve failure detection.",
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric("Heat Dissipation", "ΔT", accent=True)
    with c2:
        metric("Mechanical Power", "ω × τ")
    with c3:
        metric("Overstrain", "Torque × Wear")

    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="callout">
            <strong>Why feature engineering?</strong><br>
            Raw sensor values describe machine conditions individually.
            The engineered signals combine them into quantities that better
            represent thermal stress, mechanical loading and wear-related strain.
        </div>

        <div class="callout warning">
            <strong>Why test SMOTE?</strong><br>
            Only 3.39% of records are failures. SMOTE creates synthetic
            minority examples so the experiment can test whether additional
            failure examples improve Recall without creating excessive false alarms.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Optimization Decision</div>
            <div style="color:#94A3B8;font-size:.84rem;line-height:1.6;">
                The project promotes the feature-engineered XGBoost model
                without SMOTE when it provides the strongest practical balance
                between catching failures and limiting unnecessary maintenance alerts.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_model_selection():
    page_model_evaluation()


# ============================================================
# ROUTING
# ============================================================

if page == "Data Processing":
    page_data_processing()
elif page == "Exploratory Analysis":
    page_data_eda()
elif page == "Baseline Models":
    page_baseline()
elif page == "Advanced Optimization":
    page_optimization()
elif page == "Model Selection":
    page_model_selection()
elif page == "Model Deployment":
    page_deployment()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Predictive Maintenance ML System
        &nbsp; | &nbsp;
        Feature-Engineered XGBoost
        &nbsp; | &nbsp;
        Industrial Machine Failure Detection
    </div>
    """,
    unsafe_allow_html=True,
)