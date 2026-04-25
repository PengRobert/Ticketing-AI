"""
ui/app.py
---------
Streamlit application — two tabs:
  Tab 1 | Ticket Analyzer   : input a description → AI analysis
  Tab 2 | Analytics Dashboard: charts from processed historical data
"""

import sys
import os
from pathlib import Path

import subprocess

# Auto-build ChromaDB index on startup if empty
chroma_path = "chroma_db"
if not os.path.exists(chroma_path) or len(os.listdir(chroma_path)) == 0:
    from src.rag.indexer import index_tickets
    index_tickets()


# Make project root importable regardless of how Streamlit is launched
sys.path.insert(0, str(Path(__file__).parent.parent))

import html as _html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.settings import settings

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ticketing AI",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main > div { padding-top: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        padding: 0 24px;
        background-color: #f0f2f6;
        border-radius: 6px 6px 0 0;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4 !important;
        color: white !important;
    }
    .result-card {
        background: #f8f9fa;
        border-left: 4px solid #1f77b4;
        border-radius: 4px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .similar-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        height: 100%;
    }
    .sla-made  { color: #2ecc71; font-weight: 700; }
    .sla-missed{ color: #e74c3c; font-weight: 700; }
    .badge-priority {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

PRIORITY_COLORS = {"P1": "#e74c3c", "P2": "#e67e22", "P3": "#3498db", "P4": "#2ecc71"}


def _build_report_text(description: str, result) -> str:
    lines = [
        "=" * 60,
        "TICKET ANALYSIS REPORT",
        "=" * 60,
        "",
        "TICKET DESCRIPTION",
        "-" * 40,
        description,
        "",
        "CLASSIFICATION",
        "-" * 40,
        f"Resolution Code : {result.resolution_code}",
        f"Priority        : {result.priority}",
        "",
        "PROBLEM STATEMENT",
        "-" * 40,
        result.problem or "Not available",
        "",
        "ROOT CAUSE ANALYSIS",
        "-" * 40,
        result.rca or "Not available",
        "",
        "RECOMMENDED SOLUTION",
        "-" * 40,
        result.solution or "Not available",
        "",
        "TOP 3 SIMILAR CASES",
        "-" * 40,
    ]
    for i, case in enumerate(result.similar_cases[:3], 1):
        score_pct = int(case.similarity_score * 100)
        lines += [
            f"Case {i}: {case.ticket_no}  ({score_pct}% match)",
            f"  Description : {case.brief_description}",
            f"  Code        : {case.resolution_code}",
            f"  Priority    : {case.priority}",
            f"  SLA         : {case.sla}",
            "",
        ]
    if not result.similar_cases:
        lines.append("No similar cases found.")
    lines += ["", "=" * 60]
    return "\n".join(lines)


PRIORITY_LABELS = {
    "P1": "P1 – Critical",
    "P2": "P2 – High",
    "P3": "P3 – Medium",
    "P4": "P4 – Low",
}


def _priority_badge(priority: str) -> str:
    color = PRIORITY_COLORS.get(priority.upper(), "#95a5a6")
    label = PRIORITY_LABELS.get(priority.upper(), priority)
    return (
        f'<span class="badge-priority" '
        f'style="background:{color};color:white">{label}</span>'
    )


def _sla_badge(sla: str) -> str:
    cls = "sla-made" if sla.upper() == "MADE" else "sla-missed"
    return f'<span class="{cls}">{sla.upper()}</span>'


@st.cache_data(show_spinner=False)
def _load_processed_data() -> pd.DataFrame | None:
    path = Path(settings.PROCESSED_DATA_PATH)
    if not path.exists():
        return None
    return pd.read_csv(path)


# ── Tab definitions ──────────────────────────────────────────────────────────

def _tab_analyzer():
    """Tab 1 — Ticket Analyzer."""
    st.markdown("### Describe your ticket")
    st.caption(
        "Enter any IT support issue in plain language. "
        "The AI will classify it, find the root cause, and recommend a solution."
    )

    if "input_key" not in st.session_state:
        st.session_state["input_key"] = 0

    description = st.text_area(
        label="Ticket Description",
        placeholder=(
            "e.g. Users in the sales team are unable to log into Salesforce "
            "since this morning. The error message says 'Invalid session token'. "
            "This started after last night's SSO configuration update."
        ),
        height=130,
        label_visibility="collapsed",
        key=f"desc_{st.session_state['input_key']}",
    )

    col_btn, col_hint = st.columns([1, 5])
    with col_btn:
        run = st.button("🔍 Analyze Ticket", type="primary", width="stretch")
    with col_hint:
        st.caption("Analysis takes ~30–60 seconds via the AI pipeline.")

    # ── Run analysis ─────────────────────────────────────────────────────────
    if run:
        if not description.strip():
            st.warning("Please enter a ticket description before analyzing.")
            return

        if not settings.ANTHROPIC_API_KEY:
            st.error("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
            return

        with st.spinner("Running 4-agent AI pipeline …"):
            try:
                from src.crew.ticket_crew import analyze_ticket
                result = analyze_ticket(description)
                st.session_state["analysis_result"] = result
                st.session_state["analysis_description"] = description
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                st.session_state.pop("analysis_result", None)
                return

    # ── Display results ───────────────────────────────────────────────────────
    if "analysis_result" not in st.session_state:
        return

    result = st.session_state["analysis_result"]

    if not result.success:
        st.error(f"Analysis error: {result.error}")
        return

    st.divider()
    st.markdown("## Analysis Results")

    # Row 1: Classification
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Classification")
        rc_val = _html.escape(result.resolution_code) if result.resolution_code else "Unknown"
        st.markdown(
            f'<div class="result-card">'
            f'<b>Resolution Code</b><br>'
            f'<span style="font-size:1.4rem;font-weight:700;color:#1f77b4">{rc_val}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="result-card">'
            f'<b>Priority</b>&nbsp;&nbsp;{_priority_badge(result.priority)}'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown("#### Problem Statement")
        st.write(result.problem or "_Not available_")

    # Row 2: RCA + Solution
    col_rca, col_sol = st.columns(2)

    with col_rca:
        st.markdown("#### Root Cause Analysis")
        st.write(result.rca or "_Not available_")

    with col_sol:
        st.markdown("#### Recommended Solution")
        st.write(result.solution or "_Not available_")

    # Row 3: Similar cases
    st.markdown("---")
    st.markdown("#### Top 3 Similar Historical Cases")

    if not result.similar_cases:
        st.info("No similar cases found — ChromaDB index may be empty. Run `build_index.py`.")
    else:
        cols = st.columns(3)
        for i, (case, col) in enumerate(zip(result.similar_cases, cols)):
            score_pct = int(case.similarity_score * 100)
            with col:
                st.markdown(
                    f'<div class="similar-card">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center">'
                    f'<b style="color:#1f77b4">{case.ticket_no}</b>'
                    f'<span style="font-size:0.8rem;color:#888">{score_pct}% match</span>'
                    f'</div>'
                    f'<hr style="margin:6px 0">'
                    f'<div style="font-size:0.82rem;color:#555;margin-bottom:6px">'
                    f'{case.brief_description[:120]}{"…" if len(case.brief_description) > 120 else ""}'
                    f'</div>'
                    f'<div style="margin-bottom:4px"><b>Code:</b> {case.resolution_code}</div>'
                    f'<div style="margin-bottom:4px"><b>Priority:</b> {case.priority}</div>'
                    f'<div><b>SLA:</b> {_sla_badge(case.sla)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Action buttons ────────────────────────────────────────────────────────
    st.markdown("---")
    col_dl, col_new, _ = st.columns([1, 1, 4])

    report_text = _build_report_text(
        st.session_state.get("analysis_description", ""), result
    )
    with col_dl:
        st.download_button(
            label="📥 Download Report",
            data=report_text,
            file_name="ticket_analysis_report.txt",
            mime="text/plain",
            width="stretch",
        )

    with col_new:
        if st.button("🔄 Analyze New Ticket", width="stretch"):
            st.session_state.pop("analysis_result", None)
            st.session_state.pop("analysis_description", None)
            st.session_state["input_key"] = st.session_state.get("input_key", 0) + 1
            st.rerun()


# ── Analytics Dashboard ───────────────────────────────────────────────────────

_CHART_TEMPLATE = "plotly_white"
_PRIMARY_COLOR = "#1f77b4"


def _tab_analytics():
    """Tab 2 — Analytics Dashboard."""
    df = _load_processed_data()

    if df is None:
        st.warning(
            "Processed data not found. "
            "Run `python data_processor.py` to generate it."
        )
        return

    # ── Sanitise columns ─────────────────────────────────────────────────────
    df.columns = [c.strip().upper() for c in df.columns]
    df = df.fillna("Unknown")

    total = len(df)
    sla_col = "RESPOND_SLA"
    rc_col  = "RESOLUTION_CODE"
    pri_col = "PRIORITY"

    sla_made_rate = (
        (df[sla_col].str.upper() == "MADE").sum() / total * 100
        if sla_col in df.columns else 0.0
    )

    top_code = (
        df[rc_col].value_counts().idxmax() if rc_col in df.columns else "N/A"
    )

    # ── Metric row ────────────────────────────────────────────────────────────
    st.markdown("### Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Tickets", f"{total:,}")
    m2.metric("SLA MADE Rate", f"{sla_made_rate:.1f}%")
    m3.metric("Resolution Codes", df[rc_col].nunique() if rc_col in df.columns else 0)
    m4.metric("Top Category", top_code)

    st.divider()

    # ── Row A: Resolution Code distribution + Priority breakdown ─────────────
    col_a1, col_a2 = st.columns([3, 2])

    with col_a1:
        st.markdown("#### Resolution Code Distribution")
        if rc_col in df.columns:
            rc_counts = (
                df[rc_col].value_counts().reset_index()
            )
            rc_counts.columns = ["Resolution Code", "Count"]
            fig = px.bar(
                rc_counts,
                x="Count",
                y="Resolution Code",
                orientation="h",
                color="Count",
                color_continuous_scale="Blues",
                template=_CHART_TEMPLATE,
            )
            fig.update_layout(
                height=380,
                coloraxis_showscale=False,
                margin=dict(l=0, r=10, t=10, b=30),
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("RESOLUTION_CODE column not found in processed data.")

    with col_a2:
        st.markdown("#### Priority Breakdown")
        if pri_col in df.columns:
            pri_counts = df[pri_col].value_counts().reset_index()
            pri_counts.columns = ["Priority", "Count"]
            color_map = {
                "P1": "#e74c3c", "P2": "#e67e22",
                "P3": "#3498db", "P4": "#2ecc71",
            }
            fig = px.pie(
                pri_counts,
                names="Priority",
                values="Count",
                hole=0.45,
                color="Priority",
                color_discrete_map=color_map,
                template=_CHART_TEMPLATE,
            )
            fig.update_traces(textposition="outside", textinfo="percent+label")
            fig.update_layout(
                height=380,
                showlegend=False,
                margin=dict(l=0, r=0, t=10, b=30),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("PRIORITY column not found in processed data.")

    # ── Row B: SLA MADE rate by Resolution Code ───────────────────────────────
    st.markdown("#### SLA MADE Rate by Resolution Code")
    if rc_col in df.columns and sla_col in df.columns:
        df["_MADE"] = (df[sla_col].str.upper() == "MADE").astype(int)
        sla_by_code = (
            df.groupby(rc_col)["_MADE"]
            .agg(["sum", "count"])
            .reset_index()
        )
        sla_by_code.columns = ["Resolution Code", "Made", "Total"]
        sla_by_code["MADE Rate (%)"] = (
            sla_by_code["Made"] / sla_by_code["Total"] * 100
        ).round(1)
        sla_by_code["MISSED Rate (%)"] = 100 - sla_by_code["MADE Rate (%)"]
        sla_by_code = sla_by_code.sort_values("MADE Rate (%)")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="MADE",
            y=sla_by_code["Resolution Code"],
            x=sla_by_code["MADE Rate (%)"],
            orientation="h",
            marker_color="#2ecc71",
            text=sla_by_code["MADE Rate (%)"].apply(lambda v: f"{v:.0f}%"),
            textposition="inside",
        ))
        fig.add_trace(go.Bar(
            name="MISSED",
            y=sla_by_code["Resolution Code"],
            x=sla_by_code["MISSED Rate (%)"],
            orientation="h",
            marker_color="#e74c3c",
            text=sla_by_code["MISSED Rate (%)"].apply(lambda v: f"{v:.0f}%"),
            textposition="inside",
        ))
        fig.update_layout(
            barmode="stack",
            height=max(300, len(sla_by_code) * 36),
            template=_CHART_TEMPLATE,
            xaxis_title="Percentage (%)",
            xaxis_range=[0, 100],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=0, r=10, t=40, b=30),
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("RESOLUTION_CODE or RESPOND_SLA column not found in processed data.")

    # ── Row C: Monthly ticket volume (if dates present) ───────────────────────
    if "OPEN_DATE" in df.columns:
        st.markdown("#### Monthly Ticket Volume")
        df["_month"] = pd.to_datetime(df["OPEN_DATE"], errors="coerce").dt.to_period("M")
        monthly = (
            df.dropna(subset=["_month"])
            .groupby("_month")
            .size()
            .reset_index(name="Count")
        )
        monthly["Month"] = monthly["_month"].astype(str)
        if not monthly.empty:
            fig = px.line(
                monthly,
                x="Month",
                y="Count",
                markers=True,
                template=_CHART_TEMPLATE,
                color_discrete_sequence=[_PRIMARY_COLOR],
            )
            fig.update_layout(
                height=280,
                margin=dict(l=0, r=10, t=10, b=30),
                xaxis_title="",
            )
            st.plotly_chart(fig, width="stretch")


# ── Main layout ──────────────────────────────────────────────────────────────

def main():
    st.title("🎫 Ticketing AI")
    st.caption("AI-powered ticket analysis · Root cause identification · Historical pattern matching")

    tab1, tab2 = st.tabs(["🔍 Ticket Analyzer", "📊 Analytics Dashboard"])

    with tab1:
        _tab_analyzer()

    with tab2:
        _tab_analytics()


if __name__ == "__main__":
    main()
