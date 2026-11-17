"""Cybersecurity Threat Detection, Phishing Detection & Security Intelligence Visualization Platform.
Integrates historical cyber incident analytics, analytical risk scoring,
statistical anomaly detection, attacker attribution graphs, real-time URL phishing detection,
and live network traffic surveillance & visualization (PCAP replay, simulated scenarios, live packet capture).
"""

from pathlib import Path
import html
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.preprocessing import load_and_clean_data
from src.analysis import (
    attack_summary, yearly_trends, monthly_trends, country_attack_analysis,
    sector_analysis, impact_analysis, category_comparison, year_month_heatmap,
    attack_sector_comparison, risk_distribution_summary,
    generate_automated_intelligence_summary
)
from src.risk_engine import compute_risk_scores
from src.anomaly_detector import run_all_anomaly_detections
from src.attacker_intelligence import (
    get_initiator_country_summary, get_initiator_category_breakdown,
    get_attribution_sources, get_attacker_target_pairs, build_threat_sankey_diagram
)
from src.mitre_lifecycle import (
    get_initial_access_breakdown, get_mitre_impact_breakdown, get_attack_lifecycle_matrix
)
from src.alert_center import generate_security_alerts
from src.forecasting import calculate_trend_forecast, plot_forecast_chart
from src.phishing.detector import assess_phishing_url, get_metadata

# Real-time network intelligence imports
from src.realtime import (
    get_packet_collector, plot_traffic_timeline,
    plot_protocol_distribution, plot_top_ip_talkers,
    plot_network_communication_graph, plot_realtime_geolocation_map,
    create_sample_pcap
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "incidents.csv"
SAMPLE_PCAP_PATH = ROOT / "data" / "realtime" / "sample_traffic.pcap"

# Ensure sample PCAP is generated for immediate out-of-the-box demonstration
if not SAMPLE_PCAP_PATH.exists():
    try:
        create_sample_pcap(SAMPLE_PCAP_PATH)
    except Exception:
        pass

st.set_page_config(
    page_title="Security Intelligence & Real-Time Threat Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom SOC / Cybersecurity Theme Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root {
    --cyan: #48d6e8;
    --orange: #ff8552;
    --red: #ff4d6d;
    --green: #43d6a2;
    --purple: #8b9eff;
    --line: rgba(148, 163, 184, .14);
}
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: radial-gradient(circle at 85% 0%, #112340 0%, #090f1d 45%, #050912 100%);
    color: #edf4ff;
}
[data-testid="stHeader"] {
    background: rgba(5, 9, 18, .85);
    backdrop-filter: blur(8px);
}
.block-container {
    max-width: 1560px;
    padding-top: 1.8rem;
    padding-bottom: 3.5rem;
}
h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #edf4ff;
    letter-spacing: -0.01em;
}
p, label, span {
    color: #a9b9ce;
}
section[data-testid="stSidebar"] {
    width: 290px !important;
    min-width: 290px !important;
    background: linear-gradient(180deg, #0e1728 0%, #060b16 100%);
    border-right: 1px solid var(--line);
}
.brand {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #f5f9ff;
    padding: 10px 8px 2px;
    line-height: 1.2;
}
.brand span {
    color: var(--cyan);
}
.sidebar-sub {
    color: #8294ac;
    font-size: 10px;
    letter-spacing: 1.6px;
    padding: 0 8px 12px;
}
.sidebar-section {
    color: #8294ac;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 2px;
    padding: 12px 8px 6px;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
    gap: 4px;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"] {
    min-height: 40px;
    padding: 8px 12px;
    border-radius: 8px;
    border-left: 3px solid transparent;
    background: transparent;
    color: #c5d2e3;
    font-size: 0.9rem;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"]:hover {
    background: rgba(72, 214, 232, .08);
    border-left-color: var(--cyan);
    color: #fff;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"][aria-checked="true"] {
    background: linear-gradient(90deg, #103352, #13243a);
    border-left-color: var(--cyan);
    color: #fff;
    font-weight: 600;
}
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #121f35, #0a1322);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.2);
}
div[data-testid="stMetricLabel"] {
    color: #9fb1c9;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="stMetricValue"] {
    color: #f2f7ff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
}
.stPlotlyChart, [data-testid="stDataFrame"] {
    background: rgba(14, 23, 40, .68);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 8px;
}
.hero {
    background: linear-gradient(120deg, rgba(19, 41, 70, .95), rgba(13, 24, 44, .92) 55%, rgba(22, 45, 76, .9));
    border: 1px solid rgba(72, 214, 232, .25);
    border-radius: 16px;
    padding: 22px 26px;
    margin-bottom: 20px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
}
.kicker {
    color: var(--cyan);
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .16em;
    text-transform: uppercase;
}
.hero-title {
    color: #f5f9ff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.85rem;
    font-weight: 700;
    margin: 6px 0 4px;
}
.hero-sub {
    color: #a9bdd5;
    font-size: .9rem;
}
.hero-meta {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 14px;
}
.pill {
    border: 1px solid rgba(118, 180, 214, .25);
    background: rgba(8, 18, 33, .45);
    border-radius: 24px;
    color: #c4d7e9;
    padding: 4px 11px;
    font-size: .74rem;
}
.pill-danger {
    border-color: rgba(255, 77, 109, .4);
    background: rgba(255, 77, 109, .12);
    color: #ff94a5;
}
.pill-cyan {
    border-color: rgba(72, 214, 232, .4);
    background: rgba(72, 214, 232, .12);
    color: #79e5f3;
}
.alert-card {
    border: 1px solid var(--line);
    border-left: 4px solid var(--cyan);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    background: rgba(14, 24, 43, 0.7);
}
.alert-card.critical {
    border-left-color: var(--red);
    background: linear-gradient(90deg, rgba(255, 77, 109, .1), rgba(14, 24, 43, .6));
}
.alert-card.high {
    border-left-color: var(--orange);
    background: linear-gradient(90deg, rgba(255, 133, 82, .1), rgba(14, 24, 43, .6));
}
.alert-card.medium {
    border-left-color: #f3b76b;
}
.insight {
    background: linear-gradient(110deg, rgba(31, 73, 91, .35), rgba(18, 32, 55, .6));
    border: 1px solid rgba(72, 214, 232, .2);
    border-left: 3px solid var(--cyan);
    border-radius: 10px;
    padding: 12px 16px;
    color: #d5e6f6;
    margin: 8px 0 14px;
    font-size: .88rem;
}
.small-note {
    color: #8397b1;
    font-size: .78rem;
}
hr {
    border-color: var(--line);
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading and preprocessing cybersecurity threat dataset...")
def load_data(path: str) -> pd.DataFrame:
    return load_and_clean_data(path)


def heading(kicker: str, title: str, description: str = ""):
    desc = f'<div class="small-note">{html.escape(description)}</div>' if description else ""
    st.markdown(
        f'<div style="margin: 16px 0 10px"><div class="kicker">{html.escape(kicker)}</div>'
        f'<h3 style="margin: 3px 0 4px">{html.escape(title)}</h3>{desc}</div>',
        unsafe_allow_html=True
    )


def show_chart(fig: go.Figure, height: int = 360):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#b8c8dc", size=12),
        margin=dict(l=18, r=18, t=42, b=18),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.08),
        xaxis=dict(gridcolor="rgba(148,163,184,.12)"),
        yaxis=dict(gridcolor="rgba(148,163,184,.12)")
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# Data Source Verification
if not DATA_PATH.exists():
    st.error("Dataset not found. Please place the EuRepoC dataset CSV at data/incidents.csv.")
    st.stop()

try:
    df_raw = load_data(str(DATA_PATH))
except Exception as exc:
    st.error("Failed to load cybersecurity incident dataset.")
    st.exception(exc)
    st.stop()

# Precompute historical anomalies across the full baseline
all_anomalies = run_all_anomaly_detections(df_raw)

# Sidebar Branding & Navigation
st.sidebar.markdown('<div class="brand">SECURITY <span>INTELLIGENCE</span><br>&amp; THREAT PLATFORM</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-sub">SOC &amp; LIVE NETWORK ANALYTICS</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.markdown('<div class="sidebar-section">NAVIGATION</div>', unsafe_allow_html=True)
nav_pages = [
    "Overview",
    "Threat Intelligence",
    "Attack Analytics",
    "Geographic Intelligence",
    "Attacker Intelligence",
    "Risk & Anomalies",
    "Phishing Detector",
    "Network / Relationship Graph",
    "Attack Lifecycle (MITRE)",
    "Incident Explorer",
    "Model Performance & Methodology",
    "Real-Time Security Data"
]
page = st.sidebar.radio("Navigation", nav_pages, label_visibility="collapsed", key="active_page")

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-section">GLOBAL SOC FILTERS</div>', unsafe_allow_html=True)

# 1. Year Slider
years = df_raw.year.dropna()
ymin, ymax = int(years.min()), int(years.max())
year_range = st.sidebar.slider("Incident period", ymin, ymax, (ymin, ymax))

# 2. Country Multiselect
all_countries = sorted({c for vals in df_raw.country_list for c in vals if c})
selected_countries = st.sidebar.multiselect("Receiver country", all_countries, placeholder="All countries")

# 3. Sector Multiselect
all_sectors = sorted({s for vals in df_raw.sector_list for s in vals if s})
selected_sectors = st.sidebar.multiselect("Target sector", all_sectors, placeholder="All sectors")

# 4. Severity Tier Filter
risk_levels = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
selected_risks = st.sidebar.multiselect("Analytical risk level", risk_levels, placeholder="All severity levels")

# 5. Primary Threat Vector
threat_options = ["DDoS-related", "Ransomware-related", "Malware-related", "Both DDoS & Ransomware"]
selected_vectors = st.sidebar.multiselect("Attack category", threat_options, placeholder="All threat categories")

# Apply Dynamic Filters
filtered = df_raw[df_raw.year.between(*year_range)].copy()

if selected_countries:
    filtered = filtered[filtered.country_list.apply(lambda vals: any(c in selected_countries for c in vals))]

if selected_sectors:
    filtered = filtered[filtered.sector_list.apply(lambda vals: any(s in selected_sectors for s in vals))]

if selected_risks:
    filtered = filtered[filtered.risk_level.isin(selected_risks)]

if selected_vectors:
    if "DDoS-related" in selected_vectors:
        filtered = filtered[filtered.is_ddos]
    if "Ransomware-related" in selected_vectors:
        filtered = filtered[filtered.is_ransomware]
    if "Malware-related" in selected_vectors:
        filtered = filtered[filtered.is_malware_related]
    if "Both DDoS & Ransomware" in selected_vectors:
        filtered = filtered[filtered.is_ddos & filtered.is_ransomware]

# Top Hero Banner
crit_count = int((filtered["risk_level"] == "CRITICAL").sum()) if "risk_level" in filtered.columns else 0
high_count = int((filtered["risk_level"] == "HIGH").sum()) if "risk_level" in filtered.columns else 0
avg_risk_val = round(float(filtered["risk_score"].mean()), 1) if not filtered.empty else 0.0

hero_html = f"""
<div class="hero">
    <div class="kicker">CYBERSECURITY INTELLIGENCE &bull; SECURITY OPERATIONS CENTER</div>
    <div class="hero-title">Real-Time Threat Analytics &amp; Security Visualization</div>
    <p class="hero-sub">Unified platform fusing EuRepoC historical threat intelligence, analytical risk scoring, anomaly detection, phishing verification, and real-time network traffic surveillance.</p>
    <div class="hero-meta">
        <span class="pill pill-cyan">EuRepoC Dataset v1.3</span>
        <span class="pill">{year_range[0]} &mdash; {year_range[1]}</span>
        <span class="pill">{len(filtered):,} Incidents in Filter</span>
        <span class="pill pill-danger">{crit_count:,} Critical Threats</span>
        <span class="pill">Avg Analytical Risk: {avg_risk_val}/100</span>
        <span class="pill pill-cyan">{len(all_anomalies)} Historical Anomalies Flagged</span>
        <span class="pill">Real-Time Engine: Available</span>
    </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

if filtered.empty:
    st.warning("No incidents match the active filter criteria. Adjust the sidebar filters to view records.")
    st.stop()

# Generate active SOC Alerts
active_alerts = generate_security_alerts(filtered, all_anomalies)

# ==============================================================================
# MODULE 1: SECURITY OVERVIEW / COMMAND CENTER
# ==============================================================================
if page == "Overview":
    heading("COMMAND CENTER", "Executive Security KPIs", "Aggregated intelligence snapshot across verified incident records.")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("TOTAL INCIDENTS", f"{len(filtered):,}")
    col2.metric("CRITICAL THREATS", f"{crit_count:,}")
    col3.metric("HIGH RISK THREATS", f"{high_count:,}")
    col4.metric("DDoS EVENTS", f"{int(filtered.is_ddos.sum()):,}")
    col5.metric("RANSOMWARE", f"{int(filtered.is_ransomware.sum()):,}")
    col6.metric("AVG RISK SCORE", f"{avg_risk_val}/100")

    col7, col8, col9, col10, col11, col12 = st.columns(6)
    uniq_countries = len({c for vals in filtered.country_list for c in vals if c})
    uniq_sectors = len({s for vals in filtered.sector_list for s in vals if s})
    disrupt_cnt = int(filtered.has_disruption.sum())
    zdays_cnt = int(filtered.has_zero_day.sum()) if "has_zero_day" in filtered.columns else 0
    
    col7.metric("COUNTRIES TARGETED", f"{uniq_countries:,}")
    col8.metric("SECTORS TARGETED", f"{uniq_sectors:,}")
    col9.metric("SYSTEM DISRUPTIONS", f"{disrupt_cnt:,}")
    col10.metric("ZERO-DAY EXPLOITS", f"{zdays_cnt:,}")
    col11.metric("ANOMALIES DETECTED", f"{len(all_anomalies)}")
    col12.metric("REAL-TIME ENGINE", "READY (PCAP/Sim)")

    st.markdown("---")

    c_left, c_right = st.columns([1.6, 1.0], gap="large")

    with c_left:
        heading("ATTACK VECTOR PROFILE", "DDoS vs Ransomware vs Malware")
        comp = category_comparison(filtered)
        fig_pie = px.pie(
            comp, names="Category", values="Incidents", hole=0.6,
            color="Category",
            color_discrete_map={
                "DDoS only": "#ffb86b",
                "Ransomware only": "#8b9eff",
                "Both": "#ff4d6d",
                "Neither": "#364761"
            }
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        show_chart(fig_pie, height=330)

        heading("ANALYTICAL RISK DISTRIBUTION", "Incident severity breakdown (0-100 scale)")
        r_dist = risk_distribution_summary(filtered)
        fig_r = px.bar(
            r_dist, x="Risk Level", y="Incidents", color="Risk Level",
            color_discrete_map={
                "CRITICAL": "#ff4d6d",
                "HIGH": "#ff8552",
                "MEDIUM": "#f3b76b",
                "LOW": "#43d6a2"
            },
            text="Incidents"
        )
        fig_r.update_traces(textposition="outside")
        show_chart(fig_r, height=310)

    with c_right:
        heading("SECURITY ALERT CENTER", "Active Deterministic SOC Threat Signals", "Generated from dataset risk thresholds and statistical anomalies.")
        if active_alerts:
            for al in active_alerts[:6]:
                cls_name = al["severity"].lower()
                st.markdown(f"""
                <div class="alert-card {cls_name}">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:700; font-size:0.75rem; letter-spacing:0.1em; color:{al['color']}">{al['severity']} &bull; {al['category']}</span>
                        <span style="font-size:0.72rem; color:#8294ac">{al['period']}</span>
                    </div>
                    <div style="font-weight:600; color:#edf4ff; font-size:0.92rem; margin:3px 0 2px">{html.escape(al['title'])}</div>
                    <div style="font-size:0.82rem; color:#a9bdd5">{html.escape(al['message'])}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No active alerts triggered for the current filter selection.")

    st.markdown("---")
    heading("AUTOMATED THREAT BRIEF", "Data-Grounded Security Intelligence Summary")
    brief = generate_automated_intelligence_summary(filtered)
    st.markdown(f'<div class="insight" style="border-left-color:{brief["posture_color"]}"><b>THREAT POSTURE:</b> <span style="color:{brief["posture_color"]}; font-weight:700">{brief["threat_posture"]}</span><br><br>' + "<br>• ".join([""] + [html.escape(o) for o in brief["observations"]]) + '</div>', unsafe_allow_html=True)

# ==============================================================================
# MODULE 2 & 15: THREAT INTELLIGENCE
# ==============================================================================
elif page == "Threat Intelligence":
    heading("THREAT INTELLIGENCE", "Longitudinal Threat Evolution", "Multi-year trajectory of cyber incident reporting and vector progression.")
    
    t = yearly_trends(filtered)
    fig_trends = px.line(
        t, x="year", y=["Total", "DDoS-related", "Ransomware-related", "Malware-related"],
        markers=True,
        color_discrete_sequence=["#48d6e8", "#ffb86b", "#8b9eff", "#43d6a2"]
    )
    show_chart(fig_trends, height=400)

    c1, c2 = st.columns(2)
    with c1:
        heading("SYSTEM DISRUPTION INTENSITY", "Incidents causing operational interruption over time")
        if not filtered.empty and "year" in filtered.columns:
            disrupt_yearly = filtered.groupby(["year", "has_disruption"])["incident_id"].nunique().reset_index()
            disrupt_yearly["Disruption"] = disrupt_yearly["has_disruption"].map({True: "Disruption Observed", False: "No Disruption Recorded"})
            fig_dis = px.bar(
                disrupt_yearly, x="year", y="incident_id", color="Disruption",
                color_discrete_sequence=["#3a506b", "#ff4d6d"],
                barmode="stack"
            )
            show_chart(fig_dis, height=340)

    with c2:
        heading("HISTORICAL TREND FORECAST (MODULE 16)", "OLS Trend Projection with 80% Uncertainty Bounds", "Clearly labeled as historical exploratory projection, not guaranteed future attacks.")
        forecast_data = calculate_trend_forecast(filtered, target_metric="total", forecast_years=3)
        fig_fc = plot_forecast_chart(forecast_data, title="Projected Incident Volume (3-Year Horizon)")
        show_chart(fig_fc, height=340)

# ==============================================================================
# MODULE 2: ATTACK ANALYTICS
# ==============================================================================
elif page == "Attack Analytics":
    heading("ATTACK ANALYTICS", "Deep-Dive Vector Analysis", "Dissecting DDoS, Ransomware, and Malware tactics across sectors and time.")

    c1, c2 = st.columns(2)
    with c1:
        heading("MONTHLY INCIDENT CONCENTRATION", "Calendar month distribution")
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        m = monthly_trends(filtered)
        m["Month"] = m.month.map(lambda x: months[int(x) - 1])
        fig_m = px.bar(
            m, x="Month", y="Incidents", color="Incidents",
            color_continuous_scale=["#183454", "#48d6e8"],
            category_orders={"Month": months}
        )
        show_chart(fig_m, height=340)

    with c2:
        heading("TEMPORAL HEATMAP", "Year x Month Activity Density")
        h = year_month_heatmap(filtered)
        if not h.empty:
            fig_h = px.imshow(
                h, aspect="auto",
                color_continuous_scale=["#0e192c", "#287da1", "#f3b76b"],
                labels={"x": "Month", "y": "Year", "color": "Incidents"},
                x=months
            )
            show_chart(fig_h, height=340)
        else:
            st.info("Insufficient timestamp data for heatmap.")

    st.markdown("---")
    heading("TARGET SECTOR EXPOSURE", "Attack Vector Presence Across Victim Sectors")
    sec_comp = attack_sector_comparison(filtered)
    if not sec_comp.empty:
        long_df = sec_comp.head(14).melt(
            id_vars="Target subcategory", var_name="Attack Category", value_name="Incidents"
        )
        fig_sc = px.bar(
            long_df, x="Target subcategory", y="Incidents", color="Attack Category",
            barmode="group",
            color_discrete_sequence=["#ffb86b", "#8b9eff", "#43d6a2"]
        )
        show_chart(fig_sc, height=420)
    else:
        st.info("Insufficient sector categorization data.")

# ==============================================================================
# MODULE 2 & 5: GEOGRAPHIC INTELLIGENCE
# ==============================================================================
elif page == "Geographic Intelligence":
    heading("GEOGRAPHIC INTELLIGENCE", "Global Threat Distribution", "Receiver country concentration and international cross-border patterns.")
    
    geo = country_attack_analysis(filtered)
    if geo.empty:
        st.info("No geographic receiver country information available.")
    else:
        fig_map = px.choropleth(
            geo, locations="Country", locationmode="country names",
            color="Incidents", hover_name="Country",
            hover_data=["DDoS-related", "Ransomware-related", "Malware-related"],
            color_continuous_scale=["#10223a", "#226a8d", "#48d6e8"]
        )
        fig_map.update_geos(
            bgcolor="rgba(0,0,0,0)", showframe=False, showcoastlines=True, landcolor="#121e33"
        )
        show_chart(fig_map, height=480)

        c1, c2 = st.columns(2)
        with c1:
            heading("TOP RECEIVER COUNTRIES", "Ranked by total recorded cyber incidents")
            fig_bar = px.bar(
                geo.head(15).sort_values("Incidents"),
                x="Incidents", y="Country", orientation="h",
                color="Incidents", color_continuous_scale=["#1d3d63", "#48d6e8"]
            )
            show_chart(fig_bar, height=440)
        with c2:
            heading("ATTACKER &rarr; TARGET COUNTRY FLOWS", "Attributed cross-border incident links")
            pairs = get_attacker_target_pairs(filtered, top_n=12)
            if not pairs.empty:
                fig_pairs = px.bar(
                    pairs, x="Incidents", y="Target Country", color="Initiator Country",
                    orientation="h", barmode="stack",
                    title="Target Countries by Attributed Initiator State"
                )
                show_chart(fig_pairs, height=440)
            else:
                st.info("No cross-border attribution pairs available in active view.")

# ==============================================================================
# MODULE 5: ATTACKER / INITIATOR INTELLIGENCE
# ==============================================================================
elif page == "Attacker Intelligence":
    heading("ATTACKER INTELLIGENCE", "Threat Actor Attribution & Origin Profiling", "Analysis of attributed origin countries, state responsibility, and attribution authorities.")
    
    c1, c2 = st.columns(2)
    with c1:
        heading("ORIGIN / INITIATOR COUNTRIES", "Attributed state and regional origin of threat operations")
        init_c = get_initiator_country_summary(filtered)
        if not init_c.empty:
            fig_init = px.bar(
                init_c.head(12).sort_values("Incidents"),
                x="Incidents", y="Initiator Country", orientation="h",
                color="Incidents", color_continuous_scale=["#5e2a3b", "#ff4d6d"]
            )
            show_chart(fig_init, height=380)
        else:
            st.info("No attributed initiator country data in this selection.")

    with c2:
        heading("INITIATOR CATEGORY", "State vs Non-State threat actor classifications")
        init_cat = get_initiator_category_breakdown(filtered)
        if not init_cat.empty:
            fig_cat = px.pie(
                init_cat, names="Category", values="Incidents", hole=0.58,
                color_discrete_sequence=["#ff8552", "#8b9eff", "#48d6e8", "#52647d"]
            )
            show_chart(fig_cat, height=380)
        else:
            st.info("No initiator category data in this selection.")

    heading("ATTRIBUTION SOURCES", "Leading cybersecurity intelligence vendors and public bodies")
    attr_sources = get_attribution_sources(filtered)
    if not attr_sources.empty:
        st.dataframe(attr_sources.head(15), use_container_width=True, hide_index=True)
    else:
        st.info("No specific attribution company or authority recorded.")

# ==============================================================================
# MODULE 3 & 4: RISK & ANOMALIES
# ==============================================================================
elif page == "Risk & Anomalies":
    heading("THREAT RISK SCORING ENGINE", "Analytical Risk Scoring (0-100)", "Transparent, attribute-driven scoring based on disruption, data theft, hijacking, zero-days, and MITRE techniques.")

    r_dist = risk_distribution_summary(filtered)
    c1, c2 = st.columns([1, 1.4], gap="large")
    with c1:
        fig_pie_r = px.pie(
            r_dist, names="Risk Level", values="Incidents", hole=0.6,
            color="Risk Level",
            color_discrete_map={
                "CRITICAL": "#ff4d6d",
                "HIGH": "#ff8552",
                "MEDIUM": "#f3b76b",
                "LOW": "#43d6a2"
            }
        )
        show_chart(fig_pie_r, height=320)
    with c2:
        st.markdown("""
        <div class="insight">
            <b>ANALYTICAL RISK METHODOLOGY:</b><br>
            • <b>0–30 (LOW):</b> Standard reconnaissance or exploratory scanning; no confirmed disruption or theft.<br>
            • <b>31–60 (MEDIUM):</b> Publicly reported incident with moderate impact or non-critical target sector.<br>
            • <b>61–80 (HIGH):</b> Confirmed operational disruption, verified sensitive data theft, or critical infrastructure targeting.<br>
            • <b>81–100 (CRITICAL):</b> Extended disruption (>24h), Zero-Day usage, destructive wiper/hijacking, or critical sector impact.
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(r_dist, use_container_width=True, hide_index=True)

    st.markdown("---")
    heading("CRITICAL INCIDENTS & CONTRIBUTING FACTORS", "Top evaluated high-severity incidents")
    top_crit = filtered.sort_values("risk_score", ascending=False).head(5)
    for _, row in top_crit.iterrows():
        st.markdown(f"""
        <div class="alert-card critical">
            <div style="display:flex; justify-content:space-between;">
                <span style="font-weight:700; color:#ff4d6d">RISK SCORE: {row['risk_score']}/100 &bull; LEVEL: {row['risk_level']}</span>
                <span style="color:#8294ac">Year: {row.get('year', 'N/A')}</span>
            </div>
            <div style="font-weight:600; font-size:1.05rem; color:#fff; margin:4px 0">{html.escape(str(row['name']))}</div>
            <div style="font-size:0.85rem; color:#a9bdd5"><b>Target Country:</b> {html.escape(str(row.get('receiver_country', 'N/A')))} | <b>Target Sector:</b> {html.escape(str(row.get('receiver_subcategory', 'N/A')))}</div>
            <div style="margin-top:6px; font-size:0.82rem; color:#f0f5ff"><b>Contributing Risk Factors:</b><br>{"".join(['• ' + html.escape(f) + '<br>' for f in row.get('risk_factors', [])])}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    heading("HISTORICAL ANOMALY DETECTION (MODULE 4)", "Statistical Time-Series & Target Concentration Outliers", "Grounded in Z-score (+1.75&sigma;) and Interquartile Range (IQR) outlier evaluation against historical baselines.")

    if all_anomalies:
        for anom in all_anomalies:
            cls_name = anom["severity"].lower()
            badge_color = "#ff4d6d" if anom["severity"] == "CRITICAL" else "#ff8552"
            st.markdown(f"""
            <div class="alert-card {cls_name}">
                <div style="display:flex; justify-content:space-between;">
                    <span style="font-weight:700; color:{badge_color}">{anom['severity']} ANOMALY DETECTED &bull; {anom['metric']}</span>
                    <span style="color:#8294ac">Period: {anom['period']}</span>
                </div>
                <div style="font-size:0.88rem; color:#edf4ff; margin:6px 0">
                    <b>Observed Metric Value:</b> {anom['observed_value']} | <b>Expected Baseline Range:</b> {anom['expected_range']} | <b>Statistical Z-Score:</b> +{anom['z_score']}&sigma;
                </div>
                <div style="font-size:0.83rem; color:#a9bdd5">{html.escape(anom['explanation'])}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No statistical anomalies identified in current dataset baseline.")

# ==============================================================================
# MODULE 8, 9, 10, 11, 12: PHISHING WEBSITE DETECTOR
# ==============================================================================
elif page == "Phishing Detector":
    heading("PHISHING WEBSITE DETECTOR", "Offline Machine Learning & URL Lexical Analysis Engine", "Safely inspect user-submitted URLs using feature extraction and Random Forest classification without visiting untrusted domains.")

    st.markdown("""
    <div class="insight">
        <b>SAFE EVALUATION GUARANTEE:</b> This module extracts 21 lexical, structural, and character-entropy features entirely offline.
        Submitted URLs are treated as untrusted input and are <b>never visited, resolved, or executed</b> by the server.
    </div>
    """, unsafe_allow_html=True)

    url_input = st.text_input(
        "Enter Suspicious URL to Analyze",
        value="http://192.168.1.100/paypal-security-update/login.php?token=9281928",
        placeholder="https://example.com/login"
    )

    c_btn, _ = st.columns([1, 4])
    with c_btn:
        analyze_clicked = st.button("ANALYZE URL", type="primary", use_container_width=True)

    if url_input:
        res = assess_phishing_url(url_input)

        if "error" in res and res["error"]:
            st.error(res["error"])
        else:
            st.markdown("---")
            c_score, c_details = st.columns([1, 1.8], gap="large")

            with c_score:
                heading("PHISHING RISK ASSESSMENT", f"{res['risk_level']} RISK TIER")
                
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=res["risk_score"],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Phishing Risk Score (0-100)", 'font': {'size': 14, 'color': "#edf4ff"}},
                    number={'suffix': "/100", 'font': {'family': 'Space Grotesk', 'color': res['risk_color'], 'size': 38}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': "#8294ac"},
                        'bar': {'color': res['risk_color']},
                        'bgcolor': "rgba(255,255,255,0.05)",
                        'steps': [
                            {'range': [0, 30], 'color': "rgba(67, 214, 162, 0.15)"},
                            {'range': [30, 60], 'color': "rgba(243, 183, 107, 0.15)"},
                            {'range': [60, 80], 'color': "rgba(255, 133, 82, 0.15)"},
                            {'range': [80, 100], 'color': "rgba(255, 77, 109, 0.25)"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 3},
                            'thickness': 0.75,
                            'value': res["risk_score"]
                        }
                    }
                ))
                show_chart(fig_gauge, height=270)

                st.markdown(f"""
                <div style="background:rgba(18,31,54,.7); border:1px solid var(--line); border-radius:10px; padding:12px 16px; margin-top:8px">
                    <div style="font-size:0.78rem; color:#8294ac; text-transform:uppercase">MODEL CLASSIFICATION</div>
                    <div style="font-size:1.35rem; font-weight:700; color:{res['risk_color']}">{res['risk_badge']}</div>
                    <div style="font-size:0.84rem; color:#a9bdd5; margin-top:3px"><b>Probability:</b> {res['probability']:.4f} &bull; <b>Verdict:</b> {'PHISHING' if res['is_phishing'] else 'BENIGN'}</div>
                </div>
                """, unsafe_allow_html=True)

            with c_details:
                heading("EXPLAINABILITY (WHY WAS IT FLAGGED?)", "Feature-Attributed Evidence Breakdown")
                
                if res["factors"]:
                    st.markdown("<b>Threat Indicators Identified:</b>", unsafe_allow_html=True)
                    for f in res["factors"]:
                        st.markdown(f'<div style="color:#ff8552; font-size:0.86rem; margin:3px 0">&#10060; {html.escape(f)}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="color:#43d6a2; font-size:0.86rem;">&#10004; No suspicious structural or lexical indicators detected.</div>', unsafe_allow_html=True)

                if res["positive_indicators"]:
                    st.markdown("<br><b>Legitimate / Benign Signals:</b>", unsafe_allow_html=True)
                    for p in res["positive_indicators"]:
                        st.markdown(f'<div style="color:#43d6a2; font-size:0.86rem; margin:3px 0">&#10004; {html.escape(p)}</div>', unsafe_allow_html=True)

                st.markdown("<br><b>Security Recommendation:</b>", unsafe_allow_html=True)
                rec_color = "#ff4d6d" if res["risk_score"] >= 61 else ("#f3b76b" if res["risk_score"] >= 31 else "#43d6a2")
                st.markdown(f"""
                <div style="border-left: 3px solid {rec_color}; background: rgba(14,24,43,0.8); padding:10px 14px; border-radius:8px; font-size:0.86rem; color:#edf4ff">
                    {html.escape(res['recommendation'])}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            heading("EXTRACTED URL FEATURES", "Full 21-Dimensional Normalized Feature Vector")
            feat_df = pd.DataFrame([res["features"]]).T.reset_index()
            feat_df.columns = ["Feature Attribute", "Extracted Value"]
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

# ==============================================================================
# MODULE 6: ATTACK RELATIONSHIP / NETWORK GRAPH
# ==============================================================================
elif page == "Network / Relationship Graph":
    heading("ATTACK RELATIONSHIP GRAPH", "Interactive Multi-Stage Threat Flow (Sankey)", "Visualizing multi-hop relationships: Initiator Origin &rarr; Attack Vector &rarr; Receiver Country &rarr; Targeted Sector.")
    
    st.markdown("""
    <div class="insight">
        <b>HOW TO INTERPRET THIS GRAPH:</b> The flows map verified multi-stage relationships from left to right.
        Hover over nodes or connecting ribbons to inspect flow volumes across nations and critical infrastructure.
    </div>
    """, unsafe_allow_html=True)

    fig_sankey = build_threat_sankey_diagram(filtered)
    st.plotly_chart(fig_sankey, use_container_width=True)

# ==============================================================================
# MODULE 7: ATTACK LIFECYCLE (MITRE ATT&CK)
# ==============================================================================
elif page == "Attack Lifecycle (MITRE)":
    heading("ATTACK LIFECYCLE & MITRE MATRIX", "Tactical Progression from Initial Access to Impact", "Documented MITRE ATT&CK TA0001 (Initial Access) and TA0040 (Impact) mapping directly grounded in EuRepoC records.")

    c1, c2 = st.columns(2)
    with c1:
        heading("INITIAL ACCESS VECTORS (TA0001)", "Documented techniques used to breach target networks")
        ia = get_initial_access_breakdown(filtered)
        if not ia.empty:
            fig_ia = px.bar(
                ia.head(10).sort_values("Incidents"),
                x="Incidents", y="Initial Access Technique", orientation="h",
                color="Incidents", color_continuous_scale=["#193959", "#48d6e8"]
            )
            show_chart(fig_ia, height=360)
        else:
            st.info("No MITRE Initial Access data recorded for active filter.")

    with c2:
        heading("MITRE IMPACT TECHNIQUES (TA0040)", "Documented payload and disruption objectives")
        mi = get_mitre_impact_breakdown(filtered)
        if not mi.empty:
            fig_mi = px.bar(
                mi.head(10).sort_values("Incidents"),
                x="Impact Technique", orientation="h",
                color="Incidents", color_continuous_scale=["#3b234a", "#8b9eff"]
            )
            show_chart(fig_mi, height=360)
        else:
            st.info("No MITRE Impact techniques recorded for active filter.")

    st.markdown("---")
    heading("LIFECYCLE CROSS-TABULATION MATRIX", "Co-occurrence of Initial Access tactics with resulting Impact techniques")
    lifecycle_mat = get_attack_lifecycle_matrix(filtered)
    if not lifecycle_mat.empty:
        fig_mat = px.imshow(
            lifecycle_mat,
            labels=dict(x="Impact Technique (TA0040)", y="Initial Access (TA0001)", color="Incidents"),
            color_continuous_scale=["#0e192c", "#287da1", "#ff8552"],
            aspect="auto"
        )
        show_chart(fig_mat, height=380)
    else:
        st.info("Insufficient multi-stage MITRE records with both Initial Access and Impact documented simultaneously.")

# ==============================================================================
# MODULE 14: INCIDENT EXPLORER
# ==============================================================================
elif page == "Incident Explorer":
    heading("INCIDENT EXPLORER", "Search, Filter, and Inspect Raw Threat Intelligence", "Examine individual incidents, inspect contributing risk factors, and export records.")

    c_s, c_r, c_v = st.columns([2, 1, 1])
    with c_s:
        q = st.text_input("Search incident title or description", placeholder="Incident name, actor, or organization...")
    with c_r:
        r_filter = st.selectbox("Risk Level Tier", ["All Tiers", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
    with c_v:
        sort_by = st.selectbox("Sort By", ["Risk Score (Highest First)", "Start Date (Newest First)", "Incident ID"])

    res_df = filtered.copy()
    if q:
        res_df = res_df[
            res_df.name.str.contains(q, case=False, na=False) |
            res_df.description.str.contains(q, case=False, na=False)
        ]

    if r_filter != "All Tiers":
        res_df = res_df[res_df.risk_level == r_filter]

    if sort_by == "Risk Score (Highest First)":
        res_df = res_df.sort_values("risk_score", ascending=False)
    elif sort_by == "Start Date (Newest First)":
        res_df = res_df.sort_values("start_date", ascending=False)

    st.metric("MATCHING INCIDENTS", f"{len(res_df):,}")

    display_cols = [
        c for c in [
            "incident_id", "risk_score", "risk_level", "name", "start_date",
            "receiver_country", "receiver_subcategory", "initiator_country",
            "mitre_impact", "has_disruption", "classification_basis"
        ] if c in res_df.columns
    ]

    st.dataframe(res_df[display_cols], use_container_width=True, hide_index=True)

    if not res_df.empty:
        st.markdown("---")
        heading("DETAILED INCIDENT INSPECTOR", "Select an incident ID to inspect full dossier")
        selected_id = st.selectbox("Select Incident ID to Inspect", res_df["incident_id"].head(50).tolist())
        target_rec = res_df[res_df["incident_id"] == selected_id].iloc[0]

        d_col1, d_col2 = st.columns([1.2, 1], gap="large")
        with d_col1:
            st.markdown(f"### {target_rec.get('name', 'Incident ' + str(selected_id))}")
            st.markdown(f"**Description:** {target_rec.get('description', 'No description available.')}")
            st.markdown(f"**Start Date:** {target_rec.get('start_date', 'N/A')} | **End Date:** {target_rec.get('end_date', 'N/A')}")
            st.markdown(f"**Receiver:** {target_rec.get('receiver_name', 'N/A')} ({target_rec.get('receiver_country', 'N/A')})")
            st.markdown(f"**Target Sector:** {target_rec.get('receiver_subcategory', 'N/A')} ({target_rec.get('receiver_category', 'N/A')})")
            st.markdown(f"**Initiator / Attributed Actor:** {target_rec.get('initiator_name', 'Not attributed')} ({target_rec.get('initiator_country', 'Not attributed')})")
            st.markdown(f"**Source Disclosure:** {target_rec.get('source_url', 'N/A')}")

        with d_col2:
            st.markdown(f"""
            <div class="alert-card critical">
                <div style="font-size:0.8rem; color:#8294ac">ANALYTICAL EVALUATION</div>
                <div style="font-size:1.6rem; font-weight:700; color:#ff4d6d">Score: {target_rec.get('risk_score', 'N/A')}/100 ({target_rec.get('risk_level', 'N/A')})</div>
                <div style="font-size:0.85rem; color:#edf4ff; margin-top:8px"><b>Contributing Risk Factors:</b><br>
                {"".join(['• ' + html.escape(f) + '<br>' for f in target_rec.get('risk_factors', [])])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"**MITRE Initial Access:** {target_rec.get('mitre_initial_access', 'N/A')}")
            st.markdown(f"**MITRE Impact:** {target_rec.get('mitre_impact', 'N/A')}")
            st.markdown(f"**Has Disruption:** {target_rec.get('has_disruption', 'N/A')}")
            st.markdown(f"**Zero Days:** {target_rec.get('zero_days', 'N/A')}")

    csv_bytes = res_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Filtered Incidents CSV",
        data=csv_bytes,
        file_name="filtered_security_incidents.csv",
        mime="text/csv"
    )

# ==============================================================================
# MODULE 10, 11, 12, 13: MODEL PERFORMANCE & METHODOLOGY
# ==============================================================================
elif page == "Model Performance & Methodology":
    heading("MODEL PERFORMANCE & EVALUATION", "Phishing Detection Machine Learning Validation", "Empirical evaluation on benchmark dataset (80/20 stratified split). Real metrics with no synthetic inflation.")

    meta = get_metadata()
    if meta and "model_comparison" in meta:
        comp_dict = meta["model_comparison"]
        comp_records = []
        for model_name, m in comp_dict.items():
            comp_records.append({
                "Algorithm": model_name,
                "Accuracy": f"{m['accuracy'] * 100:.2f}%",
                "Precision": f"{m['precision'] * 100:.2f}%",
                "Recall": f"{m['recall'] * 100:.2f}%",
                "F1 Score": f"{m['f1_score']:.4f}",
                "ROC-AUC": f"{m['roc_auc']:.4f}"
            })
        st.dataframe(pd.DataFrame(comp_records), use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            heading("TOP FEATURE IMPORTANCES (RANDOM FOREST)", "Gini impurity reduction across 21 extracted URL features")
            fi = meta.get("feature_importances", {})
            if fi:
                fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"]).head(12)
                fig_fi = px.bar(
                    fi_df.sort_values("Importance"),
                    x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale=["#183454", "#48d6e8"]
                )
                show_chart(fig_fi, height=360)

        with c2:
            heading("CONFUSION MATRIX (RANDOM FOREST TEST SET)", "Evaluation on 1,200 held-out test URLs")
            cm = comp_dict["Random Forest"]["confusion_matrix"]
            cm_df = pd.DataFrame(
                cm, index=["Actual Benign", "Actual Phishing"], columns=["Pred Benign", "Pred Phishing"]
            )
            fig_cm = px.imshow(
                cm_df, text_auto=True,
                color_continuous_scale=["#0e192c", "#287da1", "#48d6e8"],
                aspect="auto"
            )
            show_chart(fig_cm, height=360)

    st.markdown("---")
    heading("SYSTEM METHODOLOGY & ACADEMIC INTEGRITY", "Data provenance, analytical scoring, and design limitations")
    st.markdown("""
    ### 1. Incident Threat Intelligence (EuRepoC Global Dataset v1.3)
    - **Unit of Analysis:** Individual documented cyber incident identified by unique `incident_id`.
    - **Attribute Normalization:** Dates parsed to ISO format; multi-value delimited fields normalized into clean sets.
    - **Threat Categories:** Source categorization fields (`operation_type`, `incident_type`) are prioritized. Text regex indicators are explicitly designated as exploratory indicators.
    - **Counting Rules:** Multi-country or multi-sector incidents contribute independently to respective aggregates.

    ### 2. Threat Risk Scoring Methodology
    - **Scope:** Analytical project-specific score normalized to `[0, 100]`.
    - **Weighted Attributes:** Long-term disruption (+25), Sensitive data theft (+20), Hijacking (+15), High-impact MITRE (+15), Zero-Day usage (+15), Critical Sector targeting (+12), EuRepoC intensity rating (+10).
    - **Risk Tiers:** Low (0–30), Medium (31–60), High (61–80), Critical (81–100).
    - **Explainability:** Each score output includes specific attribute-grounded contributing factors.

    ### 3. Machine Learning Phishing Detection
    - **Pipeline:** Lexical, structural, and Shannon entropy extraction (21 dimensions) &rarr; Stratified Train/Test split (80/20) &rarr; Random Forest Classifier with probability calibration.
    - **Safety:** Offline execution only. User-submitted URLs are strictly treated as untrusted strings and are never fetched over HTTP/DNS.

    ### 4. Known Academic Limitations
    - **Reporting Bias:** Public incident repositories reflect disclosed and publicly covered events, not a complete census of all cyberattacks.
    - **Non-Causality:** Statistical anomalies and OLS historical projections indicate past trends and reporting waves; they do not forecast guaranteed real-time cyber attacks.
    """)

# ==============================================================================
# MODULE 4 & 5 (NEW): REAL-TIME SECURITY DATA / LIVE NETWORK VISUALIZATION
# ==============================================================================
elif page == "Real-Time Security Data":
    heading("LIVE NETWORK SURVEILLANCE & TRAFFIC ANALYTICS", "Real-Time Security Data Visualization", "Ingest network events from PCAP files, simulated traffic scenarios, or live packet capture with continuous anomaly and alert analytics.")

    collector = get_packet_collector()

    # Active Data Source Banner
    active_source_label = collector.active_source
    badge_style = "pill-cyan" if "PCAP" in active_source_label or "Simulated" in active_source_label else "pill-danger"
    
    st.markdown(f"""
    <div style="background: rgba(14,24,43,0.85); border:1px solid var(--line); border-radius:12px; padding:14px 18px; margin-bottom:16px;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
            <div>
                <span style="font-size:0.78rem; color:#8294ac; text-transform:uppercase; letter-spacing:0.1em; font-weight:700;">CURRENT DATA SOURCE:</span>
                <span class="pill {badge_style}" style="font-size:0.85rem; font-weight:700; margin-left:8px;">{active_source_label}</span>
                <span style="font-size:0.82rem; color:#9fb1c9; margin-left:10px;">Status: <b>{collector.get_summary_metrics()['status']}</b></span>
            </div>
            <div style="font-size:0.78rem; color:#8294ac;">
                &bull; Simulated / Replayed data is strictly isolated in memory and labeled as analytical demonstration.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Ingestion Controls & Configuration
    with st.expander("STREAM INGESTION & CAPTURE CONTROLS", expanded=True):
        c_mode, c_opts, c_acts = st.columns([1.2, 1.6, 1.2], gap="medium")

        with c_mode:
            st.markdown("<b>Select Ingestion Mode:</b>", unsafe_allow_html=True)
            source_mode = st.radio(
                "Ingestion Source",
                ["Simulated Traffic Generator", "PCAP File Replay", "Live Network Interface Capture"],
                label_visibility="collapsed",
                key="realtime_source_mode"
            )

        with c_opts:
            if source_mode == "Simulated Traffic Generator":
                st.markdown("<b>Attack / Traffic Scenario:</b>", unsafe_allow_html=True)
                sim_scenario = st.selectbox(
                    "Scenario",
                    [
                        "Normal Enterprise Traffic Mix",
                        "Port Scan Attack (Vertical Recon)",
                        "SYN Flood DDoS Attack",
                        "DNS Tunneling & C2 Exfiltration",
                        "Horizontal Subnet Sweep (SMB/SSH)",
                        "Outbound High-Volume Transfer Spike"
                    ],
                    label_visibility="collapsed"
                )
                speed_slider = st.slider("Emission Interval (seconds per batch)", 0.2, 2.0, 0.8, 0.1)

            elif source_mode == "PCAP File Replay":
                st.markdown("<b>PCAP Selection:</b>", unsafe_allow_html=True)
                pcap_choice = st.radio(
                    "PCAP Source",
                    ["Bundled Sample PCAP (sample_traffic.pcap)", "Upload Custom .pcap / .pcapng"],
                    label_visibility="collapsed"
                )
                custom_pcap_file = None
                if pcap_choice == "Upload Custom .pcap / .pcapng":
                    custom_pcap_file = st.file_uploader("Upload PCAP", type=["pcap", "pcapng", "cap"], label_visibility="collapsed")
                speed_slider = st.slider("Replay Pace (seconds per batch)", 0.2, 2.0, 0.6, 0.1)

            else:
                st.markdown("<b>Live Capture Configuration:</b>", unsafe_allow_html=True)
                st.caption(
                    "Live network capture requires administrator privileges and Npcap/WinPcap on Windows. "
                    "If unconfigured, use PCAP Replay or the Traffic Simulator."
                )
                iface_input = st.text_input("Network Interface (Leave empty for default)", placeholder="e.g., Ethernet or Wi-Fi")

        with c_acts:
            st.markdown("<b>Stream Execution:</b>", unsafe_allow_html=True)
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("START INGESTION", type="primary", use_container_width=True):
                    if source_mode == "Simulated Traffic Generator":
                        collector.start_simulation(scenario=sim_scenario, interval=speed_slider)
                        st.rerun()
                    elif source_mode == "PCAP File Replay":
                        target_pcap = custom_pcap_file if custom_pcap_file is not None else SAMPLE_PCAP_PATH
                        try:
                            collector.start_pcap_replay(target_pcap, interval=speed_slider)
                            st.rerun()
                        except Exception as exc:
                            st.error(f"PCAP Load Error: {exc}")
                    else:
                        try:
                            collector.start_live_capture(iface=iface_input.strip() or None)
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Live Capture Failed: {exc}")

                if st.button("PAUSE / RESUME", use_container_width=True):
                    if collector.is_paused:
                        collector.resume()
                    else:
                        collector.pause()
                    st.rerun()

            with btn_col2:
                if st.button("STOP STREAM", use_container_width=True):
                    collector.stop()
                    st.rerun()

                if st.button("RESET BUFFER", use_container_width=True):
                    collector.reset_buffer()
                    st.rerun()

            # Manual single batch step
            if st.button("STEP +25 PACKETS", use_container_width=True):
                from src.realtime.traffic_simulator import generate_simulated_packets
                scen = "normal"
                if source_mode == "Simulated Traffic Generator":
                    scen = sim_scenario
                collector.ingest_batch(generate_simulated_packets(scen, count=25))
                st.rerun()

            auto_refresh = st.checkbox("Auto-refresh dashboard (2s)", value=collector.is_running, key="auto_refresh_chk")

    # Fetch Real-Time Buffer and Metrics
    metrics = collector.get_summary_metrics()
    df_stream = collector.get_dataframe(limit=2000)

    # Real-Time Metrics Deck
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6, kpi_col7, kpi_col8 = st.columns(8)
    kpi_col1.metric("PACKETS INGESTED", f"{metrics['total_packets']:,}")
    kpi_col2.metric("BYTES INGESTED", f"{metrics['total_bytes'] / 1024:.1f} KB")
    kpi_col3.metric("PACKETS / SEC", f"{metrics['pps']:.1f}")
    kpi_col4.metric("BANDWIDTH", f"{metrics['bps'] / 1024:.1f} Kbps")
    kpi_col5.metric("ACTIVE SRC IPS", f"{metrics['active_src_ips']:,}")
    kpi_col6.metric("ACTIVE DST IPS", f"{metrics['active_dst_ips']:,}")
    kpi_col7.metric("ACTIVE CONNECTIONS", f"{metrics['active_connections']:,}")
    alert_count = len(collector.alert_manager.alerts)
    kpi_col8.metric("SECURITY ALERTS", f"{alert_count:,}")

    st.markdown("---")

    # Visualizations Grid
    v_left, v_right = st.columns([1.5, 1.0], gap="large")

    with v_left:
        heading("TRAFFIC THROUGHPUT TIMELINE", "Rolling 60-Second PPS and Bandwidth", "Live rolling packet rate (PPS) and bandwidth consumption.")
        fig_timeline = plot_traffic_timeline(df_stream, window_seconds=60)
        show_chart(fig_timeline, height=330)

    with v_right:
        heading("PROTOCOL BREAKDOWN", "Observed Transport & Application Layer Mix")
        fig_proto = plot_protocol_distribution(df_stream)
        show_chart(fig_proto, height=330)

    st.markdown("---")
    ip_col1, ip_col2 = st.columns(2)
    with ip_col1:
        heading("TOP SOURCE IP TALKERS", "Ranked by transmitted packet volume (LAN vs Public tagged)")
        fig_src = plot_top_ip_talkers(df_stream, direction="src_ip", top_n=10)
        show_chart(fig_src, height=340)

    with ip_col2:
        heading("TOP DESTINATION IP TARGETS", "Ranked by received packet volume (LAN vs Public tagged)")
        fig_dst = plot_top_ip_talkers(df_stream, direction="dst_ip", top_n=10)
        show_chart(fig_dst, height=340)

    st.markdown("---")
    heading("INTERACTIVE NETWORK COMMUNICATION GRAPH", "Host Relationship Topology & Connection Density", "Nodes represent communicating hosts (sized by volume; color-coded by internal vs external/scanner role). Edges show active packet flows.")
    fig_graph = plot_network_communication_graph(df_stream, max_edges=40)
    st.plotly_chart(fig_graph, use_container_width=True)

    st.markdown("---")
    heading("GEOGRAPHIC VISUALIZATION OF EXTERNAL ENDPOINTS", "Approximate Global Distribution of Public Endpoints", "Private RFC 1918 addresses are properly identified as local LAN.")
    fig_geo = plot_realtime_geolocation_map(df_stream)
    st.plotly_chart(fig_geo, use_container_width=True)

    # Real-Time Security Alert Panel
    st.markdown("---")
    heading("REAL-TIME SECURITY ALERT CENTER", "Continuous Anomaly Detections & SOC Feed", "Generated live from traffic analysis (Port Scans, SYN Floods, Traffic Bursts, DNS Tunneling, Outbound Surges) with cooldown deduplication.")

    al_flt_col1, al_flt_col2, al_flt_col3, al_act_col = st.columns([1, 1, 1.2, 1])
    with al_flt_col1:
        al_sev = st.multiselect("Filter Severity", ["Critical", "High", "Medium", "Low"], key="rt_sev_flt")
    with al_flt_col2:
        al_cat = st.multiselect("Filter Category", ["Port Scan", "SYN Flood", "Traffic Burst", "Horizontal Scan", "DNS Anomaly", "Outbound Spike"], key="rt_cat_flt")
    with al_flt_col3:
        al_search = st.text_input("Search Alert IP", placeholder="e.g., 45.154 or 192.168", key="rt_search_ip")
    with al_act_col:
        st.write("")
        st.write("")
        if st.button("MARK ALL REVIEWED", use_container_width=True):
            collector.alert_manager.mark_all_reviewed()
            st.rerun()

    alerts_df = collector.alert_manager.get_dataframe(
        severity_filter=al_sev or None,
        category_filter=al_cat or None,
        search_ip=al_search
    )

    if not alerts_df.empty:
        for _, al in alerts_df.head(6).iterrows():
            sev = al.get("severity", "Medium")
            cls_name = sev.lower()
            color = "#ff4d6d" if sev == "Critical" else ("#ff8552" if sev == "High" else "#f3b76b")
            st.markdown(f"""
            <div class="alert-card {cls_name}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:700; color:{color}; font-size:0.75rem; letter-spacing:0.1em;">{al['alert_id']} &bull; {sev.upper()} &bull; {al['category']} &bull; Status: {al['status']}</span>
                    <span style="font-size:0.72rem; color:#8294ac;">{al.get('time_str', 'N/A')} ({al['data_source']})</span>
                </div>
                <div style="font-weight:600; font-size:0.95rem; color:#edf4ff; margin:3px 0 2px;">{html.escape(al['summary'])}</div>
                <div style="font-size:0.84rem; color:#a9bdd5;">{html.escape(al['explanation'])}</div>
                <div style="font-size:0.78rem; color:#79e5f3; margin-top:4px;"><b>Evidence:</b> {html.escape(al['evidence'])} &bull; <b>Endpoints:</b> {html.escape(str(al['src_ip']))} &rarr; {html.escape(str(al['dst_ip']))}</div>
            </div>
            """, unsafe_allow_html=True)

        al_csv = alerts_df.to_csv(index=False).encode("utf-8")
        st.download_button("Export Alerts CSV", data=al_csv, file_name="realtime_security_alerts.csv", mime="text/csv")
    else:
        st.info("No security alerts triggered under current filters. Trigger an attack scenario in the controls above to observe live alerts.")

    # Live Packet Buffer Stream Explorer
    st.markdown("---")
    heading("LIVE PACKET BUFFER INSPECTOR", "Sliding Buffer of Recent Captured Packets", "Inspect raw metadata fields (Protocols, TCP Flags, Port Pairs, Payload Lengths).")
    
    if not df_stream.empty:
        p_flt = st.selectbox("Filter Packet Protocol", ["All Protocols"] + sorted(df_stream["protocol"].unique().tolist()))
        show_df = df_stream if p_flt == "All Protocols" else df_stream[df_stream["protocol"] == p_flt]
        
        display_packet_cols = [
            "time_str", "protocol", "src_ip", "src_port", "dst_ip", "dst_port",
            "length", "tcp_flags", "info", "source_type"
        ]
        st.dataframe(show_df[display_packet_cols].tail(100).iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.info("Buffer is currently empty. Click 'START INGESTION' or 'STEP +25 PACKETS' above to populate the stream.")

    # Handle Auto-refresh loop safely
    if auto_refresh and collector.is_running and not collector.is_paused:
        time.sleep(2.0)
        st.rerun()

st.markdown("---")
st.markdown(
    '<div class="small-note" style="text-align:center;">Security Intelligence &amp; Real-Time Threat Platform &bull; EuRepoC Global Dataset v1.3 &bull; Built with Streamlit, Plotly, Scapy, and Scikit-Learn</div>',
    unsafe_allow_html=True
)
