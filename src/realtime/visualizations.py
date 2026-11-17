"""Visualizations for Real-Time Network Security Dashboard.
Provides rolling time-series charts (PPS/BPS), protocol breakdowns,
top IP rankings, interactive network communication graph, and geolocation maps.
"""

from collections import defaultdict
from typing import Dict, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.realtime.event_processor import get_ip_geolocation, is_private_ip


def plot_traffic_timeline(df: pd.DataFrame, window_seconds: int = 60) -> go.Figure:
    """Plot Packets-per-Second (PPS) and Bandwidth (Kbps) over rolling time window."""
    fig = go.Figure()

    if df.empty or "timestamp" not in df.columns:
        fig.update_layout(title="No real-time traffic captured yet")
        return fig

    # Group by second
    df_copy = df.copy()
    latest_ts = df_copy["timestamp"].max()
    min_ts = latest_ts - window_seconds
    windowed = df_copy[df_copy["timestamp"] >= min_ts].copy()

    if windowed.empty:
        windowed = df_copy.tail(100).copy()

    windowed["sec"] = windowed["timestamp"].astype(int)
    agg = windowed.groupby("sec").agg(
        pps=("length", "count"),
        bytes_total=("length", "sum"),
        time_str=("time_str", "last")
    ).reset_index().sort_values("sec")

    agg["kbps"] = (agg["bytes_total"] * 8) / 1024.0

    # Packets per second trace
    fig.add_trace(go.Scatter(
        x=agg["time_str"],
        y=agg["pps"],
        mode="lines+markers",
        name="Packets / Sec",
        line=dict(color="#48d6e8", width=2.5),
        marker=dict(size=5, color="#48d6e8"),
        yaxis="y1"
    ))

    # Bandwidth trace
    fig.add_trace(go.Scatter(
        x=agg["time_str"],
        y=agg["kbps"],
        mode="lines",
        name="Bandwidth (Kbps)",
        line=dict(color="#ff8552", width=2, dash="dot"),
        yaxis="y2"
    ))

    fig.update_layout(
        title=dict(text="Live Network Traffic Rate & Bandwidth (Rolling 60s)", font=dict(family="Space Grotesk", size=13, color="#edf4ff")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", size=11, color="#b8c8dc"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.12)", title="Timestamp"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.12)", title="Packets / Sec", side="left"),
        yaxis2=dict(
            title="Bandwidth (Kbps)",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=15, r=15, t=40, b=15)
    )
    return fig


def plot_protocol_distribution(df: pd.DataFrame) -> go.Figure:
    """Plot observed protocol breakdown."""
    if df.empty or "protocol" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="No protocol data")
        return fig

    proto_counts = df["protocol"].value_counts().reset_index()
    proto_counts.columns = ["Protocol", "Packets"]

    color_map = {
        "HTTPS": "#48d6e8",
        "HTTP": "#ffb86b",
        "DNS": "#8b9eff",
        "TCP": "#43d6a2",
        "UDP": "#364761",
        "ICMP": "#ff4d6d",
        "SMB": "#ff8552",
        "SSH": "#9c91ff"
    }

    fig = px.pie(
        proto_counts,
        names="Protocol",
        values="Packets",
        hole=0.55,
        color="Protocol",
        color_discrete_map=color_map
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", size=11, color="#b8c8dc"),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    return fig


def plot_top_ip_talkers(df: pd.DataFrame, direction: str = "src_ip", top_n: int = 10) -> go.Figure:
    """Plot top communicating IP addresses by packet count."""
    title_label = "Source IPs" if direction == "src_ip" else "Destination IPs"
    if df.empty or direction not in df.columns:
        fig = go.Figure()
        fig.update_layout(title=f"No {title_label} data")
        return fig

    counts = df[direction].value_counts().head(top_n).reset_index()
    counts.columns = ["IP Address", "Packets"]

    # Color code private vs external
    counts["Scope"] = counts["IP Address"].apply(lambda ip: "Internal LAN" if is_private_ip(ip) else "External / Public")

    fig = px.bar(
        counts.sort_values("Packets"),
        x="Packets",
        y="IP Address",
        orientation="h",
        color="Scope",
        color_discrete_map={"Internal LAN": "#48d6e8", "External / Public": "#ff8552"},
        text="Packets"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        title=dict(text=f"Top {title_label}", font=dict(family="Space Grotesk", size=13, color="#edf4ff")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", size=11, color="#b8c8dc"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.12)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.12)"),
        margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.12)
    )
    return fig


def plot_network_communication_graph(df: pd.DataFrame, max_edges: int = 40) -> go.Figure:
    """
    Construct an interactive 2D node-link network communication graph
    visualizing communicating IP hosts and their transmission volume.
    """
    fig = go.Figure()

    if df.empty or "src_ip" not in df.columns:
        fig.update_layout(title="No network connections to graph")
        return fig

    # Aggregate connection volumes between pairs
    edges_df = df.groupby(["src_ip", "dst_ip"]).agg(
        packets=("length", "count"),
        bytes=("length", "sum"),
        proto=("protocol", lambda x: x.mode()[0] if not x.empty else "MIXED")
    ).reset_index().nlargest(max_edges, "packets")

    unique_ips = list(set(edges_df["src_ip"].tolist() + edges_df["dst_ip"].tolist()))
    if not unique_ips:
        fig.update_layout(title="No connection edges found")
        return fig

    # Layout nodes in a clean circular topology
    import math
    num_nodes = len(unique_ips)
    node_positions = {}
    radius = 1.0
    for idx, ip in enumerate(unique_ips):
        angle = (2 * math.pi * idx) / num_nodes
        node_positions[ip] = (radius * math.cos(angle), radius * math.sin(angle))

    # Draw edges
    edge_x = []
    edge_y = []
    for _, r in edges_df.iterrows():
        x0, y0 = node_positions[r["src_ip"]]
        x1, y1 = node_positions[r["dst_ip"]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1.2, color="rgba(72, 214, 232, 0.3)"),
        hoverinfo="none",
        showlegend=False
    ))

    # Draw nodes
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []

    ip_volumes = defaultdict(int)
    for _, r in edges_df.iterrows():
        ip_volumes[r["src_ip"]] += r["packets"]
        ip_volumes[r["dst_ip"]] += r["packets"]

    for ip in unique_ips:
        x, y = node_positions[ip]
        node_x.append(x)
        node_y.append(y)
        
        is_priv = is_private_ip(ip)
        vol = ip_volumes[ip]
        
        if is_priv:
            color = "#48d6e8"
            scope_str = "Internal Private Host"
        elif "45.154" in ip or "194.26" in ip:
            color = "#ff4d6d"
            scope_str = "External Scanning / Suspicious Host"
        else:
            color = "#ff8552"
            scope_str = "External Public Server"

        node_color.append(color)
        node_size.append(min(32, max(14, int(math.sqrt(vol) * 3))))
        node_text.append(f"<b>Host IP:</b> {ip}<br><b>Scope:</b> {scope_str}<br><b>Active Packets:</b> {vol:,}")

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=[ip.split(".")[-1] if is_private_ip(ip) else ip.split(".")[0] + "..." for ip in unique_ips],
        textposition="top center",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1.5, color="#ffffff")
        ),
        hoverinfo="text",
        hovertext=node_text,
        showlegend=False
    ))

    fig.update_layout(
        title=dict(text="Network Communication Topology Graph", font=dict(family="Space Grotesk", size=13, color="#edf4ff")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", size=10, color="#b8c8dc"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=35, b=10)
    )
    return fig


def plot_realtime_geolocation_map(df: pd.DataFrame) -> go.Figure:
    """Plot approximate geolocation markers for public IP endpoints."""
    fig = go.Figure()

    if df.empty:
        fig.update_layout(title="No geolocation data available")
        return fig

    # Collect unique public IPs
    all_ips = set(df["src_ip"].unique().tolist() + df["dst_ip"].unique().tolist())
    geo_records = []
    for ip in all_ips:
        info = get_ip_geolocation(ip)
        if not info["is_private"] and info["lat"] is not None:
            # Count packets involving this IP
            pkts = len(df[(df["src_ip"] == ip) | (df["dst_ip"] == ip)])
            geo_records.append({
                "ip": ip,
                "entity": info["entity"],
                "country": info["country"],
                "lat": info["lat"],
                "lon": info["lon"],
                "packets": pkts
            })

    if not geo_records:
        fig.update_layout(
            title="All observed IP endpoints are Internal LAN (RFC 1918) addresses without public geo-coordinates.",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig

    geo_df = pd.DataFrame(geo_records)

    fig = px.scatter_geo(
        geo_df,
        lat="lat",
        lon="lon",
        hover_name="ip",
        hover_data=["entity", "country", "packets"],
        size="packets",
        color="country",
        projection="natural earth",
        title="Approximate Geolocation of External Communicating Endpoints"
    )
    fig.update_geos(
        bgcolor="rgba(0,0,0,0)",
        showcountries=True,
        countrycolor="rgba(148,163,184,0.2)",
        landcolor="#121e33",
        lakecolor="rgba(0,0,0,0)"
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", size=11, color="#b8c8dc"),
        margin=dict(l=10, r=10, t=35, b=10)
    )
    return fig
