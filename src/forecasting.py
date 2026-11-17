"""Statistical Trend Forecasting Module.
Provides linear and moving-average projection of historical attack volume
(total incidents, DDoS, ransomware) with statistical confidence intervals.

IMPORTANT: Strictly labeled as 'Historical Trend Forecast' for exploratory
analytical projection; not a predictive oracle or guarantee of future incidents.
"""

from typing import Any, Dict
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def calculate_trend_forecast(
    df: pd.DataFrame, target_metric: str = "total", forecast_years: int = 3
) -> Dict[str, Any]:
    """
    Fit an ordinary least-squares trend with empirical residual variance
    to project historical trends forward for the requested metric.
    """
    if df.empty or "year" not in df.columns:
        return {}

    d = df.dropna(subset=["year"]).copy()
    if d.empty:
        return {}

    annual = d.groupby("year").agg(
        total=("incident_id", "nunique"),
        ddos=("is_ddos", "sum"),
        ransomware=("is_ransomware", "sum"),
        malware=("is_malware_related", "sum")
    ).reset_index().sort_values("year")

    # Filter out extremely early years with sparse recording (e.g., pre-2015) for cleaner baseline
    annual_recent = annual[annual["year"] >= 2012]
    if len(annual_recent) < 5:
        annual_recent = annual

    metric_col = target_metric.lower()
    if metric_col not in annual_recent.columns:
        metric_col = "total"

    x = annual_recent["year"].values.astype(float)
    y = annual_recent[metric_col].values.astype(float)

    # Linear regression slope & intercept
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residuals = y - fitted
    std_err = np.std(residuals) if len(residuals) > 1 else 10.0

    last_year = int(x[-1])
    future_x = np.array([last_year + i for i in range(1, forecast_years + 1)], dtype=float)
    future_y = slope * future_x + intercept
    future_y = np.maximum(future_y, 0)  # Cannot be negative

    # 80% Confidence Interval (+- 1.28 * std_err)
    upper_ci = future_y + 1.28 * std_err
    lower_ci = np.maximum(0, future_y - 1.28 * std_err)

    hist_df = pd.DataFrame({"year": x.astype(int), "observed": y})
    proj_df = pd.DataFrame({
        "year": future_x.astype(int),
        "projected": np.round(future_y, 1),
        "lower_bound": np.round(lower_ci, 1),
        "upper_bound": np.round(upper_ci, 1)
    })

    return {
        "metric": metric_col,
        "slope": round(float(slope), 2),
        "historical": hist_df,
        "projection": proj_df,
        "std_error": round(float(std_err), 2)
    }


def plot_forecast_chart(forecast_data: Dict[str, Any], title: str = "Trend Forecast") -> go.Figure:
    """Generate interactive Plotly visualization for historical forecast."""
    fig = go.Figure()

    if not forecast_data:
        fig.update_layout(title="Insufficient data for forecasting")
        return fig

    hist = forecast_data["historical"]
    proj = forecast_data["projection"]

    # Historical observations
    fig.add_trace(go.Scatter(
        x=hist["year"],
        y=hist["observed"],
        mode="lines+markers",
        name="Historical Observed",
        line=dict(color="#48d6e8", width=3),
        marker=dict(size=7, color="#48d6e8")
    ))

    # Connect last historical point to projection
    conn_x = [hist["year"].iloc[-1], proj["year"].iloc[0]]
    conn_y = [hist["observed"].iloc[-1], proj["projected"].iloc[0]]
    fig.add_trace(go.Scatter(
        x=conn_x,
        y=conn_y,
        mode="lines",
        showlegend=False,
        line=dict(color="#ffb86b", width=2, dash="dot")
    ))

    # Projected line
    fig.add_trace(go.Scatter(
        x=proj["year"],
        y=proj["projected"],
        mode="lines+markers",
        name="Projected Trend (OLS)",
        line=dict(color="#ffb86b", width=3, dash="dash"),
        marker=dict(size=8, symbol="diamond", color="#ffb86b")
    ))

    # Confidence Interval Ribbon
    fig.add_trace(go.Scatter(
        x=proj["year"].tolist() + proj["year"].tolist()[::-1],
        y=proj["upper_bound"].tolist() + proj["lower_bound"].tolist()[::-1],
        fill="toself",
        fillcolor="rgba(255, 184, 107, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="80% Statistical Uncertainty Range"
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(family="Space Grotesk", size=14, color="#edf4ff")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", size=11, color="#b8c8dc"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.12)", dtick=1),
        yaxis=dict(gridcolor="rgba(148,163,184,0.12)", title="Annual Incident Count"),
        legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=15, r=15, t=45, b=15)
    )
    return fig
