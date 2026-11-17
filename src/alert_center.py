"""Security Alert Center Logic.
Generates deterministic, data-grounded SOC alerts based on evaluated risk scores,
statistical anomaly detections, and active filter thresholds.
Never hallucinates or generates mock alerts without factual underlying data.
"""

from typing import Any, Dict, List
import pandas as pd


def generate_security_alerts(
    df: pd.DataFrame, anomalies: List[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Generate prioritized security alerts from the currently filtered incident dataset
    and detected anomalies.
    """
    alerts: List[Dict[str, Any]] = []

    if df.empty:
        return alerts

    # 1. Critical Incidents with Risk Score >= 81
    if "risk_score" in df.columns:
        critical_df = df[df["risk_score"] >= 81].sort_values("risk_score", ascending=False)
        for _, row in critical_df.head(4).iterrows():
            rec_countries = ", ".join(row.get("country_list", [])[:2]) or "Undisclosed"
            factors = row.get("risk_factors", [])
            factor_summary = factors[0] if factors else "High intensity impact"
            alerts.append({
                "severity": "CRITICAL",
                "title": f"Critical Incident: {row.get('name', 'Incident ' + str(row['incident_id']))}",
                "category": "Incident Risk",
                "period": str(row.get("year", "N/A")),
                "message": (
                    f"Analytical Risk Score {int(row['risk_score'])}/100. Target: {rec_countries}. "
                    f"Key trigger: {factor_summary}."
                ),
                "incident_id": row["incident_id"],
                "color": "#ff4d6d"
            })

    # 2. Anomalies from Anomaly Detector
    if anomalies:
        for anom in anomalies[:4]:
            alerts.append({
                "severity": anom.get("severity", "HIGH"),
                "title": f"Statistical Anomaly: {anom.get('metric', 'Volume Outlier')}",
                "category": "Temporal Anomaly",
                "period": anom.get("period", "Historical"),
                "message": anom.get("explanation", ""),
                "incident_id": None,
                "color": "#ff8552" if anom.get("severity") == "HIGH" else "#ff4d6d"
            })

    # 3. Critical Infrastructure Surge
    if "sector_list" in df.columns:
        energy_health = df[df["sector_list"].apply(
            lambda s: any(x in ["energy", "health", "defence industry"] for x in [str(i).lower() for i in s])
        )]
        if len(energy_health) >= 50:
            alerts.append({
                "severity": "HIGH",
                "title": "Critical Infrastructure Targeting Concentration",
                "category": "Sector Intelligence",
                "period": "Active Selection",
                "message": (
                    f"Detected {len(energy_health):,} incidents targeting high-priority critical sectors "
                    "(Energy, Healthcare, or Defence Industry)."
                ),
                "incident_id": None,
                "color": "#f3b76b"
            })

    # 4. Zero-Day Vulnerability Alerts
    if "has_zero_day" in df.columns:
        zdays = df[df["has_zero_day"] == True]
        if len(zdays) > 0:
            alerts.append({
                "severity": "MEDIUM",
                "title": f"Zero-Day Exploitation Evidence ({len(zdays)} Incidents)",
                "category": "Vulnerability Intelligence",
                "period": "Dataset Records",
                "message": (
                    f"{len(zdays)} recorded incidents involve confirmed or suspected Zero-Day exploit usage."
                ),
                "incident_id": None,
                "color": "#48d6e8"
            })

    # Sort: CRITICAL first, then HIGH, then MEDIUM
    sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "INFORMATIONAL": 3}
    return sorted(alerts, key=lambda x: sev_order.get(x["severity"], 99))
