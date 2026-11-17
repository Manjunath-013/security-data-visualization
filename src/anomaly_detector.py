"""Historical Anomaly Detection Engine for Cybersecurity Incident Datasets.
Uses statistical time-series outlier detection (Rolling Baseline, Z-score, IQR)
to identify anomalous surges in incident volume, DDoS campaigns,
ransomware waves, and target sector concentrations.

All detections are explicitly labeled as Historical Anomaly Detections.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd


def detect_volume_anomalies(
    df: pd.DataFrame, freq: str = "year", z_threshold: float = 1.75
) -> List[Dict[str, Any]]:
    """
    Detect statistical outliers in overall incident volume over time.
    """
    if df.empty or "start_date" not in df.columns:
        return []

    valid = df.dropna(subset=["start_date"]).copy()
    if valid.empty:
        return []

    if freq == "year":
        grouped = valid.groupby("year")["incident_id"].nunique().sort_index()
    else:
        valid["period_key"] = valid["start_date"].dt.to_period("M").astype(str)
        grouped = valid.groupby("period_key")["incident_id"].nunique().sort_index()

    if len(grouped) < 3:
        return []

    values = grouped.values.astype(float)
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return []

    q25 = np.percentile(values, 25)
    q75 = np.percentile(values, 75)
    iqr = q75 - q25
    upper_iqr = q75 + 1.5 * iqr

    anomalies: List[Dict[str, Any]] = []
    for period, count in grouped.items():
        z = (count - mean) / std
        if z >= z_threshold or count > upper_iqr:
            severity = "CRITICAL" if z >= 2.5 else "HIGH"
            expected_low = max(0, int(round(mean - std)))
            expected_high = int(round(mean + 1.2 * std))
            anomalies.append({
                "period": str(period),
                "metric": "Overall Incident Volume Surge",
                "observed_value": int(count),
                "expected_range": f"{expected_low} – {expected_high}",
                "z_score": round(float(z), 2),
                "severity": severity,
                "category": "Volume Surge",
                "explanation": (
                    f"Reported incident count ({int(count)}) sharply exceeded the baseline "
                    f"mean ({int(round(mean))}) with a statistical Z-score of +{z:.2f}. "
                    f"Represents a significant spike in historical reporting activity."
                )
            })

    return sorted(anomalies, key=lambda x: x["z_score"], reverse=True)


def detect_threat_type_anomalies(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detect anomalous surges in specific attack vectors (DDoS and Ransomware) across years.
    """
    if df.empty or "year" not in df.columns:
        return []

    d = df.dropna(subset=["year"]).copy()
    if d.empty:
        return []

    annual = d.groupby("year").agg(
        total=("incident_id", "nunique"),
        ddos=("is_ddos", "sum"),
        ransomware=("is_ransomware", "sum")
    ).reset_index().sort_values("year")

    anomalies: List[Dict[str, Any]] = []

    # 1. DDoS Spike Detection
    ddos_vals = annual["ddos"].values.astype(float)
    if len(ddos_vals) >= 4:
        d_mean, d_std = np.mean(ddos_vals), np.std(ddos_vals)
        if d_std > 0:
            for _, row in annual.iterrows():
                z = (row["ddos"] - d_mean) / d_std
                if z >= 1.75 and row["ddos"] >= 20:
                    anomalies.append({
                        "period": str(int(row["year"])),
                        "metric": "DDoS Threat Surge",
                        "observed_value": int(row["ddos"]),
                        "expected_range": f"{max(0, int(round(d_mean - d_std)))} – {int(round(d_mean + d_std))}",
                        "z_score": round(float(z), 2),
                        "severity": "CRITICAL" if z >= 2.2 else "HIGH",
                        "category": "DDoS Outlier",
                        "explanation": (
                            f"Recorded {int(row['ddos'])} DDoS-related incidents in {int(row['year'])}, "
                            f"exceeding the historical annual average of {d_mean:.1f} (Z-score: +{z:.2f})."
                        )
                    })

    # 2. Ransomware Surge Detection
    ransom_vals = annual["ransomware"].values.astype(float)
    if len(ransom_vals) >= 4:
        r_mean, r_std = np.mean(ransom_vals), np.std(ransom_vals)
        if r_std > 0:
            for _, row in annual.iterrows():
                z = (row["ransomware"] - r_mean) / r_std
                if z >= 1.75 and row["ransomware"] >= 25:
                    anomalies.append({
                        "period": str(int(row["year"])),
                        "metric": "Ransomware Escalation",
                        "observed_value": int(row["ransomware"]),
                        "expected_range": f"{max(0, int(round(r_mean - r_std)))} – {int(round(r_mean + r_std))}",
                        "z_score": round(float(z), 2),
                        "severity": "CRITICAL" if z >= 2.2 else "HIGH",
                        "category": "Ransomware Outlier",
                        "explanation": (
                            f"Ransomware activity spiked to {int(row['ransomware'])} recorded incidents in {int(row['year'])}, "
                            f"surpassing baseline levels ({r_mean:.1f} annual average, Z-score: +{z:.2f})."
                        )
                    })

    return sorted(anomalies, key=lambda x: x["z_score"], reverse=True)


def detect_sector_targeting_anomalies(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detect statistical concentration anomalies in target sector distributions.
    """
    if df.empty or "sector_list" not in df.columns:
        return []

    exploded = df[["incident_id", "sector_list"]].explode("sector_list").dropna()
    exploded["sector_list"] = exploded["sector_list"].astype(str).str.strip()
    clean = exploded[exploded["sector_list"].ne("") & ~exploded["sector_list"].str.lower().isin(
        {"not available", "unknown", "none"}
    )]

    if clean.empty:
        return []

    counts = clean.groupby("sector_list")["incident_id"].nunique().sort_values(ascending=False)
    if len(counts) < 5:
        return []

    mean_count = counts.mean()
    std_count = counts.std()
    anomalies: List[Dict[str, Any]] = []

    # Check top sectors for extreme outlier concentration (Z > 2.5)
    for sector, count in counts.head(5).items():
        z = (count - mean_count) / std_count
        if z >= 2.5:
            anomalies.append({
                "period": "Overall Dataset Distribution",
                "metric": f"Sector Targeting Outlier: {sector}",
                "observed_value": int(count),
                "expected_range": f"{max(0, int(round(mean_count - std_count)))} – {int(round(mean_count + std_count))}",
                "z_score": round(float(z), 2),
                "severity": "HIGH",
                "category": "Target Concentration",
                "explanation": (
                    f"'{sector}' accounts for {int(count)} incidents, representing an extreme "
                    f"statistical concentration (+{z:.2f} standard deviations above the sector mean of {mean_count:.1f})."
                )
            })

    return anomalies


def run_all_anomaly_detections(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Run the unified anomaly detection suite and return consolidated records.
    """
    results: List[Dict[str, Any]] = []
    results.extend(detect_volume_anomalies(df, freq="year"))
    results.extend(detect_threat_type_anomalies(df))
    results.extend(detect_sector_targeting_anomalies(df))
    return results
