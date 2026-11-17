"""Analytical Threat Risk Scoring Engine.
Computes a normalized 0-100 Analytical Risk Score for cyber incidents
based on verified dataset attributes (disruption, data theft, hijacking,
zero-day usage, critical sector targeting, and MITRE impact techniques).

Scoring scale:
  0 - 30   : LOW
  31 - 60  : MEDIUM
  61 - 80  : HIGH
  81 - 100 : CRITICAL
"""

from typing import Any, Dict, List
import pandas as pd

CRITICAL_SECTORS = {
    "government / ministries", "civil service / administration",
    "defence industry", "military", "energy", "health",
    "finance", "telecommunications", "water", "transportation",
    "police", "judiciary"
}

HIGH_IMPACT_MITRE = {
    "disk wipe", "data destruction", "data encrypted for impact",
    "network denial of service", "endpoint denial of service",
    "defacement", "data manipulation", "inhibit system recovery"
}


def evaluate_incident_risk(row: pd.Series) -> Dict[str, Any]:
    """
    Evaluate an individual incident record and compute its Analytical Risk Score
    along with transparent, human-readable contributing factors.
    """
    score = 15  # Baseline score for publicly recorded cyber incidents
    factors: List[str] = []

    # 1. Disruption Severity
    disruption_str = str(row.get("disruption", "")).lower()
    has_disruption = bool(row.get("has_disruption", False))
    if "long-term" in disruption_str or "> 24h" in disruption_str:
        score += 25
        factors.append("Extended operational disruption (> 24 hours)")
    elif "short-term" in disruption_str or "< 24h" in disruption_str or has_disruption:
        score += 15
        factors.append("Operational disruption recorded")

    # 2. Data Theft / Exfiltration
    data_theft_str = str(row.get("data_theft", "")).lower()
    if "sensitive information" in data_theft_str:
        score += 20
        factors.append("Exfiltration of sensitive confidential or classified records")
    elif "non-sensitive" in data_theft_str or row.get("has_data_theft", False):
        score += 12
        factors.append("Confirmatory data theft / exfiltration recorded")

    # 3. System Hijacking
    hijacking_str = str(row.get("hijacking", "")).lower()
    if "hijacking" in hijacking_str and "not available" not in hijacking_str and "none" not in hijacking_str:
        score += 15
        factors.append("Host/system hijacking or infrastructure takeover")

    # 4. Zero-Day Exploit
    zero_days_str = str(row.get("zero_days", "")).lower()
    if "yes" in zero_days_str or row.get("has_zero_day", False):
        score += 15
        factors.append("Exploitation of unknown Zero-Day vulnerability")

    # 5. High-Impact MITRE Techniques
    mitre_impacts = row.get("mitre_impact_list", [])
    if isinstance(mitre_impacts, list):
        matched_techniques = [
            t for t in mitre_impacts if any(hit in t.lower() for hit in HIGH_IMPACT_MITRE)
        ]
        if matched_techniques:
            score += 15
            factors.append(f"High-impact MITRE techniques: {', '.join(matched_techniques[:3])}")

    # 6. Critical Sector Targeting
    sectors = row.get("sector_list", [])
    if isinstance(sectors, list):
        matched_sectors = [s for s in sectors if s.lower() in CRITICAL_SECTORS]
        if matched_sectors:
            score += 12
            factors.append(f"Targeted critical infrastructure: {', '.join(matched_sectors[:2])}")

    # 7. Weighted Intensity Factor
    w_intensity = row.get("weighted_intensity", None)
    if pd.notna(w_intensity):
        try:
            val = float(w_intensity)
            if val >= 6.0:
                score += 10
                factors.append(f"High EuRepoC intensity rating ({val:.1f})")
            elif val >= 3.0:
                score += 5
        except (ValueError, TypeError):
            pass

    # 8. Physical / Spatial Effects
    physical = str(row.get("physical_effects_spatial", "")).lower()
    if physical and physical not in {"none", "not available", "unknown", "nan"}:
        score += 15
        factors.append("Physical or spatial real-world effects observed")

    # 9. Attributed State / APT Actor
    initiator_cat = str(row.get("initiator_category", "")).lower()
    if "state" in initiator_cat and "non-state" not in initiator_cat:
        score += 6
        factors.append("Attributed to nation-state actor or state-sponsored APT")

    # 10. Multi-Country / Spillover Impact
    countries = row.get("country_list", [])
    if isinstance(countries, list) and len(countries) > 1:
        score += 6
        factors.append(f"International cross-border impact ({len(countries)} countries affected)")

    # Normalize bounds [0, 100]
    final_score = max(5, min(100, int(score)))

    if final_score <= 30:
        level = "LOW"
    elif final_score <= 60:
        level = "MEDIUM"
    elif final_score <= 80:
        level = "HIGH"
    else:
        level = "CRITICAL"

    if not factors:
        factors.append("Standard exploratory cyber event; no high-severity indicators reported")

    return {
        "risk_score": final_score,
        "risk_level": level,
        "contributing_factors": factors
    }


def compute_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized and batch calculation of Analytical Risk Scores for the entire dataset.
    Adds `risk_score`, `risk_level`, and `risk_factors` columns to the DataFrame.
    """
    if df.empty:
        df["risk_score"] = pd.Series(dtype=int)
        df["risk_level"] = pd.Series(dtype=str)
        df["risk_factors"] = pd.Series(dtype=object)
        return df

    out_df = df.copy()
    scores = []
    levels = []
    factors_list = []

    for _, row in out_df.iterrows():
        eval_res = evaluate_incident_risk(row)
        scores.append(eval_res["risk_score"])
        levels.append(eval_res["risk_level"])
        factors_list.append(eval_res["contributing_factors"])

    out_df["risk_score"] = scores
    out_df["risk_level"] = levels
    out_df["risk_factors"] = factors_list
    return out_df
