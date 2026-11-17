"""Analytical and Aggregation Functions for Cybersecurity Incident Intelligence.
Provides data aggregation for attack categories, temporal trends, geographical
concentrations, sector distributions, and automated SOC intelligence synthesis.
"""

from typing import Any, Dict, List
import pandas as pd


def _empty(columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def attack_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute high-level incident metrics across attack vectors and impact."""
    if df.empty:
        return _empty(["Metric", "Incidents"])

    crit_count = int((df["risk_level"] == "CRITICAL").sum()) if "risk_level" in df.columns else 0
    high_count = int((df["risk_level"] == "HIGH").sum()) if "risk_level" in df.columns else 0
    avg_risk = round(float(df["risk_score"].mean()), 1) if "risk_score" in df.columns else 0.0

    return pd.DataFrame({
        "Metric": [
            "Total incidents",
            "Critical threat incidents",
            "High risk incidents",
            "DDoS-related incidents",
            "Ransomware-related incidents",
            "Malware-related incidents",
            "Both DDoS and ransomware flags",
            "Incidents with disruption",
            "Average Analytical Risk Score"
        ],
        "Incidents": [
            df.incident_id.nunique(),
            crit_count,
            high_count,
            int(df.is_ddos.sum()),
            int(df.is_ransomware.sum()),
            int(df.is_malware_related.sum()),
            int((df.is_ddos & df.is_ransomware).sum()),
            int(df.has_disruption.sum()),
            avg_risk
        ]
    })


def yearly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate incident counts and attack types by calendar year."""
    d = df.dropna(subset=["year"])
    if d.empty:
        return _empty(["year", "Total", "DDoS-related", "Ransomware-related", "Malware-related"])
    return (
        d.groupby("year")
        .agg(
            Total=("incident_id", "nunique"),
            **{
                "DDoS-related": ("is_ddos", "sum"),
                "Ransomware-related": ("is_ransomware", "sum"),
                "Malware-related": ("is_malware_related", "sum"),
            }
        )
        .reset_index()
        .sort_values("year")
    )


def monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate incidents by month of year (1-12)."""
    d = df.dropna(subset=["month"])
    if d.empty:
        return _empty(["month", "Incidents"])
    return (
        d.groupby("month")
        .agg(Incidents=("incident_id", "nunique"))
        .reindex(range(1, 13), fill_value=0)
        .rename_axis("month")
        .reset_index()
    )


def country_attack_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Breakdown receiver countries with associated attack vectors."""
    cols = ["Country", "Incidents", "DDoS-related", "Ransomware-related", "Malware-related"]
    if df.empty:
        return _empty(cols)
    d = (
        df[["incident_id", "country_list", "is_ddos", "is_ransomware", "is_malware_related"]]
        .explode("country_list")
        .dropna(subset=["country_list"])
    )
    d = d.drop_duplicates(["incident_id", "country_list"])
    if d.empty:
        return _empty(cols)
    return (
        d.groupby("country_list")
        .agg(
            Incidents=("incident_id", "nunique"),
            **{
                "DDoS-related": ("is_ddos", "sum"),
                "Ransomware-related": ("is_ransomware", "sum"),
                "Malware-related": ("is_malware_related", "sum"),
            }
        )
        .reset_index()
        .rename(columns={"country_list": "Country"})
        .sort_values("Incidents", ascending=False)
    )


def country_analysis(df: pd.DataFrame) -> pd.DataFrame:
    return country_attack_analysis(df)[["Country", "Incidents"]]


def _explode_count(df: pd.DataFrame, field: str, output: str) -> pd.DataFrame:
    if df.empty or field not in df:
        return _empty([output, "Incidents"])
    d = df[["incident_id", field]].explode(field)
    d[field] = d[field].fillna("").astype(str).str.strip()
    d = d[
        d[field].ne("")
        & ~d[field].str.casefold().isin({"not available", "unknown", "n/a", "nan", "none"})
    ].drop_duplicates(["incident_id", field])
    if d.empty:
        return _empty([output, "Incidents"])
    return (
        d.groupby(field)
        .agg(Incidents=("incident_id", "nunique"))
        .reset_index()
        .rename(columns={field: output})
        .sort_values("Incidents", ascending=False)
    )


def sector_analysis(df: pd.DataFrame) -> pd.DataFrame:
    return _explode_count(df, "sector_list", "Target subcategory")


def impact_analysis(df: pd.DataFrame) -> pd.DataFrame:
    return _explode_count(df, "mitre_impact_list", "Impact technique")


def category_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze attack categories overlap (DDoS vs Ransomware)."""
    if df.empty:
        return _empty(["Category", "Incidents"])
    d, r = df.is_ddos.astype(bool), df.is_ransomware.astype(bool)
    return pd.DataFrame({
        "Category": ["DDoS only", "Ransomware only", "Both", "Neither"],
        "Incidents": [
            int((d & ~r).sum()),
            int((r & ~d).sum()),
            int((d & r).sum()),
            int((~d & ~r).sum())
        ]
    })


def year_month_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Generate Year x Month pivot table for incident heatmaps."""
    d = df.dropna(subset=["year", "month"])
    if d.empty:
        return pd.DataFrame()
    return (
        d.groupby(["year", "month"])["incident_id"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=range(1, 13), fill_value=0)
    )


def attack_sector_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compare DDoS, Ransomware, and Malware presence across target subcategories."""
    cols = ["Target subcategory", "DDoS-related", "Ransomware-related", "Malware-related"]
    if df.empty:
        return _empty(cols)
    d = (
        df[["incident_id", "sector_list", "is_ddos", "is_ransomware", "is_malware_related"]]
        .explode("sector_list")
        .dropna(subset=["sector_list"])
    )
    d["sector_list"] = d.sector_list.astype(str).str.strip()
    d = d[d.sector_list.ne("")].drop_duplicates(["incident_id", "sector_list"])
    if d.empty:
        return _empty(cols)
    return (
        d.groupby("sector_list")
        .agg(
            **{
                "DDoS-related": ("is_ddos", "sum"),
                "Ransomware-related": ("is_ransomware", "sum"),
                "Malware-related": ("is_malware_related", "sum"),
            }
        )
        .reset_index()
        .rename(columns={"sector_list": "Target subcategory"})
        .sort_values("Malware-related", ascending=False)
    )


def country_sector_relationship(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulate target countries and targeted sectors."""
    cols = ["Country", "Target subcategory", "Incidents"]
    if df.empty:
        return _empty(cols)
    d = (
        df[["incident_id", "country_list", "sector_list"]]
        .explode("country_list")
        .explode("sector_list")
        .dropna(subset=["country_list", "sector_list"])
    )
    d = d.drop_duplicates(["incident_id", "country_list", "sector_list"])
    if d.empty:
        return _empty(cols)
    return (
        d.groupby(["country_list", "sector_list"])
        .agg(Incidents=("incident_id", "nunique"))
        .reset_index()
        .rename(columns={"country_list": "Country", "sector_list": "Target subcategory"})
        .sort_values("Incidents", ascending=False)
    )


def risk_distribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute distribution of incidents by Analytical Risk Level."""
    if df.empty or "risk_level" not in df.columns:
        return pd.DataFrame(columns=["Risk Level", "Incidents", "Share %"])
    order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    counts = df["risk_level"].value_counts().reindex(order, fill_value=0)
    total = len(df)
    return pd.DataFrame({
        "Risk Level": counts.index,
        "Incidents": counts.values,
        "Share %": [round((v / max(1, total)) * 100, 1) for v in counts.values]
    })


def generate_automated_intelligence_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Module 15: Synthesizes a factual, data-grounded intelligence brief
    strictly based on active dataset statistics without speculation or hallucination.
    """
    if df.empty:
        return {
            "threat_posture": "NO DATA",
            "posture_color": "#8294ac",
            "observations": ["No incidents match the active filter criteria."]
        }

    total = len(df)
    crit = int((df["risk_level"] == "CRITICAL").sum()) if "risk_level" in df.columns else 0
    high = int((df["risk_level"] == "HIGH").sum()) if "risk_level" in df.columns else 0
    avg_score = round(float(df["risk_score"].mean()), 1) if "risk_score" in df.columns else 0.0

    # Determine Threat Posture
    if crit >= 50 or (crit + high) / max(1, total) >= 0.5:
        posture = "ELEVATED CRITICAL THREAT ENVIRONMENT"
        posture_color = "#ff4d6d"
    elif (crit + high) / max(1, total) >= 0.3:
        posture = "HEIGHTENED RISK / ACTIVE ATTACK CONCENTRATION"
        posture_color = "#ff8552"
    else:
        posture = "MODERATE / BASELINE EXPLOITATION POSTURE"
        posture_color = "#f3b76b"

    observations = []

    # Volume & period observation
    years = df["year"].dropna()
    if not years.empty:
        ymin, ymax = int(years.min()), int(years.max())
        observations.append(
            f"Active view encompasses {total:,} publicly documented cyber incidents across the period {ymin}–{ymax}."
        )

    # Risk concentration observation
    observations.append(
        f"Analytical risk assessment identifies {crit:,} CRITICAL incidents ({crit/total*100:.1f}%) and "
        f"{high:,} HIGH-risk incidents ({high/total*100:.1f}%), yielding an average analytical risk score of {avg_score}/100."
    )

    # Top target sectors observation
    sec = sector_analysis(df).head(3)
    if not sec.empty:
        top_secs = ", ".join(f"{r['Target subcategory']} ({r['Incidents']:,})" for _, r in sec.iterrows())
        observations.append(
            f"Target sector profile is heavily concentrated in critical infrastructure: {top_secs}."
        )

    # Primary vector observation
    ransom_cnt = int(df["is_ransomware"].sum()) if "is_ransomware" in df.columns else 0
    ddos_cnt = int(df["is_ddos"].sum()) if "is_ddos" in df.columns else 0
    disrupt_cnt = int(df["has_disruption"].sum()) if "has_disruption" in df.columns else 0

    observations.append(
        f"Recorded vector mix reveals {ransom_cnt:,} ransomware-related cases ({ransom_cnt/total*100:.1f}%) and "
        f"{ddos_cnt:,} DDoS events ({ddos_cnt/total*100:.1f}%). A total of {disrupt_cnt:,} incidents ({disrupt_cnt/total*100:.1f}%) "
        f"inflicted operational or functional disruption on victim networks."
    )

    # Initial Access observation if present
    if "mitre_initial_access" in df.columns:
        phish_cnt = df["mitre_initial_access"].str.lower().str.contains("phishing").sum()
        if phish_cnt > 0:
            observations.append(
                f"Phishing is specifically documented as the primary Initial Access mechanism (MITRE TA0001) in "
                f"{phish_cnt:,} incidents in this dataset, demonstrating the critical importance of URL and email validation."
            )

    return {
        "threat_posture": posture,
        "posture_color": posture_color,
        "observations": observations
    }
