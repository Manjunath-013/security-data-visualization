"""MITRE ATT&CK Lifecycle and Tactic Progression Module.
Analyzes documented Initial Access tactics, Execution/Compromise mechanisms,
and final Impact techniques directly grounded in the EuRepoC incident records.
"""

from typing import Any, Dict, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def get_initial_access_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze documented Initial Access vectors (MITRE ATT&CK TA0001)."""
    if df.empty or "mitre_initial_access_list" not in df.columns:
        return pd.DataFrame(columns=["Initial Access Technique", "Incidents"])

    exp = df[["incident_id", "mitre_initial_access_list"]].explode("mitre_initial_access_list").dropna()
    exp["mitre_initial_access_list"] = exp["mitre_initial_access_list"].astype(str).str.strip()
    clean = exp[
        exp["mitre_initial_access_list"].ne("")
        & ~exp["mitre_initial_access_list"].str.lower().isin({"not available", "unknown", "none"})
    ].drop_duplicates(["incident_id", "mitre_initial_access_list"])

    if clean.empty:
        return pd.DataFrame(columns=["Initial Access Technique", "Incidents"])

    res = (
        clean.groupby("mitre_initial_access_list")["incident_id"]
        .nunique()
        .reset_index()
        .rename(columns={"mitre_initial_access_list": "Initial Access Technique", "incident_id": "Incidents"})
        .sort_values("Incidents", ascending=False)
    )
    return res


def get_mitre_impact_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze documented Impact techniques (MITRE ATT&CK TA0040)."""
    if df.empty or "mitre_impact_list" not in df.columns:
        return pd.DataFrame(columns=["Impact Technique", "Incidents"])

    exp = df[["incident_id", "mitre_impact_list"]].explode("mitre_impact_list").dropna()
    exp["mitre_impact_list"] = exp["mitre_impact_list"].astype(str).str.strip()
    clean = exp[
        exp["mitre_impact_list"].ne("")
        & ~exp["mitre_impact_list"].str.lower().isin({"not available", "unknown", "none"})
    ].drop_duplicates(["incident_id", "mitre_impact_list"])

    if clean.empty:
        return pd.DataFrame(columns=["Impact Technique", "Incidents"])

    res = (
        clean.groupby("mitre_impact_list")["incident_id"]
        .nunique()
        .reset_index()
        .rename(columns={"mitre_impact_list": "Impact Technique", "incident_id": "Incidents"})
        .sort_values("Incidents", ascending=False)
    )
    return res


def get_attack_lifecycle_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-tabulate documented Initial Access against resulting Impact Technique
    where both phases are recorded in the incident.
    """
    clean = df[
        df["mitre_initial_access"].ne("")
        & ~df["mitre_initial_access"].str.lower().isin({"not available", "unknown"})
        & df["mitre_impact"].ne("")
        & ~df["mitre_impact"].str.lower().isin({"not available", "unknown"})
    ].copy()

    if clean.empty:
        return pd.DataFrame()

    exp = (
        clean[["incident_id", "mitre_initial_access_list", "mitre_impact_list"]]
        .explode("mitre_initial_access_list")
        .explode("mitre_impact_list")
        .dropna()
        .drop_duplicates(["incident_id", "mitre_initial_access_list", "mitre_impact_list"])
    )

    pivot = (
        exp.groupby(["mitre_initial_access_list", "mitre_impact_list"])["incident_id"]
        .nunique()
        .unstack(fill_value=0)
    )
    return pivot
