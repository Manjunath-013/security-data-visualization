"""Attacker and Initiator Threat Intelligence Module.
Extracts and analyzes threat actor attribution, state responsibility,
initiator categories, and attacker-to-target relationship graphs.
"""

from typing import Any, Dict, List
import pandas as pd
import plotly.graph_objects as go


def get_initiator_country_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze top initiator/origin countries attributed in incidents."""
    if df.empty or "initiator_country" not in df.columns:
        return pd.DataFrame(columns=["Initiator Country", "Incidents"])

    d = df[df["initiator_country"].str.lower().ne("not attributed") & df["initiator_country"].ne("")].copy()
    if d.empty:
        return pd.DataFrame(columns=["Initiator Country", "Incidents"])

    res = (
        d.groupby("initiator_country")["incident_id"]
        .nunique()
        .reset_index()
        .rename(columns={"initiator_country": "Initiator Country", "incident_id": "Incidents"})
        .sort_values("Incidents", ascending=False)
    )
    return res


def get_initiator_category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze breakdown of threat actors by category (State, Non-state, etc.)."""
    if df.empty or "initiator_category" not in df.columns:
        return pd.DataFrame(columns=["Category", "Incidents"])

    d = df[df["initiator_category"].ne("") & ~df["initiator_category"].str.lower().isin({"not available", "unknown"})].copy()
    if d.empty:
        return pd.DataFrame(columns=["Category", "Incidents"])

    res = (
        d.groupby("initiator_category")["incident_id"]
        .nunique()
        .reset_index()
        .rename(columns={"initiator_category": "Category", "incident_id": "Incidents"})
        .sort_values("Incidents", ascending=False)
    )
    return res


def get_attribution_sources(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze top cybersecurity companies and state bodies providing attributions."""
    if df.empty:
        return pd.DataFrame(columns=["Attributing Entity", "Attributions"])

    # Combine attributing_company and attributing_actor
    records = []
    for _, row in df.iterrows():
        comp = str(row.get("attributing_company", "")).strip()
        actor = str(row.get("attributing_actor", "")).strip()
        iid = row["incident_id"]

        ignored = {"", "not available", "unknown", "none", "nan"}
        if comp and comp.lower() not in ignored:
            records.append({"incident_id": iid, "Entity": comp, "Type": "Private Cyber Intelligence / Company"})
        elif actor and actor.lower() not in ignored:
            records.append({"incident_id": iid, "Entity": actor, "Type": "Government / Judicial / Public Authority"})

    if not records:
        return pd.DataFrame(columns=["Attributing Entity", "Type", "Attributions"])

    adf = pd.DataFrame(records).drop_duplicates(["incident_id", "Entity"])
    summary = (
        adf.groupby(["Entity", "Type"])["incident_id"]
        .nunique()
        .reset_index()
        .rename(columns={"Entity": "Attributing Entity", "incident_id": "Attributions"})
        .sort_values("Attributions", ascending=False)
    )
    return summary


def get_attacker_target_pairs(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Extract top Initiator Country -> Target Country interaction flows."""
    if df.empty:
        return pd.DataFrame(columns=["Initiator Country", "Target Country", "Incidents"])

    clean = df[
        df["initiator_country"].str.lower().ne("not attributed")
        & df["initiator_country"].ne("")
        & df["receiver_country"].str.lower().ne("not available")
        & df["receiver_country"].ne("")
    ].copy()

    if clean.empty:
        return pd.DataFrame(columns=["Initiator Country", "Target Country", "Incidents"])

    # Explode country_list for target
    exp = clean[["incident_id", "initiator_country", "country_list"]].explode("country_list").dropna()
    exp = exp.drop_duplicates(["incident_id", "initiator_country", "country_list"])

    pairs = (
        exp.groupby(["initiator_country", "country_list"])["incident_id"]
        .nunique()
        .reset_index()
        .rename(columns={
            "initiator_country": "Initiator Country",
            "country_list": "Target Country",
            "incident_id": "Incidents"
        })
        .sort_values("Incidents", ascending=False)
    )
    return pairs.head(top_n)


def build_threat_sankey_diagram(df: pd.DataFrame, max_flows: int = 40) -> go.Figure:
    """
    Build an interactive multi-stage Sankey diagram:
    Initiator Country -> Attack Type -> Target Country -> Target Sector.
    """
    clean = df[
        df["initiator_country"].str.lower().ne("not attributed")
        & df["initiator_country"].ne("")
    ].copy()

    if clean.empty or len(clean) < 5:
        # Fallback if filtered dataset has few attributed incidents
        clean = df.copy()
        clean["initiator_country"] = clean["initiator_country"].replace({"": "Unattributed", "Not attributed": "Unattributed"})

    flows = []
    for _, row in clean.iterrows():
        init_c = row.get("initiator_country", "Unattributed")
        if not init_c or init_c.lower() in {"not available", ""}:
            init_c = "Unattributed"

        # Determine primary attack vector
        if row.get("is_ransomware", False):
            vector = "Ransomware"
        elif row.get("is_ddos", False):
            vector = "DDoS"
        elif row.get("is_malware_related", False):
            vector = "Malware"
        else:
            vector = "Other Exploit"

        # Targets
        rec_countries = row.get("country_list", ["Undisclosed"])
        target_c = rec_countries[0] if rec_countries else "Undisclosed"

        sectors = row.get("sector_list", ["Unspecified"])
        sector = sectors[0] if sectors else "Unspecified"

        flows.append({
            "initiator": f"Origin: {init_c}",
            "vector": f"Vector: {vector}",
            "target": f"Target: {target_c}",
            "sector": f"Sector: {sector}"
        })

    flow_df = pd.DataFrame(flows)
    if flow_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Insufficient attributed data for flow diagram")
        return fig

    # Top links
    l1 = flow_df.groupby(["initiator", "vector"]).size().reset_index(name="value").nlargest(15, "value")
    l2 = flow_df.groupby(["vector", "target"]).size().reset_index(name="value").nlargest(15, "value")
    l3 = flow_df.groupby(["target", "sector"]).size().reset_index(name="value").nlargest(15, "value")

    all_nodes = sorted(list(set(
        l1["initiator"].tolist() + l1["vector"].tolist() +
        l2["vector"].tolist() + l2["target"].tolist() +
        l3["target"].tolist() + l3["sector"].tolist()
    )))
    node_map = {n: i for i, n in enumerate(all_nodes)}

    sources = []
    targets = []
    values = []

    for _, r in l1.iterrows():
        sources.append(node_map[r["initiator"]])
        targets.append(node_map[r["vector"]])
        values.append(int(r["value"]))

    for _, r in l2.iterrows():
        sources.append(node_map[r["vector"]])
        targets.append(node_map[r["target"]])
        values.append(int(r["value"]))

    for _, r in l3.iterrows():
        sources.append(node_map[r["target"]])
        targets.append(node_map[r["sector"]])
        values.append(int(r["value"]))

    # Node styling
    node_colors = []
    for n in all_nodes:
        if n.startswith("Origin:"):
            node_colors.append("#ff8552")
        elif n.startswith("Vector:"):
            node_colors.append("#48d6e8")
        elif n.startswith("Target:"):
            node_colors.append("#8b9eff")
        else:
            node_colors.append("#43d6a2")

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=18,
            thickness=18,
            line=dict(color="rgba(148,163,184,0.3)", width=0.5),
            label=all_nodes,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(72,214,232,0.22)"
        )
    )])

    fig.update_layout(
        font=dict(family="DM Sans", size=11, color="#edf4ff"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=25, b=10)
    )
    return fig
