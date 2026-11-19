
import pandas as pd


# ==================================================
# 1. ATTACK SUMMARY
# ==================================================

def attack_summary(df):
    """Count incidents matching each exploratory indicator."""

    return pd.DataFrame({
        "Attack Category": [
            "DDoS-related",
            "Ransomware-related",
            "Malware-related"
        ],
        "Incident Count": [
            int(df["is_ddos"].sum()),
            int(df["is_ransomware"].sum()),
            int(df["is_malware_related"].sum())
        ]
    })


# ==================================================
# 2. YEARLY TRENDS
# ==================================================

def yearly_trends(df):

    data = df.dropna(subset=["year"]).copy()

    result = (
        data.groupby("year")
        .agg(
            Total_Incidents=("incident_id", "nunique"),
            DDoS=("is_ddos", "sum"),
            Ransomware=("is_ransomware", "sum"),
            Malware=("is_malware_related", "sum")
        )
        .reset_index()
        .sort_values("year")
    )

    return result


# ==================================================
# 3. MONTHLY TRENDS
# ==================================================

def monthly_trends(df):

    data = df.dropna(subset=["year_month"]).copy()

    result = (
        data.groupby("year_month")
        .agg(
            Total_Incidents=("incident_id", "nunique"),
            DDoS=("is_ddos", "sum"),
            Ransomware=("is_ransomware", "sum"),
            Malware=("is_malware_related", "sum")
        )
        .reset_index()
        .sort_values("year_month")
    )

    return result


# ==================================================
# 4. COUNTRY ANALYSIS
# ==================================================

def country_analysis(df):

    data = df.explode("country_list")

    data = data[
        data["country_list"].notna() &
        data["country_list"].astype(str).str.strip().ne("")
    ]

    result = (
        data.groupby("country_list")
        .agg(
            Total_Incidents=("incident_id", "nunique"),
            DDoS=("is_ddos", "sum"),
            Ransomware=("is_ransomware", "sum"),
            Malware=("is_malware_related", "sum")
        )
        .reset_index()
        .rename(columns={"country_list": "Country"})
        .sort_values("Total_Incidents", ascending=False)
    )

    return result


# ==================================================
# 5. SECTOR ANALYSIS
# ==================================================

def sector_analysis(df):

    data = df.explode("sector_list")

    data = data[
        data["sector_list"].notna() &
        data["sector_list"].astype(str).str.strip().ne("")
    ]

    result = (
        data.groupby("sector_list")
        .agg(
            Total_Incidents=("incident_id", "nunique"),
            DDoS=("is_ddos", "sum"),
            Ransomware=("is_ransomware", "sum"),
            Malware=("is_malware_related", "sum")
        )
        .reset_index()
        .rename(columns={"sector_list": "Sector"})
        .sort_values("Total_Incidents", ascending=False)
    )

    return result


# ==================================================
# 6. DDoS AND RANSOMWARE COMPARISON
# ==================================================

def category_comparison(df):

    ddos = df["is_ddos"].fillna(False).astype(bool)
    ransomware = (
        df["is_ransomware"].fillna(False).astype(bool)
    )

    categories = [
        "DDoS only",
        "Ransomware only",
        "Both",
        "Neither"
    ]

    counts = [
        int((ddos & ~ransomware).sum()),
        int((~ddos & ransomware).sum()),
        int((ddos & ransomware).sum()),
        int((~ddos & ~ransomware).sum())
    ]

    return pd.DataFrame({
        "Category": categories,
        "Incident Count": counts
    })


# ==================================================
# 7. ATTACK OVERLAP ANALYSIS
# ==================================================

def attack_overlap(df):

    ddos = df["is_ddos"].fillna(False).astype(bool)
    ransomware = (
        df["is_ransomware"].fillna(False).astype(bool)
    )
    malware = (
        df["is_malware_related"].fillna(False).astype(bool)
    )

    categories = {
        "DDoS only": ddos & ~ransomware & ~malware,
        "Ransomware only": ~ddos & ransomware & ~malware,
        "Malware only": ~ddos & ~ransomware & malware,
        "DDoS + Ransomware": ddos & ransomware & ~malware,
        "DDoS + Malware": ddos & ~ransomware & malware,
        "Ransomware + Malware": ~ddos & ransomware & malware,
        "All three": ddos & ransomware & malware,
        "No matching indicator": ~ddos & ~ransomware & ~malware
    }

    return pd.DataFrame({
        "Category": list(categories.keys()),
        "Incident Count": [
            int(mask.sum()) for mask in categories.values()
        ]
    })


# ==================================================
# 8. IMPACT ANALYSIS
# ==================================================

def impact_analysis(df):

    columns = [
        col for col in [
            "weighted_intensity",
            "has_disruption",
            "economic_impact_value"
        ]
        if col in df.columns
    ]

    result = {}

    if "weighted_intensity" in columns:

        result["average_intensity"] = (
            pd.to_numeric(
                df["weighted_intensity"],
                errors="coerce"
            ).mean()
        )

    if "has_disruption" in columns:

        result["disruption_count"] = int(
            df["has_disruption"].fillna(False).sum()
        )

    if "economic_impact_value" in columns:

        economic = pd.to_numeric(
            df["economic_impact_value"],
            errors="coerce"
        )

        result["economic_records"] = int(economic.notna().sum())

    return result


# ==================================================
# 9. INCIDENT TYPE DISTRIBUTION
# ==================================================

def incident_type_analysis(df):

    data = df.copy()

    data["incident_type_list"] = (
        data["incident_type"]
        .fillna("")
        .astype(str)
        .str.split(";")
    )

    data = data.explode("incident_type_list")

    data["incident_type_list"] = (
        data["incident_type_list"].str.strip()
    )

    data = data[
        data["incident_type_list"].ne("")
    ]

    result = (
        data.groupby("incident_type_list")
        ["incident_id"]
        .nunique()
        .reset_index(name="Incident Count")
        .rename(columns={
            "incident_type_list": "Incident Type"
        })
        .sort_values("Incident Count", ascending=False)
    )

    return result


# ==================================================
# 10. INCIDENT EXPLORER
# ==================================================

def filter_incidents(
    df,
    search_text="",
    attack_type="All"
):

    result = df.copy()

    if search_text:

        search_text = search_text.strip()

        searchable = (
            result["name"].fillna("").astype(str) + " " +
            result["description"].fillna("").astype(str)
        )

        result = result[
            searchable.str.contains(
                search_text,
                case=False,
                regex=False
            )
        ]

    if attack_type == "DDoS":
        result = result[result["is_ddos"]]

    elif attack_type == "Ransomware":
        result = result[result["is_ransomware"]]

    elif attack_type == "Malware-related":
        result = result[result["is_malware_related"]]

    return result
