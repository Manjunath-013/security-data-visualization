"""Dataset Preprocessing and Feature Normalization Module.
Loads, cleans, and standardizes EuRepoC cyber incident records, normalizes
multi-value fields, computes threat indicators, and invokes the analytical risk engine.
"""

from pathlib import Path
import re
from typing import List
import pandas as pd

from src.risk_engine import compute_risk_scores

REQUIRED_COLUMNS = [
    "incident_id", "name", "description", "start_date", "incident_type",
    "receiver_country", "receiver_subcategory", "mitre_impact",
    "impact_indicator_label", "has_disruption"
]

TEXT_COLUMNS = [
    "name", "description", "incident_type", "operation_type",
    "receiver_name", "receiver_country", "receiver_subcategory", "receiver_category",
    "initiator_name", "initiator_country", "initiator_category", "initiator_subcategory",
    "attributing_actor", "attributing_company", "attributing_country",
    "mitre_impact", "mitre_initial_access", "impact_indicator_label",
    "data_theft", "disruption", "hijacking", "physical_effects_spatial",
    "zero_days", "source", "source_url"
]


def normalize_multivalue(value) -> List[str]:
    """Parse delimited values (semicolons or pipes) into clean list of strings."""
    if pd.isna(value):
        return []
    ignored = {"", "not available", "unknown", "n/a", "nan", "none", "not attributed"}
    result, seen = [], set()
    for part in re.split(r"[;|]", str(value)):
        item = part.strip()
        key = item.casefold()
        if key not in ignored and key not in seen:
            result.append(item)
            seen.add(key)
    return result


def _combined_text(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    """Safely concatenate text columns for regex indicator matching."""
    cols = [c for c in columns if c in df.columns]
    return df[cols].fillna("").astype(str).agg(" ".join, axis=1) if cols else pd.Series("", index=df.index)


def _to_bool(value) -> bool:
    """Convert mixed string/boolean representations to clean bool."""
    if pd.isna(value):
        return False
    return str(value).strip().casefold() in {"true", "1", "yes", "y"}


def load_and_clean_data(path: str = "data/incidents.csv") -> pd.DataFrame:
    """
    Load the EuRepoC dataset, perform validation, date parsing, field normalization,
    and compute analytical risk scores.
    """
    data_file = Path(path)
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_file}. Please place incidents.csv in data/")

    df = pd.read_csv(data_file, low_memory=False)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Deduplicate on unique incident identifier
    df = df.drop_duplicates(subset=["incident_id"]).copy()

    # Clean text columns
    for col in TEXT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Parse timestamps
    try:
        df["start_date"] = pd.to_datetime(df["start_date"], format="mixed", dayfirst=True, errors="coerce")
    except (TypeError, ValueError):
        df["start_date"] = pd.to_datetime(df["start_date"], dayfirst=True, errors="coerce")

    df["year"] = df["start_date"].dt.year.astype("Int64")
    df["month"] = df["start_date"].dt.month.astype("Int64")
    df["year_month"] = df["start_date"].dt.to_period("M").astype("string")

    # Parse multi-value lists
    df["country_list"] = df["receiver_country"].apply(normalize_multivalue)
    df["sector_list"] = df["receiver_subcategory"].apply(normalize_multivalue)
    df["mitre_impact_list"] = df["mitre_impact"].apply(normalize_multivalue)
    df["mitre_initial_access_list"] = df["mitre_initial_access"].apply(normalize_multivalue)
    df["initiator_country_list"] = df["initiator_country"].apply(normalize_multivalue)

    # Boolean flags and indicators
    df["has_disruption"] = df["has_disruption"].apply(_to_bool)
    df["has_data_theft"] = df["data_theft"].apply(
        lambda x: str(x).lower() not in {"", "not available", "none", "nan", "unknown"}
    )
    df["has_hijacking"] = df["hijacking"].apply(
        lambda x: "hijacking" in str(x).lower() and str(x).lower() not in {"not available", "none", "nan"}
    )
    df["has_zero_day"] = df["zero_days"].apply(
        lambda x: "yes" in str(x).lower() or str(x).strip().isdigit() and int(str(x).strip()) > 0
    )

    # Threat categorization (Exploratory indicators)
    source_text = _combined_text(df, ["operation_type", "incident_type"])
    source_ddos = source_text.str.contains(r"\bDDoS\b|distributed denial|denial.of.service", case=False, regex=True)
    source_ransom = source_text.str.contains(r"\bransomware\b|data encrypted for impact", case=False, regex=True)
    source_malware = source_text.str.contains(
        r"\bmalware\b|\bransomware\b|\btrojan\b|\bworm\b|\bspyware\b|\brootkit\b|\bbotnet\b|\bvirus\b",
        case=False, regex=True
    )

    ddos_text = _combined_text(df, ["name", "description", "mitre_impact", "mitre_initial_access"])
    ransom_text = _combined_text(df, ["name", "description", "incident_type", "mitre_impact"])
    malware_text = _combined_text(df, ["name", "description", "incident_type", "mitre_impact"])

    df["ddos_keyword_flag"] = ddos_text.str.contains(
        r"\bDDoS\b|distributed denial of service|network denial of service|endpoint denial of service",
        case=False, regex=True
    )
    df["ransomware_keyword_flag"] = ransom_text.str.contains(
        r"\bransomware\b|data encrypted for impact", case=False, regex=True
    )
    df["malware_keyword_flag"] = malware_text.str.contains(
        r"\bmalware\b|\bransomware\b|\btrojan\b|\bworm\b|\bspyware\b|\brootkit\b|\bbotnet\b|\bvirus\b",
        case=False, regex=True
    )

    df["is_ddos"] = source_ddos | df["ddos_keyword_flag"]
    df["is_ransomware"] = source_ransom | df["ransomware_keyword_flag"]
    df["is_malware_related"] = source_malware | df["malware_keyword_flag"]

    source_match = source_ddos | source_ransom | source_malware
    df["classification_basis"] = "keyword-derived exploratory indicator"
    df.loc[source_match, "classification_basis"] = "dataset operation_type/incident_type fields (also may include text indicators)"

    for col in ["weighted_intensity", "impact_indicator_score", "number_attributions"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute Analytical Risk Scores
    df = compute_risk_scores(df)

    return df
