
from pathlib import Path
import re

import pandas as pd


# ==================================================
# 1. REQUIRED DATASET COLUMNS
# ==================================================

REQUIRED_COLUMNS = [
    "incident_id",
    "name",
    "description",
    "start_date",
    "incident_type",
    "receiver_country",
    "receiver_subcategory",
    "mitre_impact",
    "impact_indicator_label",
    "has_disruption"
]


# ==================================================
# 2. AVAILABLE TEXT COLUMNS
# ==================================================

# These are actual columns in the EuRepoC dataset.
TEXT_COLUMNS = [
    "name",
    "description",
    "incident_type",
    "receiver_country",
    "receiver_regions",
    "receiver_category",
    "receiver_subcategory",
    "initiator_country",
    "mitre_impact",
    "mitre_initial_access",
    "impact_indicator_label",
    "disruption",
    "functional_impact",
    "data_theft",
    "hijacking",
    "zero_days",
    "source_url"
]


# ==================================================
# 3. NORMALIZE MULTI-VALUE FIELDS
# ==================================================

def normalize_multivalue(value):

    if pd.isna(value):
        return []

    ignored = {
        "",
        "not available",
        "unknown",
        "n/a",
        "nan",
        "none"
    }

    result = []
    seen = set()

    for part in re.split(r"[;|]", str(value)):

        item = part.strip()
        key = item.casefold()

        if key not in ignored and key not in seen:
            result.append(item)
            seen.add(key)

    return result


# ==================================================
# 4. COMBINE TEXT FROM AVAILABLE COLUMNS
# ==================================================

def _combined_text(df, columns):

    cols = [
        col for col in columns
        if col in df.columns
    ]

    if not cols:
        return pd.Series("", index=df.index, dtype="string")

    return (
        df[cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )


# ==================================================
# 5. CONVERT BOOLEAN VALUES
# ==================================================

def _to_bool(value):

    if pd.isna(value):
        return False

    return str(value).strip().casefold() in {
        "true",
        "1",
        "yes",
        "y"
    }


# ==================================================
# 6. LOAD AND CLEAN DATASET
# ==================================================

def load_and_clean_data(path="data/incidents.csv"):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Place the CSV at data/incidents.csv"
        )

    # Load dataset
    df = pd.read_csv(path, low_memory=False)

    print("Dataset loaded successfully!")
    print("Original dataset shape:", df.shape)

    # Validate required columns
    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )

    # Remove duplicate incidents
    df = df.drop_duplicates(
        subset=["incident_id"]
    ).copy()

    # Normalize available text columns
    for col in TEXT_COLUMNS:

        if col in df.columns:

            df[col] = (
                df[col]
                .fillna("")
                .astype(str)
                .str.strip()
            )

    # ==================================================
    # 7. DATE PREPROCESSING
    # ==================================================

    df["start_date"] = pd.to_datetime(
        df["start_date"],
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )

    if "end_date" in df.columns:

        df["end_date"] = pd.to_datetime(
            df["end_date"],
            format="mixed",
            dayfirst=True,
            errors="coerce"
        )

    # Extract temporal features
    df["year"] = df["start_date"].dt.year.astype("Int64")

    df["month"] = df["start_date"].dt.month.astype("Int64")

    df["year_month"] = (
        df["start_date"]
        .dt.to_period("M")
        .astype("string")
    )

    # ==================================================
    # 8. NORMALIZE MULTI-VALUE FIELDS
    # ==================================================

    df["country_list"] = (
        df["receiver_country"]
        .apply(normalize_multivalue)
    )

    df["sector_list"] = (
        df["receiver_subcategory"]
        .apply(normalize_multivalue)
    )

    df["mitre_impact_list"] = (
        df["mitre_impact"]
        .apply(normalize_multivalue)
    )

    if "mitre_initial_access" in df.columns:

        df["mitre_initial_access_list"] = (
            df["mitre_initial_access"]
            .apply(normalize_multivalue)
        )

    # ==================================================
    # 9. DATASET-LABEL-BASED INDICATORS
    # ==================================================

    source_text = _combined_text(
        df,
        ["incident_type"]
    )

    source_ddos = source_text.str.contains(
        r"\bDDoS\b|distributed denial|denial.of.service",
        case=False,
        regex=True
    )

    source_ransom = source_text.str.contains(
        r"\bransomware\b|data encrypted for impact",
        case=False,
        regex=True
    )

    source_malware = source_text.str.contains(
        r"\bmalware\b|\bransomware\b|\btrojan\b|"
        r"\bworm\b|\bspyware\b|\brootkit\b|"
        r"\bbotnet\b|\bvirus\b",
        case=False,
        regex=True
    )


    # ==================================================
    # 10. KEYWORD-BASED INDICATORS
    # ==================================================

    ddos_text = _combined_text(
        df,
        ["name", "description", "mitre_impact",
        "mitre_initial_access"]
    )

    ransom_text = _combined_text(
        df,
        ["name", "description", "mitre_impact"]
    )

    malware_text = _combined_text(
        df,
        ["name", "description", "mitre_impact"]
    )

    df["ddos_keyword_flag"] = ddos_text.str.contains(
        r"\bDDoS\b|distributed denial of service|"
        r"network denial of service|"
        r"endpoint denial of service",
        case=False,
        regex=True
    )

    df["ransomware_keyword_flag"] = ransom_text.str.contains(
        r"\bransomware\b|data encrypted for impact",
        case=False,
        regex=True
    )

    df["malware_keyword_flag"] = malware_text.str.contains(
        r"\bmalware\b|\bransomware\b|\btrojan\b|"
        r"\bworm\b|\bspyware\b|\brootkit\b|"
        r"\bbotnet\b|\bvirus\b",
        case=False,
        regex=True
    )


    # ==================================================
    # 11. FINAL EXPLORATORY INDICATORS
    # ==================================================

    df["is_ddos"] = (
        source_ddos | df["ddos_keyword_flag"]
    )

    df["is_ransomware"] = (
        source_ransom | df["ransomware_keyword_flag"]
    )

    df["is_malware_related"] = (
        source_malware | df["malware_keyword_flag"]
    )


    # ==================================================
    # 12. CLASSIFICATION BASIS
    # ==================================================

    source_match = (
        source_ddos | source_ransom | source_malware
    )

    keyword_match = (
        df["ddos_keyword_flag"] |
        df["ransomware_keyword_flag"] |
        df["malware_keyword_flag"]
    )

    df["classification_basis"] = "No matching indicator"

    df.loc[
        keyword_match & ~source_match,
        "classification_basis"
    ] = "Keyword-derived exploratory indicator"

    df.loc[
        source_match & ~keyword_match,
        "classification_basis"
    ] = "Dataset incident_type label"

    df.loc[
        source_match & keyword_match,
        "classification_basis"
    ] = "Dataset label and keyword indicator"
    # ==================================================
    # 13. DISRUPTION FLAG
    # ==================================================

    df["has_disruption"] = (
        df["has_disruption"]
        .apply(_to_bool)
    )

    # ==================================================
    # 14. NUMERIC COLUMN CONVERSION
    # ==================================================

    numeric_columns = [
        "weighted_intensity",
        "impact_indicator_score",
        "economic_impact_value",
        "affected_entities_value",
        "affected_third_countries_value"
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # ==================================================
    # 15. FINAL OUTPUT
    # ==================================================

    print("\nPreprocessing completed successfully!")

    print("Cleaned dataset shape:", df.shape)

    print("\nExploratory indicator counts:")

    print("DDoS:", df["is_ddos"].sum())

    print("Ransomware:", df["is_ransomware"].sum())

    print("Malware-related:", df["is_malware_related"].sum())

    print("\nClassification basis:")
    print(df["classification_basis"].value_counts())

    return df
