
# src/feature_engineering.py

import re
import pandas as pd


# Only fields available as incident information.
# MITRE target columns and outcome/impact indicators are excluded.
INPUT_COLUMNS = [
    "name",
    "description",
    "incident_type",
    "receiver_category",
    "receiver_subcategory",
    "receiver_country",
    "receiver_regions",
]

TARGET_COLUMNS = [
    "mitre_initial_access",
    "mitre_impact",
]

MIN_CLASS_COUNT = 10


def clean_text(value):
    """Convert missing values to empty text and normalize whitespace."""

    if pd.isna(value):
        return ""

    value = str(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def build_model_text(df):
    """
    Combine selected incident fields into one text feature.

    The output is a pandas Series that can be passed to
    TfidfVectorizer.
    """

    missing = [col for col in INPUT_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required model input columns: {missing}"
        )

    combined = pd.Series("", index=df.index, dtype="object")

    for col in INPUT_COLUMNS:
        values = df[col].apply(clean_text)

        # Add field names to preserve context.
        combined = combined + " " + col + ": " + values

    return combined.str.strip()


def prepare_target_data(df, target_column):
    """
    Remove unavailable targets and classes with fewer than
    MIN_CLASS_COUNT examples.

    Returns:
        X: combined incident text
        y: target labels
        excluded_classes: dictionary of excluded class counts
        original_counts: original target distribution
    """

    if target_column not in TARGET_COLUMNS:
        raise ValueError(
            f"Unsupported target column: {target_column}"
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    data = df.copy()

    data[target_column] = (
        data[target_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    original_counts = data[target_column].value_counts().to_dict()

    # Exclude missing and unavailable labels.
    valid = data[
        (data[target_column] != "") &
        (data[target_column].str.lower() != "not available")
    ].copy()

    counts = valid[target_column].value_counts()

    # Keep only classes with sufficient examples.
    supported_classes = counts[
        counts >= MIN_CLASS_COUNT
    ].index

    excluded_classes = counts[
        counts < MIN_CLASS_COUNT
    ].to_dict()

    valid = valid[
        valid[target_column].isin(supported_classes)
    ].copy()

    if valid.empty:
        raise ValueError(
            f"No training records available for {target_column}."
        )

    X = build_model_text(valid)
    y = valid[target_column]

    if y.nunique() < 2:
        raise ValueError(
            f"At least two supported classes are required "
            f"for {target_column}."
        )

    return X, y, excluded_classes, original_counts
