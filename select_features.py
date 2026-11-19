
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==================================================
# 1. PATH CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "incidents.csv"

OUTPUT_DIR = BASE_DIR / "outputs"

# Create outputs folder automatically
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD DATASET
# ==================================================

df = pd.read_csv(DATA_PATH, low_memory=False)

print("Dataset loaded successfully!")
print("Original dataset shape:", df.shape)


# ==================================================
# 3. SELECT IMPORTANT FEATURES
# ==================================================

selected_features = [
    "incident_id",
    "name",
    "description",
    "start_date",
    "end_date",
    "incident_type",
    "receiver_country",
    "receiver_regions",
    "receiver_category",
    "receiver_subcategory",
    "mitre_initial_access",
    "mitre_impact",
    "user_interaction",
    "has_disruption",
    "data_theft",
    "disruption",
    "hijacking",
    "weighted_intensity",
    "functional_impact",
    "economic_impact",
    "economic_impact_value",
    "source_url"
]

# Keep only columns available in the dataset
available_features = [
    col for col in selected_features if col in df.columns
]

missing_features = [
    col for col in selected_features if col not in df.columns
]

selected_df = df[available_features].copy()

print("\nSelected features:", len(available_features))

if missing_features:
    print("Features not found in dataset:", missing_features)


# ==================================================
# 4. SAVE SELECTED DATASET
# ==================================================

selected_data_path = BASE_DIR / "data" / "selected_incidents.csv"

selected_df.to_csv(
    selected_data_path,
    index=False
)

print("\nSelected dataset saved successfully!")
print("Saved at:", selected_data_path)


# ==================================================
# 5. GRAPH 1: MISSING VALUE ANALYSIS
# ==================================================

missing_percent = (
    selected_df.isnull().mean() * 100
).sort_values(ascending=False)

missing_percent = missing_percent[missing_percent > 0]

plt.figure(figsize=(10, 6))

if not missing_percent.empty:

    sns.barplot(
        x=missing_percent.values,
        y=missing_percent.index
    )

    plt.xlabel("Missing Values (%)")
    plt.ylabel("Features")
    plt.title("Missing Value Analysis")

else:

    plt.text(
        0.5, 0.5,
        "No missing values found",
        ha="center",
        va="center"
    )

    plt.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_missing_values.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# 6. GRAPH 2: INCIDENT TYPE DISTRIBUTION
# ==================================================

if "incident_type" in selected_df.columns:

    incident_types = (
        selected_df["incident_type"]
        .dropna()
        .astype(str)
        .str.split(";")
        .explode()
        .str.strip()
    )

    incident_types = incident_types[
        incident_types.ne("")
    ]

    top_types = incident_types.value_counts().head(10)

    plt.figure(figsize=(10, 6))

    if not top_types.empty:

        sns.barplot(
            x=top_types.values,
            y=top_types.index
        )

        plt.xlabel("Number of Records")
        plt.ylabel("Incident Type")
        plt.title("Top Incident Types")

    else:

        plt.text(
            0.5, 0.5,
            "No incident type data available",
            ha="center",
            va="center"
        )

        plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "02_incident_types.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ==================================================
# 7. GRAPH 3: AFFECTED SECTOR DISTRIBUTION
# ==================================================

if "receiver_category" in selected_df.columns:

    sectors = (
        selected_df["receiver_category"]
        .dropna()
        .astype(str)
        .str.split(";")
        .explode()
        .str.strip()
    )

    sectors = sectors[sectors.ne("")]

    top_sectors = sectors.value_counts().head(10)

    plt.figure(figsize=(10, 6))

    if not top_sectors.empty:

        sns.barplot(
            x=top_sectors.values,
            y=top_sectors.index
        )

        plt.xlabel("Number of Records")
        plt.ylabel("Target Sector")
        plt.title("Top Affected Sectors")

    else:

        plt.text(
            0.5, 0.5,
            "No sector data available",
            ha="center",
            va="center"
        )

        plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "03_affected_sectors.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ==================================================
# 8. GRAPH 4: IMPACT SEVERITY DISTRIBUTION
# ==================================================

if "weighted_intensity" in selected_df.columns:

    intensity = pd.to_numeric(
        selected_df["weighted_intensity"],
        errors="coerce"
    ).dropna()

    plt.figure(figsize=(10, 6))

    if not intensity.empty:

        sns.histplot(
            intensity,
            bins=10,
            kde=True
        )

        plt.xlabel("Weighted Intensity")
        plt.ylabel("Number of Incidents")
        plt.title("Distribution of Incident Impact Severity")

    else:

        plt.text(
            0.5, 0.5,
            "No valid intensity values available",
            ha="center",
            va="center"
        )

        plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "04_impact_severity.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ==================================================
# 9. COMPLETION
# ==================================================

print("\n" + "=" * 45)
print("FEATURE SELECTION AND VISUALIZATION COMPLETED")
print("=" * 45)

print("\nGenerated output files:")

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print("-", file.name)

print("\nTotal graphs generated:", len(list(OUTPUT_DIR.glob("*.png"))))
