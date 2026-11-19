
import pandas as pd

df = pd.read_csv("data/selected_incidents.csv")

target_columns = ["mitre_initial_access", "mitre_impact"]

for col in target_columns:
    print("\n" + "=" * 60)
    print(f"TARGET COLUMN: {col}")
    print("=" * 60)

    print("\nMissing values:", df[col].isna().sum())

    print("\nUnique raw values:", df[col].nunique())

    print("\nMost frequent labels:")
    print(df[col].value_counts(dropna=False).head(25))

    multi_label_count = (
        df[col].fillna("").astype(str).str.contains(";").sum()
    )
    print("\nRows containing multiple labels:", multi_label_count)
