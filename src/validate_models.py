
# src/validate_models.py

import pandas as pd

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.feature_engineering import (
    prepare_target_data,
    build_model_text,
    INPUT_COLUMNS
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "selected_incidents.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "ml"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def create_pipeline(classifier):

    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_features=20000,
                min_df=1,
                sublinear_tf=True
            )
        ),
        ("classifier", classifier)
    ])


def get_classifiers():

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Linear SVM": LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    }


def make_text_without_incident_type(df):

    columns = [
        col for col in INPUT_COLUMNS
        if col != "incident_type"
    ]

    combined = pd.Series("", index=df.index, dtype="object")

    for col in columns:
        values = df[col].fillna("").astype(str)
        combined += " " + col + ": " + values

    return combined.str.strip()


def validate_target(df, target):

    print("\n" + "=" * 70)
    print("VALIDATION TARGET:", target)
    print("=" * 70)

    X, y, excluded, _ = prepare_target_data(df, target)

    # Recreate the eligible rows using the same filtering rules.
    valid = df.loc[X.index].copy()

    X_without_type = make_text_without_incident_type(valid)

    print("Eligible records:", len(y))
    print("Number of classes:", y.nunique())
    print("Excluded rare classes:", excluded)

    min_count = y.value_counts().min()

    if min_count < 5:
        raise ValueError(
            "5-fold stratified validation requires at least "
            "5 examples in every retained class."
        )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted"
    }

    configurations = {
        "All features": X,
        "Without incident_type": X_without_type
    }

    records = []

    for config_name, features in configurations.items():

        for model_name, classifier in get_classifiers().items():

            pipeline = create_pipeline(classifier)

            scores = cross_validate(
                pipeline,
                features,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                error_score="raise"
            )

            record = {
                "target": target,
                "features": config_name,
                "model": model_name
            }

            for metric in scoring:
                values = scores[f"test_{metric}"]

                record[f"{metric}_mean"] = values.mean()
                record[f"{metric}_std"] = values.std()

            records.append(record)

            print(f"\n{config_name} | {model_name}")

            for metric in scoring:
                mean = record[f"{metric}_mean"]
                std = record[f"{metric}_std"]

                print(f"{metric}: {mean:.4f} (+/- {std:.4f})")

    return records


def main():

    df = pd.read_csv(DATA_PATH)

    all_records = []

    targets = [
        "mitre_initial_access",
        "mitre_impact"
    ]

    for target in targets:
        records = validate_target(df, target)
        all_records.extend(records)

    results = pd.DataFrame(all_records)

    results = results.sort_values(
        ["target", "f1_macro_mean"],
        ascending=[True, False]
    )

    output_path = OUTPUT_DIR / "cross_validation_results.csv"

    results.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    print(
        results[
            [
                "target",
                "features",
                "model",
                "accuracy_mean",
                "f1_macro_mean",
                "f1_weighted_mean"
            ]
        ].to_string(index=False)
    )

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()
