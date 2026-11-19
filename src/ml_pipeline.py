"""Single-target ML pipeline for MITRE ATT&CK Impact prediction.

Run from the project root:
    python -m src.ml_pipeline

The pipeline trains ONE target only: mitre_impact.
MITRE Initial Access remains available for historical analysis, not ML prediction.
"""

import json
import re
import warnings
from pathlib import Path

import joblib
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "selected_incidents.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs" / "ml"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
MIN_CLASS_COUNT = 10
CV_SPLITS = 4

# The only ML target in this project.
TARGET_COLUMN = "mitre_impact"
MODEL_PATH = MODEL_DIR / "mitre_impact_final_model.joblib"
METRICS_PATH = OUTPUT_DIR / "mitre_impact_final_metrics.json"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "mitre_impact_final_confusion_matrix.png"

# Only fields expected to be available when a new incident is described.
# Do not add target labels or post-incident outcome/impact fields here.
TEXT_COLUMNS = [
    "name",
    "description",
    "incident_type",
    "receiver_category",
    "receiver_subcategory",
    "receiver_country",
    "receiver_regions",
]


# ----------------------------------------------------------------
# 2. DATA PREPARATION
# ----------------------------------------------------------------

def clean_value(value):
    """Normalize text and represent missing feature values explicitly."""
    if pd.isna(value):
        return "__missing__"
    value = re.sub(r"\s+", " ", str(value).strip()).replace(";", " ")
    return value if value else "__missing__"


def prepare_features(df):
    """Return model inputs in the exact order used during training."""
    X = df.copy()
    for column in TEXT_COLUMNS:
        if column not in X.columns:
            X[column] = "__missing__"
        X[column] = X[column].apply(clean_value)
    return X[TEXT_COLUMNS].copy()


def prepare_target(df):
    """Remove unknown labels and exclude classes with fewer than 10 records."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COLUMN}")

    y = df[TARGET_COLUMN].fillna("").astype(str).str.strip()
    known_mask = y.ne("") & y.str.lower().ne("not available")

    X_known = prepare_features(df).loc[known_mask].copy()
    y_known = y.loc[known_mask].copy()

    counts = y_known.value_counts()
    supported_classes = counts[counts >= MIN_CLASS_COUNT].index
    rare_classes = {
        str(label): int(count)
        for label, count in counts[counts < MIN_CLASS_COUNT].items()
    }

    keep = y_known.isin(supported_classes)
    X = X_known.loc[keep].copy()
    y = y_known.loc[keep].copy()

    if y.nunique() < 2:
        raise ValueError("At least two supported MITRE Impact classes are required.")
    if y.value_counts().min() < CV_SPLITS:
        raise ValueError(
            f"Every retained class needs at least {CV_SPLITS} records "
            "for the configured stratified cross-validation."
        )

    unknown_count = int((~known_mask).sum())
    return X, y, rare_classes, unknown_count


# ----------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ----------------------------------------------------------------

def make_word_tfidf(max_features, ngram_range=(1, 2)):
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=1,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )


def create_feature_transformer():
    """Separate word TF-IDF by field; description also gets character TF-IDF."""
    transformers = [
        ("description_word", make_word_tfidf(30000), "description"),
        (
            "description_char",
            TfidfVectorizer(
                lowercase=True,
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=30000,
                min_df=1,
                sublinear_tf=True,
            ),
            "description",
        ),
        ("name_word", make_word_tfidf(10000), "name"),
        ("incident_type", make_word_tfidf(5000), "incident_type"),
        ("sector", make_word_tfidf(5000), "receiver_category"),
        ("subsector", make_word_tfidf(8000), "receiver_subcategory"),
        ("country", make_word_tfidf(2000, (1, 1)), "receiver_country"),
        ("region", make_word_tfidf(2000, (1, 1)), "receiver_regions"),
    ]
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )


# ----------------------------------------------------------------
# 4. MODEL CANDIDATES AND TUNING
# ----------------------------------------------------------------

def get_model_configurations():
    c_values = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    return {
        "Linear SVM": {
            "estimator": LinearSVC(
                class_weight="balanced", random_state=RANDOM_STATE
            ),
            "parameters": {"classifier__C": c_values},
        },
        "Logistic Regression": {
            "estimator": LogisticRegression(
                max_iter=4000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),
            "parameters": {"classifier__C": c_values},
        },
        "Complement Naive Bayes": {
            "estimator": ComplementNB(),
            "parameters": {
                "classifier__alpha": [0.1, 0.25, 0.5, 1.0, 2.0]
            },
        },
    }


def create_pipeline(classifier):
    return Pipeline([
        ("features", create_feature_transformer()),
        ("classifier", classifier),
    ])


def tune_models(X_train, y_train):
    """Select the candidate with the highest training-fold CV macro F1."""
    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    searches = {}
    comparison = []

    print("\nMODEL SELECTION: STRATIFIED CROSS-VALIDATION")
    print("-" * 68)

    for model_name, config in get_model_configurations().items():
        print(f"\nTuning {model_name}...")
        search = GridSearchCV(
            estimator=create_pipeline(config["estimator"]),
            param_grid=config["parameters"],
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        searches[model_name] = search
        comparison.append({
            "model": model_name,
            "cv_macro_f1": float(search.best_score_),
            "best_parameters": search.best_params_,
        })
        print(f"Best CV macro F1: {search.best_score_:.4f}")
        print(f"Best parameters: {search.best_params_}")

    comparison.sort(key=lambda row: row["cv_macro_f1"], reverse=True)
    best_name = comparison[0]["model"]
    print("\nMODEL COMPARISON")
    print(pd.DataFrame([
        {"Model": row["model"], "CV Macro F1": row["cv_macro_f1"]}
        for row in comparison
    ]).to_string(index=False))
    print(f"\nSelected model: {best_name}")
    return best_name, searches[best_name], comparison


# ----------------------------------------------------------------
# 5. EVALUATION
# ----------------------------------------------------------------

def calculate_metrics(y_true, y_pred):
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
    }


def save_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(
        figsize=(max(10, len(labels) * 1.15), max(8, len(labels) * 0.9))
    )
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=labels
    )
    display.plot(
        ax=ax, cmap="Blues", xticks_rotation=45,
        values_format="d", colorbar=False,
    )
    ax.set_title("MITRE Impact - Independent Holdout Confusion Matrix")
    ax.set_xlabel("Predicted MITRE Impact")
    ax.set_ylabel("Actual MITRE Impact")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return CONFUSION_MATRIX_PATH


# ----------------------------------------------------------------
# 6. TRAIN, EVALUATE, REFIT AND SAVE THE ONE MODEL
# ----------------------------------------------------------------

def train_impact_model(df):
    print("\n" + "=" * 72)
    print("SINGLE ML TASK: MITRE ATT&CK IMPACT PREDICTION")
    print("=" * 72)

    X, y, rare_classes, unknown_count = prepare_target(df)

    print("\nDATASET SUMMARY")
    print(f"Total records: {len(df)}")
    print(f"Unknown / unavailable target labels: {unknown_count}")
    print(f"Eligible labelled records: {len(y)}")
    print(f"Retained classes: {y.nunique()}")
    print("\nCLASS DISTRIBUTION")
    print(y.value_counts().to_string())

    print("\nEXCLUDED RARE CLASSES (< 10 records)")
    if rare_classes:
        for label, count in rare_classes.items():
            print(f"  {label}: {count}")
    else:
        print("  None")

    # Holdout is isolated from hyperparameter selection.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\nTraining split: {len(X_train)}")
    print(f"Independent holdout split: {len(X_test)}")

    best_name, best_search, comparison = tune_models(X_train, y_train)
    holdout_model = best_search.best_estimator_

    # One final evaluation on the untouched holdout set.
    y_pred = holdout_model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)

    print("\nINDEPENDENT HOLDOUT METRICS")
    for metric, value in metrics.items():
        print(f"{metric:<22}: {value:.4f}")

    report_dict = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("\nHOLDOUT CLASS SUPPORT")
    print(y_test.value_counts().sort_index().to_string())

    labels = sorted(y.unique())
    matrix_path = save_confusion_matrix(y_test, y_pred, labels)
    print(f"\nConfusion matrix saved: {matrix_path}")

    # Train the deployable artifact on all eligible labelled records.
    # Holdout metrics above remain the independent evaluation results.
    final_classifier = get_model_configurations()[best_name]["estimator"]
    final_model = create_pipeline(final_classifier)
    final_model.set_params(**best_search.best_params_)
    final_model.fit(X, y)

    model_bundle = {
        "pipeline": final_model,
        "target": TARGET_COLUMN,
        "model_name": best_name,
        "input_columns": TEXT_COLUMNS,
        "classes": [str(label) for label in final_model.classes_],
        "best_parameters": best_search.best_params_,
        "training_records": int(len(X)),
        "minimum_class_count": MIN_CLASS_COUNT,
        "excluded_rare_classes": rare_classes,
        "unknown_target_records": unknown_count,
        "evaluation_note": (
            "Metrics are from the independent holdout split. The saved "
            "model is refitted on all eligible labelled records."
        ),
    }
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"\nFinal model saved: {MODEL_PATH}")

    metrics_output = {
        "target": TARGET_COLUMN,
        "best_model": best_name,
        "total_dataset_records": int(len(df)),
        "unknown_target_records": unknown_count,
        "training_records_before_final_refit": int(len(X_train)),
        "holdout_records": int(len(X_test)),
        "final_model_training_records": int(len(X)),
        "classes": labels,
        "class_distribution": {
            str(label): int(count) for label, count in y.value_counts().items()
        },
        "excluded_rare_classes": rare_classes,
        "model_comparison": comparison,
        "best_parameters": best_search.best_params_,
        "holdout_metrics": metrics,
        "classification_report": report_dict,
        "confusion_matrix_path": str(matrix_path),
        "final_model_path": str(MODEL_PATH),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics_output, file, indent=4)
    print(f"Metrics saved: {METRICS_PATH}")
    return metrics_output


# ----------------------------------------------------------------
# 7. PREDICTION: MITRE IMPACT ONLY
# ----------------------------------------------------------------

_CACHED_BUNDLE = None

def get_model_bundle():
    """Load and cache the trained model bundle in memory."""
    global _CACHED_BUNDLE
    if _CACHED_BUNDLE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}. Run training first."
            )
        _CACHED_BUNDLE = joblib.load(MODEL_PATH)
    return _CACHED_BUNDLE

def predict_new_incident(incident):
    """Predict one MITRE Impact label using the saved model bundle."""
    bundle = get_model_bundle()
    pipeline = bundle["pipeline"]
    input_df = prepare_features(pd.DataFrame([incident]))
    prediction = str(pipeline.predict(input_df)[0])

    result = {
        "target": TARGET_COLUMN,
        "prediction": prediction,
        "model": bundle["model_name"],
        "score_type": None,
        "top_scores": [],
    }

    classifier = pipeline.named_steps["classifier"]

    # Scores are useful for ranking candidate labels, but SVM decision
    # scores are NOT probabilities and must not be displayed as confidence.
    if hasattr(classifier, "decision_function"):
        scores = np.asarray(pipeline.decision_function(input_df))
        if scores.ndim == 1:
            scores = np.array([[-scores[0], scores[0]]])
        scores = scores[0]
        ranked = sorted(
            zip(classifier.classes_, scores),
            key=lambda item: item[1],
            reverse=True,
        )
        result["score_type"] = "relative decision score (not probability)"
    elif hasattr(classifier, "predict_proba"):
        scores = pipeline.predict_proba(input_df)[0]
        ranked = sorted(
            zip(classifier.classes_, scores),
            key=lambda item: item[1],
            reverse=True,
        )
        result["score_type"] = "model probability estimate"
    else:
        ranked = []

    result["top_scores"] = [
        {"label": str(label), "score": float(score)}
        for label, score in ranked[:3]
    ]
    return result


def run_sample_prediction():
    sample_incident = {
        "name": "Example suspicious email campaign",
        "description": (
            "An organization reported a targeted email campaign containing "
            "malicious links that redirected employees to a credential "
            "collection page."
        ),
        "incident_type": "Data theft",
        "receiver_category": "Corporate",
        "receiver_subcategory": "Business",
        "receiver_country": "United States",
        "receiver_regions": "North America",
    }

    result = predict_new_incident(sample_incident)
    print("\nSAMPLE INCIDENT PREDICTION")
    print("-" * 60)
    print("Incident:", sample_incident["name"])
    print("Target:", result["target"])
    print("Predicted MITRE Impact:", result["prediction"])
    print("Model:", result["model"])
    if result["top_scores"]:
        print(f"Top candidates ({result['score_type']}):")
        for item in result["top_scores"]:
            print(f"  {item['label']}: {item['score']:.4f}")
    print(
        "\nNote: This is a model estimate from historical reported incidents, "
        "not a verified MITRE ATT&CK classification. SVM decision scores "
        "are not probabilities."
    )


# ----------------------------------------------------------------
# 8. MAIN
# ----------------------------------------------------------------

def main():
    print("\n" + "=" * 72)
    print("SECURITY DATA VISUALIZATION - ONE MODEL ML PIPELINE")
    print("=" * 72)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Expected file: data/selected_incidents.csv"
        )

    df = pd.read_csv(DATA_PATH)
    print(f"\nLoaded: {DATA_PATH}")
    print(f"Dataset shape: {df.shape}")

    missing = [col for col in TEXT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing model input columns: {missing}")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    print("\nModel input columns:")
    for column in TEXT_COLUMNS:
        print(f"  - {column}")
    print(f"\nOnly prediction target: {TARGET_COLUMN}")

    result = train_impact_model(df)

    print("\nFINAL SUMMARY")
    print("-" * 60)
    print("Selected model:", result["best_model"])
    print(f"Holdout accuracy: {result['holdout_metrics']['accuracy']:.4f}")
    print(f"Holdout macro F1: {result['holdout_metrics']['macro_f1']:.4f}")
    print(
        "Holdout weighted F1: "
        f"{result['holdout_metrics']['weighted_f1']:.4f}"
    )
    print("Final model records:", result["final_model_training_records"])

    run_sample_prediction()

    print("\nGENERATED FILES")
    print("Model:", MODEL_PATH)
    print("Metrics:", METRICS_PATH)
    print("Confusion matrix:", CONFUSION_MATRIX_PATH)
    print("\nMITRE Impact ML pipeline completed.")


if __name__ == "__main__":
    main()
