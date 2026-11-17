"""Phishing detection inference and explainability engine.
Performs offline evaluation of user-submitted URLs using the trained ML model,
extracts features, computes analytical risk score (0-100), and generates
human-understandable explanations of threat indicators.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import joblib
import pandas as pd

from src.phishing.features import FEATURE_NAMES, extract_url_features, sanitize_url

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT_DIR / "models" / "phishing_model" / "phishing_detector.pkl"
META_PATH = ROOT_DIR / "models" / "phishing_model" / "model_metadata.json"

_CACHED_MODEL = None
_CACHED_META = None


def get_model():
    """Load and cache the trained phishing model."""
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Phishing model not found at {MODEL_PATH}. Run src/phishing/train.py first."
            )
        _CACHED_MODEL = joblib.load(MODEL_PATH)
    return _CACHED_MODEL


def get_metadata() -> Dict[str, Any]:
    """Load and cache the model training metadata and evaluation statistics."""
    global _CACHED_META
    if _CACHED_META is None:
        if META_PATH.exists():
            with open(META_PATH, "r", encoding="utf-8") as f:
                _CACHED_META = json.load(f)
        else:
            _CACHED_META = {}
    return _CACHED_META


def assess_phishing_url(raw_url: str) -> Dict[str, Any]:
    """
    Safely evaluate a URL for phishing characteristics.
    Returns risk score (0-100), severity level, extracted features,
    explainable contributing factors, and security recommendations.
    """
    cleaned_url = sanitize_url(raw_url)
    if not cleaned_url or len(cleaned_url.strip()) < 4:
        return {
            "error": "Please enter a valid URL.",
            "url": raw_url,
            "risk_score": 0,
            "risk_level": "UNKNOWN",
            "is_phishing": False,
            "probability": 0.0,
            "factors": [],
            "positive_indicators": [],
            "features": {},
            "recommendation": "Provide a complete URL to analyze."
        }

    features = extract_url_features(cleaned_url)
    feature_df = pd.DataFrame([features])[FEATURE_NAMES]

    # Predict probability via ML model
    model = get_model()
    try:
        if hasattr(model, "predict_proba"):
            phish_prob = float(model.predict_proba(feature_df)[0][1])
        else:
            phish_prob = float(model.predict(feature_df)[0])
    except Exception as exc:
        phish_prob = 0.5

    # Scale to 0-100 Analytical Risk Score
    risk_score = int(round(phish_prob * 100))

    # Categorize Risk Level
    if risk_score <= 30:
        risk_level = "LOW"
        risk_badge = "Likely Benign"
        risk_color = "#43d6a2"
    elif risk_score <= 60:
        risk_level = "MEDIUM"
        risk_badge = "Suspicious"
        risk_color = "#f3b76b"
    elif risk_score <= 80:
        risk_level = "HIGH"
        risk_badge = "Probable Phishing"
        risk_color = "#ff8552"
    else:
        risk_level = "CRITICAL"
        risk_badge = "High-Risk Threat"
        risk_color = "#ff4d6d"

    # Explainability: Extract contributing factors
    factors: List[str] = []
    positive_indicators: List[str] = []

    if features["is_ip"] == 1:
        factors.append("Direct IP address used in host instead of standard domain name")
    if features["suspicious_keyword_count"] > 0:
        factors.append(f"Contains {features['suspicious_keyword_count']} high-risk authentication/banking credential keywords")
    if features["num_subdomains"] >= 2:
        factors.append(f"Excessive subdomain nesting ({features['num_subdomains']} subdomains) often used in spoofing")
    if features["has_punycode"] == 1:
        factors.append("Punycode / internationalized domain format detected (potential homograph attack)")
    if features["is_shortened"] == 1:
        factors.append("Known URL shortener service masking true destination")
    if features["num_at"] > 0:
        factors.append("Contains '@' character in URL, commonly used to disguise destination domain")
    if features["num_hyphens"] >= 3:
        factors.append(f"High number of hyphens ({features['num_hyphens']}) typical of deceptive typosquats")
    if features["url_length"] > 75:
        factors.append(f"Abnormally long URL ({features['url_length']} characters)")
    if features["domain_length"] > 30:
        factors.append(f"Unusually long domain name ({features['domain_length']} characters)")
    if features["entropy"] > 4.5:
        factors.append(f"High character randomness / entropy ({features['entropy']}), characteristic of DGA or hash tokens")
    if features["is_https"] == 0:
        factors.append("Insecure HTTP transport (no TLS encryption)")
    else:
        positive_indicators.append("TLS encryption enabled (HTTPS scheme present)")

    if features["num_subdomains"] <= 1 and features["is_ip"] == 0:
        positive_indicators.append("Standard domain hierarchy without suspicious nesting")
    if features["suspicious_keyword_count"] == 0:
        positive_indicators.append("No credential harvesting keywords detected in URL path")
    if features["entropy"] < 3.8:
        positive_indicators.append("Normal natural-language character entropy")

    # Actionable Recommendation
    if risk_score >= 61:
        recommendation = (
            "CRITICAL: Do NOT enter credentials, passwords, or personal financial details. "
            "The URL displays strong structural and lexical hallmarks of a phishing campaign."
        )
    elif risk_score >= 31:
        recommendation = (
            "CAUTION: Exercise vigilance. Verify the sender and domain identity independently "
            "before clicking links or submitting any corporate or sensitive information."
        )
    else:
        recommendation = (
            "NORMAL: Structural indicators align with legitimate web patterns. Standard browsing "
            "hygiene applies."
        )

    return {
        "url": cleaned_url,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_badge": risk_badge,
        "risk_color": risk_color,
        "is_phishing": risk_score >= 50,
        "probability": round(phish_prob, 4),
        "factors": factors,
        "positive_indicators": positive_indicators,
        "features": features,
        "recommendation": recommendation
    }
