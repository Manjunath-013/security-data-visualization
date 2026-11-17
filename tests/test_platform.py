"""Comprehensive Test Suite for Security Intelligence & Phishing Detection Platform.
Tests URL feature extraction, ML prediction pipeline, analytical risk engine,
anomaly detection, dataset preprocessing, and edge case resiliency.
"""

from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from src.anomaly_detector import detect_volume_anomalies, run_all_anomaly_detections
from src.phishing.detector import assess_phishing_url, get_metadata, get_model
from src.phishing.features import FEATURE_NAMES, extract_url_features, sanitize_url
from src.preprocessing import _to_bool, normalize_multivalue
from src.risk_engine import compute_risk_scores, evaluate_incident_risk


class TestPhishingFeatures(unittest.TestCase):
    """Test URL feature extraction and parsing."""

    def test_sanitize_url(self):
        self.assertEqual(sanitize_url("example.com"), "https://example.com")
        self.assertEqual(sanitize_url("http://test.org/path"), "http://test.org/path")
        self.assertEqual(sanitize_url(""), "")

    def test_feature_keys(self):
        feats = extract_url_features("https://www.google.com/search?q=cybersecurity")
        for k in FEATURE_NAMES:
            self.assertIn(k, feats, f"Missing feature: {k}")

    def test_ip_address_detection(self):
        ip_url = "http://192.168.1.100/login/auth"
        feats = extract_url_features(ip_url)
        self.assertEqual(feats["is_ip"], 1)

        normal_url = "https://www.chase.com/login"
        feats_normal = extract_url_features(normal_url)
        self.assertEqual(feats_normal["is_ip"], 0)

    def test_suspicious_keyword_count(self):
        phish_url = "http://account-verify-update.com/login/secure/banking/signin"
        feats = extract_url_features(phish_url)
        self.assertGreaterEqual(feats["suspicious_keyword_count"], 4)

    def test_entropy_computation(self):
        entropy = extract_url_features("https://example.com")["entropy"]
        self.assertGreater(entropy, 0.0)


class TestPhishingDetector(unittest.TestCase):
    """Test offline ML inference and explainability."""

    def test_benign_url_assessment(self):
        res = assess_phishing_url("https://www.wikipedia.org/wiki/Computer_security")
        self.assertIn("risk_score", res)
        self.assertIn("risk_level", res)
        self.assertLessEqual(res["risk_score"], 40)
        self.assertIn("recommendation", res)

    def test_phishing_url_assessment(self):
        phish = "http://192.168.1.50/paypal-security-update/login.php?token=9281928"
        res = assess_phishing_url(phish)
        self.assertGreaterEqual(res["risk_score"], 60)
        self.assertIn(res["risk_level"], ["HIGH", "CRITICAL"])
        self.assertTrue(len(res["factors"]) > 0)

    def test_malformed_url_resilience(self):
        res = assess_phishing_url("   ")
        self.assertEqual(res["risk_level"], "UNKNOWN")

    def test_metadata_loading(self):
        meta = get_metadata()
        self.assertIn("production_model", meta)
        self.assertIn("feature_importances", meta)


class TestRiskEngine(unittest.TestCase):
    """Test incident risk scoring engine."""

    def test_low_risk_incident(self):
        row = pd.Series({
            "has_disruption": False,
            "disruption": "Not available",
            "data_theft": "Not available",
            "hijacking": "Not available",
            "zero_days": "No",
            "mitre_impact_list": [],
            "sector_list": ["Retail"],
            "country_list": ["Germany"]
        })
        res = evaluate_incident_risk(row)
        self.assertEqual(res["risk_level"], "LOW")
        self.assertLessEqual(res["risk_score"], 30)

    def test_critical_risk_incident(self):
        row = pd.Series({
            "has_disruption": True,
            "disruption": "Long-term disruption (> 24h)",
            "data_theft": "sensitive information stolen",
            "hijacking": "Host hijacking",
            "zero_days": "Yes",
            "mitre_impact_list": ["Disk Wipe", "Data Destruction"],
            "sector_list": ["Government / ministries", "Energy"],
            "initiator_category": "State",
            "country_list": ["Ukraine", "Poland"],
            "weighted_intensity": 8.0
        })
        res = evaluate_incident_risk(row)
        self.assertEqual(res["risk_level"], "CRITICAL")
        self.assertGreaterEqual(res["risk_score"], 81)
        self.assertTrue(any("disruption" in f.lower() for f in res["contributing_factors"]))


class TestAnomalyDetector(unittest.TestCase):
    """Test historical statistical anomaly detection."""

    def test_volume_anomaly(self):
        # Create synthetic time series with a giant spike in 2024
        dates = ["2020-01-01"] * 50 + ["2021-01-01"] * 60 + ["2022-01-01"] * 55 + ["2023-01-01"] * 70 + ["2024-01-01"] * 600
        df = pd.DataFrame({
            "incident_id": range(len(dates)),
            "start_date": pd.to_datetime(dates),
            "year": pd.to_datetime(dates).year
        })
        anomalies = detect_volume_anomalies(df, freq="year")
        self.assertTrue(len(anomalies) > 0)
        self.assertEqual(anomalies[0]["period"], "2024")

    def test_empty_dataframe_handling(self):
        anomalies = run_all_anomaly_detections(pd.DataFrame())
        self.assertEqual(anomalies, [])


class TestPreprocessingHelpers(unittest.TestCase):
    """Test data normalization helpers."""

    def test_normalize_multivalue(self):
        raw = "Germany; France | Italy; Unknown ; "
        res = normalize_multivalue(raw)
        self.assertEqual(res, ["Germany", "France", "Italy"])

    def test_to_bool(self):
        self.assertTrue(_to_bool("True"))
        self.assertTrue(_to_bool("yes"))
        self.assertTrue(_to_bool(1))
        self.assertFalse(_to_bool("false"))
        self.assertFalse(_to_bool(None))


if __name__ == "__main__":
    unittest.main()
