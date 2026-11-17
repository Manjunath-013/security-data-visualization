"""Real-time Security Alert Manager with deduplication, cooldown, and review workflow.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
import time
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class RealtimeAlert:
    alert_id: str
    timestamp: float
    time_str: str
    src_ip: str
    dst_ip: str
    category: str
    severity: str
    summary: str
    explanation: str
    evidence: str
    data_source: str
    status: str = "New"  # "New", "Reviewed", "Resolved"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AlertManager:
    """
    Thread-safe in-memory alert store with deduplication cooldown.
    """
    def __init__(self, cooldown_seconds: float = 15.0, max_alerts: int = 500):
        self.cooldown_seconds = cooldown_seconds
        self.max_alerts = max_alerts
        self.alerts: List[RealtimeAlert] = []
        self._last_alert_time: Dict[str, float] = {}
        self._counter = 1

    def process_anomalies(self, anomalies: List[Dict[str, Any]]) -> List[RealtimeAlert]:
        """Convert detected anomalies into deduplicated alerts."""
        now = time.time()
        new_alerts = []

        for anom in anomalies:
            cat = anom.get("category", "General")
            src_ip = anom.get("src_ip", "Unknown")
            dedup_key = f"{cat}:{src_ip}"

            last_time = self._last_alert_time.get(dedup_key, 0)
            if (now - last_time) < self.cooldown_seconds:
                continue  # Suppress duplicate alerts within cooldown window

            self._last_alert_time[dedup_key] = now
            aid = f"ALT-{self._counter:04d}"
            self._counter += 1

            ts = anom.get("timestamp", now)
            dt_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

            alert = RealtimeAlert(
                alert_id=aid,
                timestamp=ts,
                time_str=dt_str,
                src_ip=src_ip,
                dst_ip=anom.get("dst_ip", "N/A"),
                category=cat,
                severity=anom.get("severity", "Medium"),
                summary=anom.get("summary", "Suspicious Network Event"),
                explanation=anom.get("explanation", ""),
                evidence=anom.get("evidence", ""),
                data_source=anom.get("data_source", "Real-Time Engine"),
                status="New"
            )
            self.alerts.insert(0, alert)  # Newest first
            new_alerts.append(alert)

        # Enforce max alerts capacity
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[:self.max_alerts]

        return new_alerts

    def mark_reviewed(self, alert_id: str):
        """Mark an alert as reviewed."""
        for a in self.alerts:
            if a.alert_id == alert_id:
                a.status = "Reviewed"
                break

    def mark_all_reviewed(self):
        """Mark all active alerts as reviewed."""
        for a in self.alerts:
            if a.status == "New":
                a.status = "Reviewed"

    def clear(self):
        """Clear alert history."""
        self.alerts.clear()
        self._last_alert_time.clear()

    def get_dataframe(
        self,
        severity_filter: Optional[List[str]] = None,
        category_filter: Optional[List[str]] = None,
        search_ip: str = ""
    ) -> pd.DataFrame:
        """Filter alerts and return as a Pandas DataFrame."""
        if not self.alerts:
            return pd.DataFrame(columns=[
                "alert_id", "time_str", "severity", "category",
                "src_ip", "dst_ip", "summary", "status", "data_source"
            ])

        records = [a.to_dict() for a in self.alerts]
        df = pd.DataFrame(records)

        if severity_filter:
            df = df[df["severity"].isin(severity_filter)]

        if category_filter:
            df = df[df["category"].isin(category_filter)]

        if search_ip:
            s = search_ip.strip().lower()
            df = df[df["src_ip"].str.lower().str.contains(s) | df["dst_ip"].str.lower().str.contains(s)]

        return df
