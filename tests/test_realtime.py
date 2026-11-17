"""Unit Tests for Real-Time Network Security & Visualization Module.
Validates PCAP parsing, traffic simulation, anomaly indicators, alert deduplication,
and event normalization using synthetic fixtures without requiring live network access.
"""

from pathlib import Path
import unittest
import time

from src.realtime.event_processor import is_private_ip, get_ip_geolocation, resolve_protocol
from src.realtime.traffic_simulator import generate_simulated_packets
from src.realtime.anomaly_detector import detect_realtime_anomalies
from src.realtime.alert_manager import AlertManager
from src.realtime.packet_collector import PacketCollector
from src.realtime.pcap_reader import create_sample_pcap, read_pcap_events


class TestEventProcessor(unittest.TestCase):
    """Test IP classification, protocol resolution, and geocoding."""

    def test_private_ip_check(self):
        self.assertTrue(is_private_ip("192.168.1.1"))
        self.assertTrue(is_private_ip("10.0.0.5"))
        self.assertTrue(is_private_ip("172.16.0.100"))
        self.assertTrue(is_private_ip("127.0.0.1"))
        self.assertFalse(is_private_ip("8.8.8.8"))
        self.assertFalse(is_private_ip("1.1.1.1"))
        self.assertFalse(is_private_ip("45.154.255.88"))

    def test_protocol_resolution(self):
        self.assertEqual(resolve_protocol("TCP", 54321, 443), "HTTPS")
        self.assertEqual(resolve_protocol("TCP", 54321, 80), "HTTP")
        self.assertEqual(resolve_protocol("UDP", 54321, 53), "DNS")
        self.assertEqual(resolve_protocol("TCP", 54321, 22), "SSH")
        self.assertEqual(resolve_protocol("ICMP", None, None), "ICMP")

    def test_geolocation_helper(self):
        res_priv = get_ip_geolocation("192.168.1.10")
        self.assertTrue(res_priv["is_private"])
        self.assertIn("Local", res_priv["country"])

        res_pub = get_ip_geolocation("8.8.8.8")
        self.assertFalse(res_pub["is_private"])
        self.assertEqual(res_pub["country"], "United States")
        self.assertIsNotNone(res_pub["lat"])


class TestTrafficSimulator(unittest.TestCase):
    """Test deterministic traffic generation scenarios."""

    def test_normal_traffic_generation(self):
        events = generate_simulated_packets("normal", count=20)
        self.assertEqual(len(events), 20)
        self.assertTrue(all(e.source_type == "Simulated Traffic" for e in events))

    def test_port_scan_simulation(self):
        events = generate_simulated_packets("port_scan", count=25)
        self.assertEqual(len(events), 25)
        # Verify scanner IP and TCP SYN flags
        self.assertTrue(any(e.tcp_flags == "SYN" for e in events))
        unique_ports = {e.dst_port for e in events}
        self.assertGreaterEqual(len(unique_ports), 8)

    def test_syn_flood_simulation(self):
        events = generate_simulated_packets("syn_flood", count=30)
        self.assertEqual(len(events), 30)
        self.assertTrue(all(e.dst_port == 443 for e in events))
        self.assertTrue(all("SYN" in e.tcp_flags for e in events))


class TestAnomalyDetector(unittest.TestCase):
    """Test real-time suspicious pattern detections."""

    def test_port_scan_detection(self):
        events = generate_simulated_packets("port_scan", count=30)
        anomalies = detect_realtime_anomalies(events, window_seconds=60.0)
        categories = [a["category"] for a in anomalies]
        self.assertIn("Port Scan", categories)
        port_scan_anom = next(a for a in anomalies if a["category"] == "Port Scan")
        self.assertEqual(port_scan_anom["src_ip"], "45.154.255.88")
        self.assertIn(port_scan_anom["severity"], ["Critical", "High"])

    def test_syn_flood_detection(self):
        events = generate_simulated_packets("syn_flood", count=35)
        anomalies = detect_realtime_anomalies(events, window_seconds=60.0)
        categories = [a["category"] for a in anomalies]
        self.assertIn("SYN Flood", categories)

    def test_dns_anomaly_detection(self):
        events = generate_simulated_packets("dns_anomaly", count=15)
        anomalies = detect_realtime_anomalies(events, window_seconds=60.0)
        categories = [a["category"] for a in anomalies]
        self.assertIn("DNS Anomaly", categories)


class TestAlertManager(unittest.TestCase):
    """Test alert deduplication and cooldown."""

    def test_deduplication_cooldown(self):
        mgr = AlertManager(cooldown_seconds=5.0)
        mock_anom = [{
            "category": "Port Scan",
            "severity": "High",
            "src_ip": "10.0.0.99",
            "dst_ip": "192.168.1.1",
            "summary": "Port scan from 10.0.0.99",
            "explanation": "Scanned 15 ports",
            "evidence": "15 ports",
            "data_source": "Simulated Traffic",
            "timestamp": time.time()
        }]

        # First alert should be created
        a1 = mgr.process_anomalies(mock_anom)
        self.assertEqual(len(a1), 1)
        self.assertEqual(len(mgr.alerts), 1)

        # Immediate second identical anomaly should be suppressed by cooldown
        a2 = mgr.process_anomalies(mock_anom)
        self.assertEqual(len(a2), 0)
        self.assertEqual(len(mgr.alerts), 1)

    def test_mark_reviewed(self):
        mgr = AlertManager(cooldown_seconds=1.0)
        mock_anom = [{
            "category": "Traffic Burst",
            "severity": "Medium",
            "src_ip": "Aggregated",
            "dst_ip": "Multiple",
            "summary": "Traffic burst",
            "explanation": "High PPS",
            "evidence": "50 PPS",
            "data_source": "Simulated Traffic",
            "timestamp": time.time()
        }]
        a = mgr.process_anomalies(mock_anom)
        aid = a[0].alert_id
        self.assertEqual(a[0].status, "New")

        mgr.mark_reviewed(aid)
        df = mgr.get_dataframe()
        self.assertEqual(df[df["alert_id"] == aid]["status"].iloc[0], "Reviewed")


class TestPcapIntegration(unittest.TestCase):
    """Test sample PCAP creation and parsing."""

    def test_sample_pcap_read(self):
        pcap_path = Path("data/realtime/sample_traffic.pcap")
        if not pcap_path.exists():
            create_sample_pcap(pcap_path)

        events = read_pcap_events(pcap_path)
        self.assertGreater(len(events), 30)
        self.assertTrue(all(e.source_type == "PCAP Replay" for e in events))


if __name__ == "__main__":
    unittest.main()
