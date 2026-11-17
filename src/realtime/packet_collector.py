"""Thread-safe Packet Collector and Stream Engine.
Coordinates live network capture, PCAP replay, and simulated traffic streams
into a bounded, thread-safe event buffer with rolling metrics calculation.
"""

from collections import deque
from datetime import datetime
from pathlib import Path
import threading
import time
from typing import Any, Deque, Dict, List, Optional
import pandas as pd

from src.realtime.event_processor import PacketEvent
from src.realtime.pcap_reader import read_pcap_events
from src.realtime.traffic_simulator import generate_simulated_packets
from src.realtime.anomaly_detector import detect_realtime_anomalies
from src.realtime.alert_manager import AlertManager


class PacketCollector:
    """
    Singleton-friendly thread-safe packet ingestion engine.
    """
    def __init__(self, buffer_size: int = 5000):
        self.buffer_size = buffer_size
        self._buffer: Deque[PacketEvent] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        
        self.alert_manager = AlertManager()
        
        # State tracking
        self.is_running = False
        self.is_paused = False
        self.active_source = "Idle / Not Started"
        self.current_scenario = "normal"
        self.packets_collected_total = 0
        self.bytes_collected_total = 0
        self.start_time: Optional[float] = None
        
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # PCAP Replay buffer
        self._pcap_queue: Deque[PacketEvent] = deque()

    def start_simulation(self, scenario: str = "normal", interval: float = 1.0):
        """Start generating simulated network traffic in background worker."""
        self.stop()
        with self._lock:
            self._stop_event.clear()
            self.is_running = True
            self.is_paused = False
            self.active_source = "Simulated Traffic"
            self.current_scenario = scenario
            self.start_time = time.time()

        self._worker_thread = threading.Thread(
            target=self._simulation_loop, args=(scenario, interval), daemon=True
        )
        self._worker_thread.start()

    def start_pcap_replay(self, pcap_file_or_bytes: Any, interval: float = 0.5):
        """Load and replay packets from a PCAP file."""
        self.stop()
        events = read_pcap_events(pcap_file_or_bytes)
        if not events:
            raise ValueError("No valid IP packets found in provided PCAP.")

        with self._lock:
            self._stop_event.clear()
            self._pcap_queue = deque(events)
            self.is_running = True
            self.is_paused = False
            self.active_source = "PCAP Replay"
            self.start_time = time.time()

        self._worker_thread = threading.Thread(
            target=self._pcap_replay_loop, args=(interval,), daemon=True
        )
        self._worker_thread.start()

    def start_live_capture(self, iface: Optional[str] = None):
        """
        Attempt live network capture using Scapy sniff.
        Note: Requires Npcap/WinPcap and elevated privileges on Windows.
        """
        self.stop()
        try:
            from scapy.all import sniff
        except ImportError:
            raise RuntimeError("Scapy is required for live network capture.")

        with self._lock:
            self._stop_event.clear()
            self.is_running = True
            self.is_paused = False
            self.active_source = "Live Capture"
            self.start_time = time.time()

        self._worker_thread = threading.Thread(
            target=self._live_sniff_loop, args=(iface,), daemon=True
        )
        self._worker_thread.start()

    def pause(self):
        with self._lock:
            self.is_paused = True

    def resume(self):
        with self._lock:
            self.is_paused = False

    def stop(self):
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        with self._lock:
            self.is_running = False
            self.is_paused = False
            self._pcap_queue.clear()

    def reset_buffer(self):
        with self._lock:
            self._buffer.clear()
            self.packets_collected_total = 0
            self.bytes_collected_total = 0
            self.start_time = time.time()
            self.alert_manager.clear()

    def ingest_packet(self, event: PacketEvent):
        """Add a single packet event into buffer and evaluate anomalies."""
        with self._lock:
            self._buffer.append(event)
            self.packets_collected_total += 1
            self.bytes_collected_total += event.length

    def ingest_batch(self, events: List[PacketEvent]):
        """Add a batch of packet events into buffer and evaluate anomalies."""
        if not events:
            return
        with self._lock:
            for ev in events:
                self._buffer.append(ev)
                self.packets_collected_total += 1
                self.bytes_collected_total += ev.length

        # Run anomaly detection on latest window
        recent_list = self.get_recent_events(limit=200)
        detected_anoms = detect_realtime_anomalies(recent_list, window_seconds=45.0)
        if detected_anoms:
            self.alert_manager.process_anomalies(detected_anoms)

    def _simulation_loop(self, scenario: str, interval: float):
        while not self._stop_event.is_set():
            if not self.is_paused:
                batch = generate_simulated_packets(scenario=self.current_scenario, count=random_count(scenario))
                self.ingest_batch(batch)
            time.sleep(interval)

    def _pcap_replay_loop(self, interval: float):
        while not self._stop_event.is_set() and self._pcap_queue:
            if not self.is_paused:
                batch = []
                with self._lock:
                    for _ in range(min(15, len(self._pcap_queue))):
                        if self._pcap_queue:
                            ev = self._pcap_queue.popleft()
                            # Normalize timestamp to present for live rolling visuals
                            ev.timestamp = time.time()
                            ev.time_str = datetime.now().strftime("%H:%M:%S")
                            batch.append(ev)
                self.ingest_batch(batch)
            time.sleep(interval)
        
        # When PCAP finishes replaying, mark status
        with self._lock:
            self.is_running = False

    def _live_sniff_loop(self, iface: Optional[str]):
        from scapy.all import sniff
        from src.realtime.pcap_reader import extract_packet_metadata

        def _pkt_callback(pkt):
            if self._stop_event.is_set():
                return True  # Stop sniffing
            if not self.is_paused:
                ev = extract_packet_metadata(pkt, source_type="Live Capture")
                if ev:
                    self.ingest_batch([ev])

        try:
            sniff(
                prn=_pkt_callback,
                store=False,
                stop_filter=lambda x: self._stop_event.is_set(),
                iface=iface
            )
        except Exception as exc:
            with self._lock:
                self.is_running = False
                self.active_source = f"Live Capture Error: {exc}"

    def get_recent_events(self, limit: int = 500) -> List[PacketEvent]:
        with self._lock:
            return list(self._buffer)[-limit:]

    def get_dataframe(self, limit: int = 2000) -> pd.DataFrame:
        with self._lock:
            if not self._buffer:
                return pd.DataFrame(columns=[
                    "timestamp", "time_str", "src_ip", "dst_ip", "src_port",
                    "dst_port", "protocol", "length", "tcp_flags", "info",
                    "source_type", "is_outbound"
                ])
            events = list(self._buffer)[-limit:]
            return pd.DataFrame([e.to_dict() for e in events])

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Calculate high-level traffic summary statistics."""
        with self._lock:
            events = list(self._buffer)
            total_pkts = self.packets_collected_total
            total_bytes = self.bytes_collected_total

        if not events:
            return {
                "total_packets": total_pkts,
                "total_bytes": total_bytes,
                "pps": 0.0,
                "bps": 0.0,
                "active_src_ips": 0,
                "active_dst_ips": 0,
                "active_connections": 0,
                "status": "Running" if self.is_running else ("Paused" if self.is_paused else "Stopped"),
                "source": self.active_source
            }

        now = time.time()
        window_events = [e for e in events if now - e.timestamp <= 10.0]
        window_duration = 10.0

        if window_events:
            pps = round(len(window_events) / window_duration, 1)
            bps = round(sum(e.length for e in window_events) * 8 / window_duration, 1)
        else:
            pps = 0.0
            bps = 0.0

        src_ips = {e.src_ip for e in events[-200:]}
        dst_ips = {e.dst_ip for e in events[-200:]}
        connections = {(e.src_ip, e.dst_ip) for e in events[-200:]}

        status_str = "Paused" if self.is_paused else ("Running" if self.is_running else "Stopped")

        return {
            "total_packets": total_pkts,
            "total_bytes": total_bytes,
            "pps": pps,
            "bps": bps,
            "active_src_ips": len(src_ips),
            "active_dst_ips": len(dst_ips),
            "active_connections": len(connections),
            "status": status_str,
            "source": self.active_source
        }


def random_count(scenario: str) -> int:
    s = scenario.lower()
    if s == "syn_flood":
        return 35
    elif s == "port_scan":
        return 20
    elif s == "traffic_burst":
        return 45
    return 15


# Global singleton instance
_GLOBAL_COLLECTOR: Optional[PacketCollector] = None
_COLLECTOR_LOCK = threading.Lock()


def get_packet_collector() -> PacketCollector:
    global _GLOBAL_COLLECTOR
    with _COLLECTOR_LOCK:
        if _GLOBAL_COLLECTOR is None:
            _GLOBAL_COLLECTOR = PacketCollector(buffer_size=5000)
    return _GLOBAL_COLLECTOR
