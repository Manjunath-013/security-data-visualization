"""Real-time network traffic anomaly and suspicious activity analytics.
Analyzes sliding packet windows to detect Port Scans, SYN Floods, Traffic Bursts,
Horizontal Scanning, DNS Tunneling/Anomalies, and Outbound Exfiltration patterns.
"""

from collections import defaultdict
import math
import time
from typing import Any, Dict, List
import pandas as pd

from src.realtime.event_processor import PacketEvent, is_private_ip


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    entropy = 0.0
    length = len(text)
    counts = {}
    for char in text:
        counts[char] = counts.get(char, 0) + 1
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return round(entropy, 3)


def detect_realtime_anomalies(events: List[PacketEvent], window_seconds: float = 60.0) -> List[Dict[str, Any]]:
    """
    Evaluate the recent sliding packet buffer to identify explainable suspicious activities.
    """
    if not events:
        return []

    anomalies: List[Dict[str, Any]] = []
    now = time.time()
    
    # Filter to rolling window if events have recent timestamps
    # For PCAPs or older replays, normalize by using the latest timestamp in the batch
    latest_ts = max(e.timestamp for e in events)
    window_start = latest_ts - window_seconds
    recent = [e for e in events if e.timestamp >= window_start]
    if not recent:
        recent = events[-100:]  # Fallback to last 100 packets

    source_type = recent[0].source_type if recent else "Network Data"

    # 1. Port Scan Detection (Source IP contacting >= 8 distinct destination ports)
    src_ports_map = defaultdict(set)
    src_targets_map = defaultdict(set)
    for e in recent:
        if e.dst_port is not None:
            src_ports_map[e.src_ip].add(e.dst_port)
            src_targets_map[e.src_ip].add(e.dst_ip)

    for src_ip, ports in src_ports_map.items():
        if len(ports) >= 8:
            target_hosts = list(src_targets_map[src_ip])
            target_display = target_hosts[0] if len(target_hosts) == 1 else f"{len(target_hosts)} hosts"
            anomalies.append({
                "category": "Port Scan",
                "severity": "High" if len(ports) < 18 else "Critical",
                "src_ip": src_ip,
                "dst_ip": target_display,
                "summary": f"Vertical Port Scan Detected from {src_ip}",
                "explanation": (
                    f"Host {src_ip} probed {len(ports)} distinct destination ports within {window_seconds:.0f}s. "
                    f"This rapid port enumeration pattern is characteristic of reconnaissance scanning."
                ),
                "evidence": f"{len(ports)} unique ports scanned (e.g., {', '.join(str(p) for p in sorted(list(ports))[:5])}...)",
                "data_source": source_type,
                "timestamp": latest_ts
            })

    # 2. Horizontal Subnet Scanning (Source IP contacting >= 6 distinct hosts on the same port)
    port_targets_map = defaultdict(lambda: defaultdict(set))
    for e in recent:
        if e.dst_port is not None:
            port_targets_map[e.src_ip][e.dst_port].add(e.dst_ip)

    for src_ip, p_map in port_targets_map.items():
        for port, targets in p_map.items():
            if len(targets) >= 6:
                anomalies.append({
                    "category": "Horizontal Scan",
                    "severity": "High",
                    "src_ip": src_ip,
                    "dst_ip": f"{len(targets)} Subnet Hosts",
                    "summary": f"Horizontal Subnet Probe on Port {port}",
                    "explanation": (
                        f"Host {src_ip} reached out to {len(targets)} separate destination IP addresses "
                        f"on port {port} ({e.protocol}). Indicates automated service sweeping."
                    ),
                    "evidence": f"Contacted {len(targets)} destination IPs on single port {port}",
                    "data_source": source_type,
                    "timestamp": latest_ts
                })

    # 3. SYN Flood Indicators (High ratio of SYN packets without ACK completion)
    tcp_pkts = [e for e in recent if e.protocol in {"TCP", "HTTPS", "HTTP"}]
    if len(tcp_pkts) >= 15:
        syn_only = [e for e in tcp_pkts if "SYN" in e.tcp_flags and "ACK" not in e.tcp_flags]
        syn_ratio = len(syn_only) / max(1, len(tcp_pkts))
        if syn_ratio >= 0.65 and len(syn_only) >= 12:
            target_counts = defaultdict(int)
            for e in syn_only:
                target_counts[e.dst_ip] += 1
            top_target = max(target_counts.items(), key=lambda x: x[1])[0]

            anomalies.append({
                "category": "SYN Flood",
                "severity": "Critical",
                "src_ip": "Multiple / Distributed Sources",
                "dst_ip": top_target,
                "summary": f"TCP SYN Flood Burst Targeting {top_target}",
                "explanation": (
                    f"Observed an abnormal concentration of {len(syn_only)} unacknowledged TCP SYN packets "
                    f"targeting {top_target} ({syn_ratio*100:.1f}% of observed TCP traffic). "
                    f"Note: High SYN rates indicate potential denial-of-service attempts but do not verify complete service outage."
                ),
                "evidence": f"{len(syn_only)} SYN packets with {syn_ratio*100:.1f}% SYN ratio",
                "data_source": source_type,
                "timestamp": latest_ts
            })

    # 4. DNS Tunneling & Query Anomaly Detection
    dns_pkts = [e for e in recent if e.protocol == "DNS" and "Query:" in e.info]
    for d in dns_pkts:
        query_text = d.info.replace("Query:", "").strip()
        subdomain = query_text.split(".")[0] if "." in query_text else query_text
        entropy = _shannon_entropy(subdomain)
        if len(query_text) >= 32 or entropy >= 4.0:
            anomalies.append({
                "category": "DNS Anomaly",
                "severity": "Medium",
                "src_ip": d.src_ip,
                "dst_ip": d.dst_ip,
                "summary": "Suspicious DNS Tunneling / High-Entropy Query",
                "explanation": (
                    f"High character entropy ({entropy:.2f}) or excessive length ({len(query_text)} chars) "
                    f"detected in DNS query: '{query_text}'. Common indicator of C2 data tunneling or DGA."
                ),
                "evidence": f"Query: {query_text} (Entropy: {entropy:.2f}, Length: {len(query_text)})",
                "data_source": source_type,
                "timestamp": d.timestamp
            })
            break  # Prevent duplicate alert flood per window

    # 5. Outbound High-Volume Transfer / Exfiltration Anomaly
    outbound_pkts = [e for e in recent if e.is_outbound]
    if outbound_pkts:
        out_bytes_by_pair = defaultdict(int)
        for e in outbound_pkts:
            out_bytes_by_pair[(e.src_ip, e.dst_ip)] += e.length

        for (src_ip, dst_ip), total_bytes in out_bytes_by_pair.items():
            if total_bytes >= 25000:  # 25KB+ in short window
                anomalies.append({
                    "category": "Outbound Spike",
                    "severity": "Medium",
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "summary": f"Unusually Large Outbound Data Transfer ({total_bytes/1024:.1f} KB)",
                    "explanation": (
                        f"Host {src_ip} transmitted {total_bytes/1024:.1f} KB to external endpoint {dst_ip}. "
                        f"Identified as a statistical outbound volume surge; represents potential exfiltration risk."
                    ),
                    "evidence": f"{total_bytes:,} bytes transferred to external host {dst_ip}",
                    "data_source": source_type,
                    "timestamp": latest_ts
                })

    # 6. Overall Traffic Rate Burst
    if len(recent) >= 40:
        duration = max(1.0, latest_ts - min(e.timestamp for e in recent))
        pps = len(recent) / duration
        if pps >= 35.0:
            anomalies.append({
                "category": "Traffic Burst",
                "severity": "Medium",
                "src_ip": "Network Aggregate",
                "dst_ip": "Multiple Hosts",
                "summary": f"Traffic Rate Spike ({pps:.1f} pkts/sec)",
                "explanation": f"Packet arrival rate spiked to {pps:.1f} pkts/sec across the recent observation window.",
                "evidence": f"{len(recent)} packets captured over {duration:.1f} seconds",
                "data_source": source_type,
                "timestamp": latest_ts
            })

    return anomalies
