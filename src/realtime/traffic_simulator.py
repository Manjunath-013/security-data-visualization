"""Deterministic Traffic Simulator for Network Security Demonstrations.
Generates realistic baseline enterprise network traffic alongside controlled
suspicious scenarios (Port Scan, SYN Flood, DNS Tunneling, Horizontal Scan, Exfiltration).
"""

from datetime import datetime
import random
import time
from typing import List

from src.realtime.event_processor import PacketEvent, is_private_ip, resolve_protocol


INTERNAL_HOSTS = [
    "192.168.1.10", "192.168.1.15", "192.168.1.22", "192.168.1.45",
    "192.168.1.102", "192.168.1.105", "192.168.1.150", "192.168.1.200"
]

EXTERNAL_SERVERS = [
    "1.1.1.1", "8.8.8.8", "9.9.9.9", "13.107.4.52", "151.101.1.140",
    "208.67.222.222", "198.51.100.44", "45.154.255.88", "194.26.29.112"
]

COMMON_DOMAINS = [
    "google.com", "github.com", "microsoft.com", "cloudflare.com",
    "wikipedia.org", "internal.corp.local", "api.github.com", "aws.amazon.com"
]


def generate_simulated_packets(scenario: str = "normal", count: int = 25) -> List[PacketEvent]:
    """
    Generate a batch of simulated PacketEvents according to the chosen scenario.
    Always labeled with source_type='Simulated Traffic'.
    """
    now = time.time()
    events: List[PacketEvent] = []

    scenario_clean = scenario.lower().replace(" ", "_")

    if scenario_clean == "port_scan":
        # Rapid SYN scanning of diverse destination ports from external scanner 45.154.255.88
        target_ip = "192.168.1.10"
        scanner_ip = "45.154.255.88"
        scan_ports = [21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 445, 1433, 1521, 3306, 3389, 5432, 8080, 8443, 9000, 9200, 27017]
        for i in range(count):
            p_time = now - (count - i) * 0.05
            port = random.choice(scan_ports)
            ev = PacketEvent(
                timestamp=p_time,
                time_str=datetime.fromtimestamp(p_time).strftime("%H:%M:%S"),
                src_ip=scanner_ip,
                dst_ip=target_ip,
                src_port=random.randint(40000, 65000),
                dst_port=port,
                protocol="TCP",
                length=random.randint(54, 66),
                tcp_flags="SYN",
                info=f"Port Scan probe to port {port}",
                source_type="Simulated Traffic",
                is_outbound=False
            )
            events.append(ev)

    elif scenario_clean == "syn_flood":
        # SYN Flood barrage targeted at Web Server 192.168.1.10:443
        target_ip = "192.168.1.10"
        for i in range(count):
            p_time = now - (count - i) * 0.02
            spoofed_src = f"{random.randint(11, 210)}.{random.randint(1, 250)}.{random.randint(1, 250)}.{random.randint(1, 250)}"
            ev = PacketEvent(
                timestamp=p_time,
                time_str=datetime.fromtimestamp(p_time).strftime("%H:%M:%S"),
                src_ip=spoofed_src,
                dst_ip=target_ip,
                src_port=random.randint(1024, 65535),
                dst_port=443,
                protocol="HTTPS",
                length=random.randint(54, 74),
                tcp_flags="SYN",
                info="High-rate TCP SYN flood request",
                source_type="Simulated Traffic",
                is_outbound=False
            )
            events.append(ev)

    elif scenario_clean == "dns_anomaly":
        # DNS Tunneling / Query Flood with randomized encoded subdomains
        src_host = "192.168.1.150"
        dns_server = "9.9.9.9"
        for i in range(count):
            p_time = now - (count - i) * 0.08
            encoded_payload = "".join(random.choices("abcdef0123456789", k=24))
            domain = f"{encoded_payload}.data-sync.c2-network.cc"
            ev = PacketEvent(
                timestamp=p_time,
                time_str=datetime.fromtimestamp(p_time).strftime("%H:%M:%S"),
                src_ip=src_host,
                dst_ip=dns_server,
                src_port=random.randint(50000, 60000),
                dst_port=53,
                protocol="DNS",
                length=random.randint(110, 240),
                tcp_flags="",
                info=f"Query: {domain}",
                source_type="Simulated Traffic",
                is_outbound=True
            )
            events.append(ev)

    elif scenario_clean == "horizontal_scan":
        # Scanning the same port (445 SMB or 22 SSH) across multiple internal hosts
        scanner_ip = "192.168.1.200"
        port = 445
        for i in range(count):
            p_time = now - (count - i) * 0.06
            target_ip = f"192.168.1.{(i % 40) + 1}"
            ev = PacketEvent(
                timestamp=p_time,
                time_str=datetime.fromtimestamp(p_time).strftime("%H:%M:%S"),
                src_ip=scanner_ip,
                dst_ip=target_ip,
                src_port=random.randint(45000, 60000),
                dst_port=port,
                protocol="SMB",
                length=random.randint(54, 66),
                tcp_flags="SYN",
                info=f"Horizontal probe on port {port}",
                source_type="Simulated Traffic",
                is_outbound=False
            )
            events.append(ev)

    elif scenario_clean == "outbound_exfil":
        # Massive outbound transfer from internal database to suspicious external IP
        src_ip = "192.168.1.15"
        dst_ip = "194.26.29.112"
        for i in range(count):
            p_time = now - (count - i) * 0.04
            ev = PacketEvent(
                timestamp=p_time,
                time_str=datetime.fromtimestamp(p_time).strftime("%H:%M:%S"),
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=58912,
                dst_port=8443,
                protocol="HTTPS",
                length=random.randint(1200, 1514),
                tcp_flags="PSH ACK",
                info=f"High-volume outbound payload segment ({i+1}/{count})",
                source_type="Simulated Traffic",
                is_outbound=True
            )
            events.append(ev)

    else:
        # Normal Enterprise Traffic Mix (HTTP, HTTPS, DNS, SSH, NTP)
        for i in range(count):
            p_time = now - (count - i) * (1.0 / max(1, count))
            traffic_type = random.choice(["https", "http", "dns", "ssh", "internal_smb"])
            src_ip = random.choice(INTERNAL_HOSTS)

            if traffic_type == "https":
                dst_ip = random.choice(EXTERNAL_SERVERS)
                sport = random.randint(40000, 65000)
                dport = 443
                proto = "HTTPS"
                flags = random.choice(["SYN ACK", "ACK", "PSH ACK", "FIN ACK"])
                info = f"TLS Application Data {random.randint(300, 1400)} bytes"
                length = random.randint(200, 1480)
            elif traffic_type == "http":
                dst_ip = random.choice(EXTERNAL_SERVERS)
                sport = random.randint(40000, 65000)
                dport = 80
                proto = "HTTP"
                flags = "ACK"
                info = "GET /index.html HTTP/1.1"
                length = random.randint(150, 800)
            elif traffic_type == "dns":
                dst_ip = random.choice(["1.1.1.1", "8.8.8.8", "9.9.9.9"])
                sport = random.randint(50000, 65000)
                dport = 53
                proto = "DNS"
                flags = ""
                qname = random.choice(COMMON_DOMAINS)
                info = f"Standard Query A {qname}"
                length = random.randint(68, 120)
            elif traffic_type == "ssh":
                dst_ip = "192.168.1.10"
                sport = random.randint(45000, 55000)
                dport = 22
                proto = "SSH"
                flags = "PSH ACK"
                info = "Encrypted SSHv2 session communication"
                length = random.randint(80, 250)
            else:
                dst_ip = random.choice(INTERNAL_HOSTS)
                sport = random.randint(49152, 65000)
                dport = 445
                proto = "SMB"
                flags = "ACK"
                info = "SMB2 Tree Connect Response"
                length = random.randint(110, 500)

            ev = PacketEvent(
                timestamp=p_time,
                time_str=datetime.fromtimestamp(p_time).strftime("%H:%M:%S"),
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=sport,
                dst_port=dport,
                protocol=proto,
                length=length,
                tcp_flags=flags,
                info=info,
                source_type="Simulated Traffic",
                is_outbound=is_private_ip(src_ip) and not is_private_ip(dst_ip)
            )
            events.append(ev)

    return events
