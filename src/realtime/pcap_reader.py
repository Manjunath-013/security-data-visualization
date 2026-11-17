"""PCAP file reader, packet metadata extractor, and sample PCAP generator.
"""

from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict, Generator, List, Optional

from scapy.all import (
    IP, IPv6, TCP, UDP, ICMP, DNS, DNSQR, Raw,
    rdpcap, wrpcap, Ether
)

from src.realtime.event_processor import (
    PacketEvent, is_private_ip, resolve_protocol
)


def extract_packet_metadata(pkt: Any, source_type: str = "PCAP Replay") -> Optional[PacketEvent]:
    """Safely extract metadata attributes from a Scapy packet object."""
    if not pkt.haslayer(IP) and not pkt.haslayer(IPv6):
        return None

    is_ipv4 = pkt.haslayer(IP)
    ip_layer = pkt[IP] if is_ipv4 else pkt[IPv6]
    src_ip = str(ip_layer.src)
    dst_ip = str(ip_layer.dst)

    pkt_time = float(pkt.time) if hasattr(pkt, "time") else time.time()
    dt_str = datetime.fromtimestamp(pkt_time).strftime("%H:%M:%S")

    src_port = None
    dst_port = None
    tcp_flags = ""
    proto_name = "IP"
    info = ""

    if pkt.haslayer(TCP):
        tcp_layer = pkt[TCP]
        src_port = int(tcp_layer.sport)
        dst_port = int(tcp_layer.dport)
        proto_name = "TCP"
        # Extract flags
        flags = []
        f_val = str(tcp_layer.flags)
        if "S" in f_val: flags.append("SYN")
        if "A" in f_val: flags.append("ACK")
        if "F" in f_val: flags.append("FIN")
        if "R" in f_val: flags.append("RST")
        if "P" in f_val: flags.append("PSH")
        tcp_flags = " ".join(flags) if flags else f_val
        info = f"Seq={tcp_layer.seq} Flags=[{tcp_flags}]"

    elif pkt.haslayer(UDP):
        udp_layer = pkt[UDP]
        src_port = int(udp_layer.sport)
        dst_port = int(udp_layer.dport)
        proto_name = "UDP"
        info = f"Len={udp_layer.len}"

    elif pkt.haslayer(ICMP):
        proto_name = "ICMP"
        info = f"Type={pkt[ICMP].type} Code={pkt[ICMP].code}"

    # Application protocol identification
    resolved_proto = resolve_protocol(proto_name, src_port, dst_port)

    # DNS query details
    if pkt.haslayer(DNS) and pkt.haslayer(DNSQR):
        resolved_proto = "DNS"
        try:
            qname = pkt[DNSQR].qname.decode(errors="ignore").rstrip(".")
            info = f"Query: {qname}"
        except Exception:
            pass

    # Outbound determination: src is private and dst is public
    is_outbound = is_private_ip(src_ip) and not is_private_ip(dst_ip)

    return PacketEvent(
        timestamp=pkt_time,
        time_str=dt_str,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        protocol=resolved_proto,
        length=len(pkt),
        tcp_flags=tcp_flags,
        info=info,
        source_type=source_type,
        is_outbound=is_outbound
    )


def read_pcap_events(file_path_or_bytes: Any, max_packets: int = 5000) -> List[PacketEvent]:
    """Read a PCAP file and return a list of normalized PacketEvents."""
    events: List[PacketEvent] = []
    
    # Handle Path, str, or raw bytes / BytesIO
    if isinstance(file_path_or_bytes, Path):
        target = str(file_path_or_bytes)
    elif isinstance(file_path_or_bytes, bytes):
        import io
        target = io.BytesIO(file_path_or_bytes)
    elif hasattr(file_path_or_bytes, "read"):
        target = file_path_or_bytes
    else:
        target = str(file_path_or_bytes)

    try:
        pkts = rdpcap(target)
    except Exception as exc:
        raise ValueError(f"Could not parse PCAP: {exc}")

    for i, pkt in enumerate(pkts):
        if i >= max_packets:
            break
        ev = extract_packet_metadata(pkt, source_type="PCAP Replay")
        if ev:
            events.append(ev)
    return events


def create_sample_pcap(output_path: Path):
    """
    Generate a bundled sample PCAP with normal traffic, port scans, SYN floods, and DNS queries
    so the user can immediately test PCAP replay without requiring an external file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pkts = []
    base_time = time.time() - 120

    # 1. Normal DNS and HTTPS traffic
    # DNS Query to 8.8.8.8
    p1 = Ether() / IP(src="192.168.1.105", dst="8.8.8.8") / UDP(sport=54210, dport=53) / DNS(rd=1, qd=DNSQR(qname="www.google.com"))
    p1.time = base_time + 0.1
    pkts.append(p1)

    # HTTPS Handshake to 151.101.1.140
    p2 = Ether() / IP(src="192.168.1.105", dst="151.101.1.140") / TCP(sport=51234, dport=443, flags="S", seq=1000)
    p2.time = base_time + 0.2
    pkts.append(p2)

    p3 = Ether() / IP(src="151.101.1.140", dst="192.168.1.105") / TCP(sport=443, dport=51234, flags="SA", seq=5000, ack=1001)
    p3.time = base_time + 0.25
    pkts.append(p3)

    p4 = Ether() / IP(src="192.168.1.105", dst="151.101.1.140") / TCP(sport=51234, dport=443, flags="A", seq=1001, ack=5001)
    p4.time = base_time + 0.3
    pkts.append(p4)

    # 2. Suspicious Port Scan from external host 45.154.255.88 against 192.168.1.50
    target_ports = [21, 22, 23, 25, 53, 80, 110, 139, 143, 443, 445, 1433, 1521, 3306, 3389, 5432, 8080, 8443, 9200, 27017]
    for i, port in enumerate(target_ports):
        scan_pkt = Ether() / IP(src="45.154.255.88", dst="192.168.1.50") / TCP(sport=40000 + i, dport=port, flags="S", seq=2000 + i)
        scan_pkt.time = base_time + 5.0 + (i * 0.05)
        pkts.append(scan_pkt)

    # 3. SYN Flood against Web Server 192.168.1.10:443
    for i in range(25):
        syn_pkt = Ether() / IP(src="194.26.29.112", dst="192.168.1.10") / TCP(sport=30000 + (i * 7) % 30000, dport=443, flags="S", seq=30000 + i)
        syn_pkt.time = base_time + 15.0 + (i * 0.04)
        pkts.append(syn_pkt)

    # 4. DNS Exfiltration / Anomaly
    for i in range(5):
        hex_data = f"token{i:02d}e4b7c89f2a0134d.threat-c2.cc"
        dns_exfil = Ether() / IP(src="192.168.1.200", dst="9.9.9.9") / UDP(sport=59000 + i, dport=53) / DNS(rd=1, qd=DNSQR(qname=hex_data))
        dns_exfil.time = base_time + 25.0 + (i * 0.2)
        pkts.append(dns_exfil)

    # 5. ICMP Ping sweep
    for i in range(1, 6):
        icmp_pkt = Ether() / IP(src="192.168.1.105", dst=f"192.168.1.{i}") / ICMP(type=8, code=0)
        icmp_pkt.time = base_time + 30.0 + (i * 0.1)
        pkts.append(icmp_pkt)

    wrpcap(str(output_path), pkts)
    return len(pkts)
