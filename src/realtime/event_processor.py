"""Real-time network event normalization, protocol resolution, and IP classification.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
import ipaddress
import re
from typing import Any, Dict, Optional, Tuple


@dataclass
class PacketEvent:
    timestamp: float
    time_str: str
    src_ip: str
    dst_ip: str
    src_port: Optional[int]
    dst_port: Optional[int]
    protocol: str
    length: int
    tcp_flags: str
    info: str
    source_type: str
    is_outbound: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


WELL_KNOWN_PORTS = {
    20: "FTP-Data",
    21: "FTP",
    22: "SSH",
    23: "Telnet",
    25: "SMTP",
    53: "DNS",
    67: "DHCP",
    68: "DHCP",
    80: "HTTP",
    110: "POP3",
    123: "NTP",
    137: "NetBIOS",
    139: "NetBIOS",
    143: "IMAP",
    161: "SNMP",
    389: "LDAP",
    443: "HTTPS",
    445: "SMB",
    1433: "MSSQL",
    1521: "Oracle",
    3306: "MySQL",
    3389: "RDP",
    5432: "PostgreSQL",
    8080: "HTTP-Alt",
    8443: "HTTPS-Alt"
}

# Representative approximate public geolocation reference for demonstrations
# Maps well-known public networks or sample prefixes to approximate locations
KNOWN_GEOLOCATIONS = {
    "1.1.1.1": ("Cloudflare DNS", "Australia / Global Anycast", -33.8688, 151.2093),
    "1.0.0.1": ("Cloudflare DNS Secondary", "Australia / Global Anycast", -33.8688, 151.2093),
    "8.8.8.8": ("Google Public DNS", "United States", 37.4056, -122.0775),
    "8.8.4.4": ("Google DNS Secondary", "United States", 37.4056, -122.0775),
    "9.9.9.9": ("Quad9 DNS", "Switzerland", 47.3769, 8.5417),
    "208.67.222.222": ("OpenDNS Cisco", "United States", 37.7749, -122.4194),
    "13.107.4.52": ("Microsoft Azure", "United States", 47.6062, -122.3321),
    "151.101.1.140": ("Fastly CDN", "United States", 37.7749, -122.4194),
    "185.220.101.5": ("Tor Exit Node / Threat Feed", "Germany", 52.5200, 13.4050),
    "194.26.29.112": ("Suspected C2 Infrastructure", "Netherlands", 52.3676, 4.9041),
    "45.154.255.88": ("External Scanning Host", "Russia", 55.7558, 37.6173),
    "103.251.167.20": ("External Scanning Host", "India", 28.6139, 77.2090),
    "180.76.76.76": ("Baidu DNS", "China", 39.9042, 116.4074),
    "198.51.100.44": ("Simulated External Server", "United States", 38.9072, -77.0369)
}


def is_private_ip(ip_str: str) -> bool:
    """Check if an IP string is within RFC 1918 private / loopback / link-local scope."""
    try:
        ip = ipaddress.ip_address(ip_str.strip())
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
    except ValueError:
        return False


def resolve_protocol(proto_num_or_str: Any, src_port: Optional[int], dst_port: Optional[int]) -> str:
    """Resolve standard application or transport protocol label."""
    p_str = str(proto_num_or_str).upper()
    
    ports = [p for p in (src_port, dst_port) if p is not None]
    
    if 53 in ports:
        return "DNS"
    if 443 in ports or 8443 in ports:
        return "HTTPS"
    if 80 in ports or 8080 in ports:
        return "HTTP"
    if 22 in ports:
        return "SSH"
    if 445 in ports or 139 in ports:
        return "SMB"
    if 3389 in ports:
        return "RDP"
    if 123 in ports:
        return "NTP"
    
    if "ICMP" in p_str or p_str == "1":
        return "ICMP"
    if "TCP" in p_str or p_str == "6":
        return "TCP"
    if "UDP" in p_str or p_str == "17":
        return "UDP"
    
    return p_str if p_str else "OTHER"


def get_ip_geolocation(ip_str: str) -> Dict[str, Any]:
    """
    Get approximate geolocation for public IP addresses.
    Private/RFC1918 addresses are explicitly marked as Local/Internal.
    """
    clean_ip = ip_str.strip()
    if is_private_ip(clean_ip):
        return {
            "ip": clean_ip,
            "is_private": True,
            "entity": "Internal LAN / Private Network",
            "country": "Local / RFC 1918",
            "lat": None,
            "lon": None
        }
    
    if clean_ip in KNOWN_GEOLOCATIONS:
        entity, country, lat, lon = KNOWN_GEOLOCATIONS[clean_ip]
        return {
            "ip": clean_ip,
            "is_private": False,
            "entity": entity,
            "country": country,
            "lat": lat,
            "lon": lon
        }
    
    # Generic public IP fallback using deterministic hash for demo visual positioning
    # so external public IPs map consistently without making external API calls
    h = hash(clean_ip) % 1000
    regions = [
        ("United States", 37.0902, -95.7129),
        ("Germany", 51.1657, 10.4515),
        ("United Kingdom", 55.3781, -3.4360),
        ("Netherlands", 52.1326, 5.2913),
        ("Japan", 36.2048, 138.2529),
        ("Singapore", 1.3521, 103.8198),
        ("Australia", -25.2744, 133.7751)
    ]
    country, lat, lon = regions[h % len(regions)]
    # Add minor deterministic jitter so markers don't overlap exactly
    lat += ((h % 17) - 8) * 0.25
    lon += (((h // 17) % 17) - 8) * 0.25
    
    return {
        "ip": clean_ip,
        "is_private": False,
        "entity": f"Public AS{10000 + (h * 47) % 50000}",
        "country": country,
        "lat": round(lat, 4),
        "lon": round(lon, 4)
    }
