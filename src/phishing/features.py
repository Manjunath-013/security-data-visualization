"""URL feature extraction module for phishing detection.
Extracts lexical, structural, and statistical features without visiting or executing untrusted URLs.
"""

import math
import re
from typing import Any, Dict, List
from urllib.parse import urlsplit

SUSPICIOUS_KEYWORDS: List[str] = [
    "login", "signin", "verify", "verification", "account", "update", "banking",
    "secure", "security", "webscr", "ebayisapi", "confirm", "password",
    "authenticate", "authentication", "wallet", "support", "recovery", "service",
    "validation", "billing", "invoice", "authorize", "token", "credential"
]

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "is.gd", "ow.ly", "buff.ly", "rebrand.ly",
    "cutt.ly", "tiny.cc", "goo.gl", "qr.ae", "shorte.st", "adf.ly", "lnkd.in"
}

FEATURE_NAMES: List[str] = [
    "url_length",
    "domain_length",
    "path_length",
    "query_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_questionmarks",
    "num_equals",
    "num_at",
    "num_subdomains",
    "is_ip",
    "is_https",
    "has_port",
    "num_digits",
    "digit_ratio",
    "suspicious_keyword_count",
    "has_punycode",
    "is_shortened",
    "entropy"
]


def _shannon_entropy(text: str) -> float:
    """Calculate Shannon entropy of a string."""
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
    return round(entropy, 4)


def _is_ip_address(domain: str) -> int:
    """Check if domain is an IPv4 or IPv6 address."""
    # Strip port if present
    host = domain.split(":")[0].strip("[]")
    # IPv4 regex
    ipv4_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"
    if re.match(ipv4_pattern, host):
        parts = host.split(".")
        if all(0 <= int(p) <= 255 for p in parts if p.isdigit()):
            return 1
    # Basic IPv6 check
    if ":" in host and len(host.split(":")) >= 3:
        return 1
    return 0


def sanitize_url(raw_url: str) -> str:
    """Sanitize and format raw user URL input safely."""
    if not raw_url:
        return ""
    url = str(raw_url).strip()
    if not url:
        return ""
    # Normalize scheme if missing
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", url):
        url = "https://" + url
    return url


def extract_url_features(raw_url: str) -> Dict[str, Any]:
    """
    Extract reproducible lexical, structural, and security features from a URL.
    Returns a dictionary mapping feature names to numerical values.
    """
    url = sanitize_url(raw_url)
    if not url:
        return {k: 0 for k in FEATURE_NAMES}

    try:
        parsed = urlsplit(url)
    except Exception:
        # Fallback for ill-formed URL structures
        parsed = None

    scheme = parsed.scheme.lower() if parsed else ""
    netloc = parsed.netloc.lower() if parsed else ""
    path = parsed.path if parsed else ""
    query = parsed.query if parsed else ""

    # Clean domain (strip auth credentials or custom port for length calculations)
    domain_only = netloc.split("@")[-1].split(":")[0]

    # Calculate basic lengths
    url_len = len(url)
    dom_len = len(domain_only)
    path_len = len(path)
    query_len = len(query)

    # Character counts
    num_dots = url.count(".")
    num_hyphens = url.count("-")
    num_underscores = url.count("_")
    num_slashes = url.count("/")
    num_questionmarks = url.count("?")
    num_equals = url.count("=")
    num_at = url.count("@")

    # Subdomain calculation
    parts = domain_only.split(".")
    # Standard domain with TLD has len 2 (e.g., example.com)
    # 3 parts -> 1 subdomain, etc.
    num_subdomains = max(0, len(parts) - 2) if len(parts) >= 2 else 0

    # IP detection
    is_ip = _is_ip_address(domain_only)

    # HTTPS flag
    is_https = 1 if scheme == "https" else 0

    # Custom port
    has_port = 1 if (":" in netloc and not netloc.startswith("[")) else 0

    # Digits and ratios
    num_digits = sum(c.isdigit() for c in url)
    digit_ratio = round(num_digits / max(1, url_len), 4)

    # Keywords in URL path/query/subdomain (lowercased)
    url_lower = url.lower()
    keyword_hits = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)

    # Homograph / Punycode check
    has_punycode = 1 if "xn--" in domain_only else 0

    # Known shortener
    is_shortened = 1 if domain_only in SHORTENER_DOMAINS else 0

    # Entropy
    entropy = _shannon_entropy(url)

    return {
        "url_length": url_len,
        "domain_length": dom_len,
        "path_length": path_len,
        "query_length": query_len,
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_underscores": num_underscores,
        "num_slashes": num_slashes,
        "num_questionmarks": num_questionmarks,
        "num_equals": num_equals,
        "num_at": num_at,
        "num_subdomains": num_subdomains,
        "is_ip": is_ip,
        "is_https": is_https,
        "has_port": has_port,
        "num_digits": num_digits,
        "digit_ratio": digit_ratio,
        "suspicious_keyword_count": keyword_hits,
        "has_punycode": has_punycode,
        "is_shortened": is_shortened,
        "entropy": entropy
    }
