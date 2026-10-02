"""PII / secret filtering for training corpora (audit P0, doc section 16).

Public availability does not mean training suitability: web and code
dumps contain emails, phones, keys, and credentials. `scrub_doc` masks
matches with a typed placeholder and reports per-class hit counts so
runs can audit what was removed.
"""
import re

PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d[\d .()-]{7,}\d)(?!\d)"),
    "ipv4": re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[^-]*-----END [A-Z ]*PRIVATE KEY-----", re.S),
    "aws_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    "api_assignment": re.compile(r"(?i)(api[_-]?key|secret|password|passwd|pwd|bearer)\s*[:=]\s*['\"]?\S+['\"]?"),
    "generic_secret": re.compile(r"(?i)(token|credential)s?\s*[:=]\s*['\"]?\S{8,}['\"]?"),
}


def scrub_doc(text: str) -> tuple[str, dict[str, int]]:
    """Mask PII/secrets. Returns (scrubbed, {class: hits})."""
    hits: dict[str, int] = {}
    for name, pat in PATTERNS.items():
        text, n = pat.subn(f"<|{name}|>", text)
        if n:
            hits[name] = n
    return text, hits


def scrub_docs(docs: list[str]) -> tuple[list[str], dict[str, int]]:
    """Scrub many docs; aggregate hit counts for the run audit."""
    total: dict[str, int] = {}
    out = []
    for doc in docs:
        clean, hits = scrub_doc(doc)
        out.append(clean)
        for k, v in hits.items():
            total[k] = total.get(k, 0) + v
    return out, total
