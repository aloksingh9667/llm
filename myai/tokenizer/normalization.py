"""Text normalization for the tokenizer training corpus (Step 3a).

English-first rules, stdlib only:
  1. Unicode normalize (default NFKC)
  2. Drop control characters except newline/tab
  3. Collapse horizontal whitespace, cap consecutive newlines
"""
import re
import unicodedata

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HSPACE_RE = re.compile(r"[ \t\u00a0\u2000-\u200b\u202f\u205f\u3000]+")


def normalize_text(
    text: str,
    unicode_form: str = "NFKC",
    collapse_whitespace: bool = True,
    strip_controls: bool = True,
    max_newlines: int = 2,
) -> str:
    """Normalize one document. Deterministic: same input -> same output."""
    if unicode_form:
        text = unicodedata.normalize(unicode_form, text)
    if strip_controls:
        text = _CONTROL_RE.sub("", text)
    if collapse_whitespace:
        # Collapse runs of horizontal whitespace *inside* each line but
        # preserve leading indentation (matters for code).
        lines = []
        for ln in text.split("\n"):
            stripped = ln.lstrip(" \t\u00a0\u2000-\u200b\u202f\u205f\u3000")
            leading = ln[: len(ln) - len(stripped)]
            rest = _HSPACE_RE.sub(" ", stripped).rstrip()
            lines.append(leading + rest)
        text = "\n".join(lines)
        if max_newlines >= 0:
            text = re.sub(r"\n{3,}", "\n" * max(1, max_newlines), text)
        text = text.strip("\n")
    return text
