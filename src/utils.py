"""
utils.py — Shared utility helpers used across src/ and mcp/
"""

import re
from datetime import datetime

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def is_valid_date(s: str) -> bool:
    """Return True only if s is a strict YYYY-MM-DD calendar date."""
    if not _DATE_RE.match(s):
        return False
    try:
        datetime.fromisoformat(s)
        return True
    except ValueError:
        return False
