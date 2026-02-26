"""tests/test_security.py — Security-focused tests for the MCP server

These tests verify the security logic (date validation, path traversal prevention).
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import is_valid_date


# ---------------------------------------------------------------------------
# Replicate only the path-confinement logic from mcp/server.py _load_json.
# Importing mcp.server directly causes a namespace conflict with the installed
# `mcp` package, so we test the guard logic in isolation here.
# ---------------------------------------------------------------------------
def _load_json_safe(data_dir: Path, rel_path: str) -> dict | None:
    """Mirror of the path-traversal guard in mcp/server.py _load_json."""
    local = data_dir / rel_path
    try:
        resolved = local.resolve()
        data_root = data_dir.resolve()
        resolved.relative_to(data_root)
    except ValueError:
        return None
    if resolved.exists():
        with open(resolved, encoding="utf-8") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Date validation tests
# ---------------------------------------------------------------------------
class TestIsValidDate:
    def test_valid_date(self):
        assert is_valid_date("2024-01-15") is True

    def test_valid_date_boundary(self):
        assert is_valid_date("2015-01-01") is True
        assert is_valid_date("2099-12-31") is True

    def test_invalid_date_traversal(self):
        """Path traversal sequences must be rejected."""
        assert is_valid_date("../../etc/passwd") is False
        assert is_valid_date("../secret") is False
        assert is_valid_date("2024-01-01/../../etc") is False

    def test_invalid_date_format(self):
        assert is_valid_date("20240115") is False
        assert is_valid_date("2024/01/15") is False
        assert is_valid_date("not-a-date") is False
        assert is_valid_date("") is False

    def test_invalid_calendar_date(self):
        """Structurally valid but impossible dates must be rejected."""
        assert is_valid_date("2024-13-01") is False
        assert is_valid_date("2024-02-30") is False

    def test_invalid_date_with_extra_chars(self):
        assert is_valid_date("2024-01-15 extra") is False
        assert is_valid_date(" 2024-01-15") is False


# ---------------------------------------------------------------------------
# Path traversal prevention tests
# ---------------------------------------------------------------------------
class TestLoadJsonPathTraversal:
    def test_traversal_blocked(self, tmp_path):
        """Traversal paths must not leak files outside data_dir."""
        secret = tmp_path.parent / "secret.json"
        secret.write_text('{"secret": "value"}', encoding="utf-8")

        result = _load_json_safe(tmp_path, "../secret.json")
        assert result is None

    def test_deep_traversal_blocked(self, tmp_path):
        """Deep traversal sequences are also blocked."""
        secret = tmp_path.parent / "secret.json"
        secret.write_text('{"secret": "value"}', encoding="utf-8")

        result = _load_json_safe(tmp_path, "history/../../secret.json")
        assert result is None

    def test_valid_path_allowed(self, tmp_path):
        """Valid relative paths within data_dir must be readable."""
        legit = tmp_path / "latest.json"
        legit.write_text('{"date": "2024-01-15"}', encoding="utf-8")

        result = _load_json_safe(tmp_path, "latest.json")
        assert result == {"date": "2024-01-15"}

    def test_subdirectory_allowed(self, tmp_path):
        """Files in subdirectories within data_dir must be readable."""
        (tmp_path / "history").mkdir()
        snap = tmp_path / "history" / "2024-01-15.json"
        snap.write_text('{"date": "2024-01-15"}', encoding="utf-8")

        result = _load_json_safe(tmp_path, "history/2024-01-15.json")
        assert result == {"date": "2024-01-15"}

    def test_missing_file_returns_none(self, tmp_path):
        """A valid path that does not exist returns None."""
        result = _load_json_safe(tmp_path, "latest.json")
        assert result is None
