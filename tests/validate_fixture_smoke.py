"""Compatibility entry point for the offline module-based fixture smoke validation.

This replaces the notebook-cell executor used before the Python-project refactor.
"""
from pathlib import Path

import pytest


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).with_name("test_fixture_harness.py")), "-q"]))
