"""
Unit and regression test for the QuantumPioneer package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import QuantumPioneer


def test_QuantumPioneer_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "QuantumPioneer" in sys.modules
