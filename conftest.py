"""A pytest ``conftest.py`` file."""

import warnings
from typing import Any

import pandas as pd


def pytest_configure(config: Any) -> None:
    """Silence the pandas :exc:`~pd.errors.PerformanceWarning`."""
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
