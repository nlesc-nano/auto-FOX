"""Tests for :mod:`FOX.functions.periodic`."""

from typing import Any, Type
from itertools import combinations, chain

import pytest
import numpy as np
from assertionlib import assertion

from FOX.functions.periodic import parse_periodic


class TestParsePeriodic:
    """Tests for :func:`FOX.functions.periodic.parse_periodic`."""

    @pytest.mark.parametrize(
        "xyz",
        chain(
            ["xyz", "zxxyxyyzzyz", "x", "y", "z", 0, 1, 2],
            combinations("xyz", r=1),
            combinations("xyz", r=2),
            combinations("xyz", r=3),
            combinations(range(3), r=1),
            combinations(range(3), r=2),
            combinations(range(3), r=3),
            [np.arange(3), np.array(["x", "y", "z"])],
        )
    )
    def test_pass(self, xyz: Any) -> None:
        out = parse_periodic(xyz)
        assertion.issubset(out, {0, 1, 2})

    @pytest.mark.parametrize(
        "xyz,exc",
        [
            (["a", "b"], ValueError),
            (np.array([], dtype=np.intp), ValueError),
            ([-1, 0, 1], ValueError),
            (np.arange(9).reshape(3, 3), ValueError),
            ([True, False], TypeError),
            ([0.0, 1.0], TypeError),
            (b"xyz", TypeError),
        ]
    )
    def test_raises(self, xyz: Any, exc: Type[Exception]) -> None:
        with pytest.raises(exc):
            parse_periodic(xyz)
