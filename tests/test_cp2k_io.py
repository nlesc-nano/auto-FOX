"""Tests for :mod:`FOX.io.cp2k`."""

from __future__ import annotations

import os
import io
from pathlib import Path
from contextlib import nullcontext
from typing import Any, Type, Mapping, IO, ContextManager, Callable

import pytest
import numpy as np
from FOX.io import lattice_from_cell

PATH = Path("tests") / "test_files"

CELL = PATH / "md_lattice.cell"
CELL_CLOSED = open(CELL)
CELL_CLOSED.close()

REF = np.load(PATH / "lattice_from_cell.npy")


class TestLatticeFromCell:
    """Tests for :func:`FOX.io.lattice_from_cell`."""

    @pytest.mark.parametrize("step", [None, 2])
    @pytest.mark.parametrize("stop", [None, 10])
    @pytest.mark.parametrize("start", [None, 2])
    @pytest.mark.parametrize(
        "context",
        [
            lambda: nullcontext(CELL),
            lambda: nullcontext(os.fsdecode(CELL)),
            lambda: nullcontext(os.fsencode(CELL)),
            lambda: open(CELL, "rt"),
            lambda: open(CELL, "rb"),
        ],
        ids=["PathLike[str]", "str", "bytes", "IO[str]", "IO[bytes]"],
    )
    def test_passes(
        self,
        context: Callable[[], ContextManager[str | bytes | os.PathLike[Any] | IO[Any]]],
        start: None | int,
        stop: None | int,
        step: None | int,
    ) -> None:
        with context() as f:
            value = lattice_from_cell(f, start=start, stop=stop, step=step)
        np.testing.assert_allclose(value, REF[start:stop:step])

    @pytest.mark.parametrize(
        "f,kwargs,exc",
        [
            (CELL_CLOSED, {}, ValueError),
            (2.0, {}, TypeError),
            (iter([]), {}, StopIteration),
            (open(os.devnull, "w"), {}, io.UnsupportedOperation),
            (CELL, {"start": -1}, ValueError),
            (CELL, {"stop": 1.5}, ValueError),
        ]
    )
    def test_raises(self, f: Any, kwargs: Mapping[str, Any], exc: Type[Exception]) -> None:
        with pytest.raises(exc):
            lattice_from_cell(f, **kwargs)
