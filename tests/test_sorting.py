"""Tests for :mod:`FOX.functions.sorting`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Mapping, Any, Type
from itertools import chain

import numpy as np
import h5py
import pytest
from FOX.functions.sorting import sort_param

if TYPE_CHECKING:
    import numpy.typing as npt

PATH = Path('tests') / 'test_files'
HDF5 = PATH / 'sort_param.hdf5'


def construct_param(sep: str) -> Tuple[Tuple[str, npt.ArrayLike, str], ...]:
    s = sep
    a = ("")
    b = ("Cd", "Se")
    c = (f"Cd{s}Se", f"Se{s}Cd")
    d = (f"Cd{s}Cd{s}Cd", f"Cd{s}Cd{s}Se", f"Cd{s}Se{s}Se", f"Cd{s}Se{s}Cd", f"Se{s}Se{s}Se")

    ret = [
        np.array(i, ndmin=j) for i in chain(a, b, c, d) for j in range(3)
    ]
    ret += [i.tolist() for i in ret]
    ret += [np.array([], dtype=np.str_), np.array([[]], dtype=np.str_),
            np.array([[[]]], dtype=np.str_)]
    return tuple((str(i), item, sep) for i, item in enumerate(ret))


PARAM = construct_param(" ") + construct_param("|")


@pytest.mark.parametrize("name,param,sep", PARAM)
def test_passes(name: str, param: npt.ArrayLike, sep: str) -> None:
    """Test :func:`FOX.functions.sorting.sort_param`."""
    name += sep
    with h5py.File(HDF5, "r", libver="latest") as f:
        ref = f[name][...].astype(np.str_)

    out = sort_param(param, sep)
    np.testing.assert_array_equal(out, ref)


@pytest.mark.parametrize(
    "kwargs,exc",
    [
        ({"param": range(3)}, TypeError),
        ({"param": range(3), "casting": "safe"}, TypeError),
        ({"param": "Cd Cd Cd Cd"}, NotImplementedError),
        ({"param": ["Cd Cd Cd", "Cd Cd Cd"]}, ValueError),
    ]
)
def test_raises(kwargs: Mapping[str, Any], exc: Type[Exception]) -> None:
    """Test :func:`FOX.functions.sorting.sort_param`."""
    with pytest.raises(exc):
        sort_param(**kwargs)
