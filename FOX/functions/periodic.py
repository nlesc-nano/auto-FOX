"""Utility functions related to calculations on periodic systems.

Index
-----
.. currentmodule:: FOX.functions.periodic
.. autosummary::
    parse_periodic

API
---
.. autofunction:: parse_periodic

"""

from __future__ import annotations

from types import MappingProxyType
from typing import TypeVar, Any, Sequence, Mapping, Union, Iterable, TYPE_CHECKING
from itertools import chain

import numpy as np

if TYPE_CHECKING:
    import sys
    if sys.version_info >= (3, 8):
        from typing import Literal, TypedDict
    else:
        from typing_extensions import Literal, TypedDict

    SCT = TypeVar("SCT", bound=np.generic)
    NDArray = np.ndarray[Any, np.dtype[SCT]]

    class _XYZDict(TypedDict):
        x: Literal[0]
        y: Literal[1]
        z: Literal[2]


def parse_periodic(
    xyz: Union[
        int, str,
        Sequence[str], Sequence[int],
        NDArray[np.str_], NDArray[np.integer],
    ]
) -> NDArray[np.intp]:
    """Parse the passed periodicity specifier and convert it into an array of integers.

    Parameters
    ----------
    xyz : :class:`str` or :class:`Sequence[int] <collections.abc.Sequence>`
        A string or sequence of integers representing .
        Expects either ``"x"``, ``"y"`` and/or ``"z"`` or ``0``, ``1`` and/or ``2``.

    Returns
    -------
    :class:`np.ndarray[np.intp] <numpy.ndarray>`
        An array with the indices representing the ``x``, ``y`` and/or ``z`` axes.

    """
    ar: NDArray[np.str_ | np.integer] = np.array(xyz, ndmin=1)
    if ar.ndim != 1:
        raise ValueError(f"Expected a 1D array; observed dimensionality: {ar.ndim}")
    elif ar.size == 0:
        raise ValueError("Expected a non-empty array")

    if ar.dtype.kind == "U":
        return _parse_char(ar)
    elif ar.dtype.kind in "ui":
        return _parse_int(ar)
    else:
        raise TypeError(f"Invalid dtype: {ar.dtype}")


_XYZ_DICT: _XYZDict = MappingProxyType({  # type: ignore[assignment]
    "x": 0,
    "y": 1,
    "z": 2,
})

_012_DICT: Mapping[int, int] = MappingProxyType({
    0: 0,
    1: 1,
    2: 2,
})


def _parse_char(ar: NDArray[np.str_]) -> NDArray[np.intp]:
    """Helper for :func:`parse_periodic`; parses :class:`numpy.str_`-based arrays."""
    ar_low: Iterable[Literal["x", "y", "z"]] = np.fromiter(
        chain.from_iterable(np.char.lower(ar)),
        dtype="U1",
    )
    try:
        ret: NDArray[np.intp] = np.fromiter({_XYZ_DICT[i] for i in ar_low}, dtype=np.intp)
    except KeyError as ex:
        raise ValueError(f"Invalid axis specifier: {ex}") from None
    ret.sort()
    return ret


def _parse_int(ar: NDArray[np.integer]) -> NDArray[np.intp]:
    """Helper for :func:`parse_periodic`; parses :class:`numpy.integer`-based arrays."""
    try:
        ret: NDArray[np.intp] = np.fromiter({_012_DICT[i] for i in ar}, dtype=np.intp)
    except KeyError as ex:
        raise ValueError(f"Invalid axis specifier: {ex}") from None
    ret.sort()
    return ret
