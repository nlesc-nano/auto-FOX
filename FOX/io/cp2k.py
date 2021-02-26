"""Functions for reading various CP2K-specific files.

Index
-----
.. currentmodule:: FOX.io
.. autosummary::
    lattice_from_cell

API
---
.. autoclass:: lattice_from_cell

"""

from __future__ import annotations

import os
import io
from itertools import chain, islice
from contextlib import nullcontext
from typing import Any, TYPE_CHECKING, TypeVar, ContextManager, IO

import numpy as np

if TYPE_CHECKING:
    from numpy import float64 as f8

    SCT = TypeVar("SCT", bound=np.generic)
    NDArray = np.ndarray[Any, np.dtype[SCT]]

__all__ = ["lattice_from_cell"]


def lattice_from_cell(
    f: str | bytes | os.PathLike[Any] | IO[Any],
    *,
    start: None | int = None,
    stop: None | int = None,
    step: None | int = None,
    **kwargs: Any,
) -> NDArray[f8]:
    r"""Read the lattice vectors over the course of trajectory from a cp2k ``.cell`` file.

    Examples
    --------
    .. code-block:: python

        >>> from FOX.io import lattice_from_cell
        >>> import numpy as np

        >>> file: str = ...
        >>> lattice: np.ndarray = lattice_from_cell(file)

        >>> print(lattice)
        [[[ 2.56410575e+01  0.00000000e+00  0.00000000e+00]
          [-2.42430000e-06  3.57209995e+01  0.00000000e+00]
          [-2.04520000e-06  1.05180000e-05  2.47990135e+01]]

         [[ 2.57058882e+01 -3.25424621e-02 -7.29052350e-03]
          [-4.53386687e-02  3.57603379e+01 -1.85884753e-02]
          [-7.05385980e-03 -1.28941496e-02  2.47920255e+01]]

         [[ 2.57840364e+01 -6.22863141e-02 -1.18056966e-02]
          [-8.67802884e-02  3.58200590e+01 -3.36974289e-02]
          [-1.14266041e-02 -2.33830077e-02  2.48013783e+01]]

        ...

         [[ 2.54829490e+01  9.60182721e-02  3.26757600e-02]
          [ 1.37222700e-01  3.63232404e+01  1.15509941e-01]
          [ 3.64005150e-02  7.73351616e-02  2.54793484e+01]]

         [[ 2.54963145e+01  6.30815199e-02  3.92088303e-02]
          [ 9.04496393e-02  3.63126879e+01  8.81486322e-02]
          [ 4.28814403e-02  5.79440286e-02  2.55123757e+01]]

         [[ 2.55036429e+01  2.90665818e-02  3.71377603e-02]
          [ 4.21389796e-02  3.62839548e+01  5.99095945e-02]
          [ 4.07402274e-02  3.79233571e-02  2.55459613e+01]]]


    Parameters
    ----------
    f : path- or file-like
        A :term:`path-like object` or :term:`file-like object` representing the to-be read file.
        File-like objects must be opened in read and text mode.
    start/stop/step : :class:`int`, optional
        Specify a subset of the to-be returned cell parameters according to standard slice notation:
        ``[start:stop:step]``. Note that passed values must be positive or zero-valued.
    \**kwargs : :data:`~typing.Any`
        Further keyword arguments for :func:`open`.

    Returns
    -------
    :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(n, 3, 3)`
        A 3D array with the all lattice vectors from **f**.
        If **start**, **stop** and/or **step** are specified then a, user-specified,
        subset of the lattice vectors will be returned.

    """
    try:
        cm: ContextManager[IO[Any]] = open(f, **kwargs)  # type: ignore[arg-type]
    except TypeError:
        cm = nullcontext(f)  # type: ignore[arg-type]

    with cm as f:
        try:
            _ = next(f)
        except (io.UnsupportedOperation, StopIteration):
            raise
        except Exception as ex:
            if getattr(f, "closed", False):
                raise
            raise TypeError("Expected a path- or file-like object; "
                            f"observed type: {f.__class__.__name__!r}") from ex

        iterator = chain.from_iterable(i.split()[2:11] for i in islice(f, start, stop, step))
        return np.fromiter(iterator, dtype=np.float64).reshape(-1, 3, 3)
