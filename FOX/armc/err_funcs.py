"""A module with ARMC error functions.

Index
-----
.. currentmodule:: FOX.armc
.. autosummary::
    mse_normalized
    mse_normalized_weighted
    default_error_func

API
---
.. autofunction:: mse_normalized
.. autofunction:: mse_normalized_weighted
.. data:: default_error_func
    :value: FOX.armc.mse_normalized

    An alias for :func:`FOX.arc.mse_normalized`.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
    from numpy import float64 as f8

__all__ = ["mse_normalized", "mse_normalized_weighted", "default_error_func"]


@overload
def _mse(qm: NDArray[f8], mm: NDArray[f8], axes: None = ...) -> f8: ...
@overload
def _mse(qm: NDArray[f8], mm: NDArray[f8], axes: tuple[int, ...]) -> NDArray[f8]: ...
def _mse(qm, mm, axes=None):  # noqa: E302
    """Compute the mean squared error of qm and mm and reduce over the specified axes."""
    mse = (qm - mm)**2
    return mse.sum(axis=axes)


def mse_normalized(qm: ArrayLike, mm: ArrayLike) -> f8:
    """Return a normalized mean square error (MSE) over the flattened input."""
    qm = np.asarray(qm, dtype=np.float64)
    mm = np.asarray(mm, dtype=np.float64)
    scalar = _mse(qm, mm, axes=None)
    return scalar / np.abs(qm).sum()


def mse_normalized_weighted(qm: ArrayLike, mm: ArrayLike) -> f8:
    """Return a normalized mean square error (MSE) over the flattened subarrays of the input.

    >1D array-likes are herein treated as stacks of flattened arrays.

    """
    qm = np.asarray(qm, dtype=np.float64)
    mm = np.asarray(mm, dtype=np.float64)
    if qm.ndim == mm.ndim == 0:
        return _mse(qm, mm, axes=None)

    axes_qm = tuple(range(1, qm.ndim)) if qm.ndim != 0 else ()
    axes_qm_mm = tuple(range(1, max(qm.ndim, mm.ndim)))

    vector = _mse(qm, mm, axes=axes_qm_mm)
    vector /= np.abs(qm).sum(axis=axes_qm)
    return (vector**2).sum() / vector.size


default_error_func = mse_normalized
