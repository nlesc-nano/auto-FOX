"""A module with ARMC error functions.

Index
-----
.. currentmodule:: FOX.armc
.. autosummary::
    mse_normalized
    mse_normalized_weighted
    mse_normalized_max
    mse_normalized_v2
    mse_normalized_weighted_v2
    default_error_func

API
---
.. autofunction:: mse_normalized
.. autofunction:: mse_normalized_weighted
.. autofunction:: mse_normalized_max
.. autofunction:: mse_normalized_v2
.. autofunction:: mse_normalized_weighted_v2
.. autofunction:: err_normalized
.. autofunction:: err_normalized_weighted
.. data:: default_error_func
    :value: FOX.armc.mse_normalized

    An alias for :func:`FOX.arc.mse_normalized`.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
    from numpy import float64 as f8

__all__ = [
    "mse_normalized",
    "mse_normalized_weighted",
    "default_error_func",
    "mse_normalized_v2",
    "mse_normalized_weighted_v2",
    "err_normalized",
    "err_normalized_weighted",
]


@overload
def _get_mse(qm: NDArray[f8], mm: NDArray[f8], axis: None = ...) -> f8: ...
@overload
def _get_mse(qm: NDArray[f8], mm: NDArray[f8], axis: tuple[int, ...]) -> NDArray[f8]: ...
def _get_mse(qm, mm, axis=None):  # noqa: E302
    """Compute the mean squared error of qm and mm and reduce over the specified axes."""
    mse = (qm - mm)**2
    return mse.sum(axis=axis)


def mse_normalized(qm: ArrayLike, mm: ArrayLike) -> f8:
    """Return a normalized mean square error (MSE) over the flattened input."""
    qm = np.asarray(qm, dtype=np.float64)
    mm = np.asarray(mm, dtype=np.float64)
    scalar = _get_mse(qm, mm, axis=None)
    return scalar / np.abs(qm).sum()


def mse_normalized_weighted(qm: ArrayLike, mm: ArrayLike) -> f8:
    """Return a normalized mean square error (MSE) over the flattened subarrays of the input.

    >1D array-likes are herein treated as stacks of flattened arrays.

    """
    if isinstance(qm, pd.DataFrame):
        qm = np.array(qm, dtype=np.float64, ndmin=1, copy=False).T
    else:
        qm = np.array(qm, dtype=np.float64, ndmin=1, copy=False)

    if isinstance(mm, pd.DataFrame):
        mm = np.array(mm, dtype=np.float64, ndmin=1, copy=False).T
    else:
        mm = np.array(mm, dtype=np.float64, ndmin=1, copy=False)

    axes_qm = tuple(range(1, qm.ndim))
    axes_qm_mm = tuple(range(1, max(qm.ndim, mm.ndim)))

    vector = _get_mse(qm, mm, axis=axes_qm_mm)
    vector /= np.abs(qm).sum(axis=axes_qm)
    return (vector**2).sum() / vector.size


def mse_normalized_max(qm: ArrayLike, mm: ArrayLike) -> f8:
    """Return the maximum normalized mean square error (MSE) over the flattened subarrays of the input.

    >1D array-likes are herein treated as stacks of flattened arrays.

    """  # noqa: E501
    if isinstance(qm, pd.DataFrame):
        qm = np.array(qm, dtype=np.float64, ndmin=1, copy=False).T
    else:
        qm = np.array(qm, dtype=np.float64, ndmin=1, copy=False)

    if isinstance(mm, pd.DataFrame):
        mm = np.array(mm, dtype=np.float64, ndmin=1, copy=False).T
    else:
        mm = np.array(mm, dtype=np.float64, ndmin=1, copy=False)

    axes_qm = tuple(range(1, qm.ndim))
    axes_qm_mm = tuple(range(1, max(qm.ndim, mm.ndim)))

    vector = _get_mse(qm, mm, axis=axes_qm_mm)
    vector /= np.abs(qm).sum(axis=axes_qm)
    return vector.max()


def mse_normalized_v2(qm: ArrayLike, mm: ArrayLike) -> f8:
    """Return a normalized mean square error (MSE) over the flattened input.

    Normalize before squaring the error.

    """
    qm = np.asarray(qm, dtype=np.float64)
    mm = np.asarray(mm, dtype=np.float64)

    delta = np.abs(qm - mm)
    delta /= np.abs(qm).sum()
    return (delta**2).sum()


def mse_normalized_weighted_v2(qm: ArrayLike, mm: ArrayLike) -> f8:
    """Return a normalized mean square error (MSE) over the flattened subarrays of the input.

    >1D array-likes are herein treated as stacks of flattened arrays.

    Normalize before squaring the error.

    """
    if isinstance(qm, pd.DataFrame):
        qm = np.array(qm, dtype=np.float64, ndmin=1, copy=False).T
    else:
        qm = np.array(qm, dtype=np.float64, ndmin=1, copy=False)

    if isinstance(mm, pd.DataFrame):
        mm = np.array(mm, dtype=np.float64, ndmin=1, copy=False).T
    else:
        mm = np.array(mm, dtype=np.float64, ndmin=1, copy=False)

    axes_qm_mm = tuple(range(1, max(qm.ndim, mm.ndim)))
    axes_qm = tuple(range(1, qm.ndim))
    padding_qm = len(axes_qm) * (None,)

    delta = np.abs(qm - mm)
    delta /= np.abs(qm).sum(axis=axes_qm)[(..., *padding_qm)]
    err_vec = np.sum(delta**2, axis=axes_qm_mm)
    return (err_vec**2).sum() / err_vec.size


def err_normalized(qm: ArrayLike, mm: ArrayLike) -> f8:
    """Return a normalized wrror over the flattened input.

    Normalize before taking the exponent - 1 of the error.

    """
    qm = np.asarray(qm, dtype=np.float64)
    mm = np.asarray(mm, dtype=np.float64)

    delta = np.abs(qm - mm)
    delta /= np.abs(qm).sum()
    return delta.sum()


def err_normalized_weighted(qm: ArrayLike, mm: ArrayLike) -> f8:
    """Return a normalized error over the flattened subarrays of the input.

    >1D array-likes are herein treated as stacks of flattened arrays.

    """
    if isinstance(qm, pd.DataFrame):
        qm = np.asarray(qm, dtype=np.float64).T
    else:
        qm = np.array(qm, dtype=np.float64, ndmin=1, copy=False)

    if isinstance(mm, pd.DataFrame):
        mm = np.asarray(mm, dtype=np.float64).T
    else:
        mm = np.array(mm, dtype=np.float64, ndmin=1, copy=False)

    axes_qm_mm = tuple(range(1, max(qm.ndim, mm.ndim)))
    axes_qm = tuple(range(1, qm.ndim))
    padding_qm = len(axes_qm) * (None,)

    delta = np.abs(qm - mm)
    delta /= np.abs(qm).sum(axis=axes_qm)[(..., *padding_qm)]
    err_vec = delta.sum(axis=axes_qm_mm)
    return (err_vec**2).sum() / err_vec.size


default_error_func = mse_normalized
