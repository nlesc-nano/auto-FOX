"""A module with functions for sorting forcefield parameters.

Index
-----
.. currentmodule:: FOX.functions.sorting
.. autosummary::
    sort_param

API
---
.. autofunction:: sort_param

"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    SCT = TypeVar("SCT", bound=np.generic)
    NDArray = np.ndarray[Any, np.dtype[SCT]]

__all__ = ["sort_param"]


def sort_param(
    param: npt.ArrayLike,
    seperator: str = " ",
    check_duplicates: bool = True,
) -> NDArray[np.str_]:
    """Sort all atoms in an atom-based parameter set.

    Parameters represented by two atoms are simply sorted in alphabetical order.
    For parameters consisting of three atoms only the first and last atoms are
    sorted alphabetically.
    Parameters consisting of four or more atoms are not supported.

    Examples
    --------
    .. code-block ::

        >>> from FOX.functions.sorting import sort_param

        >>> param1 = [
        ...     "Cd Cd",
        ...     "Se Cd",
        ...     "Se Se",
        ... ]

        >>> param2 = [
        ...     "Cd Cd Cd",
        ...     "Se Cd Cd",
        ...     "Se Se Se",
        ... ]

        >>> sort_param(param1)
        array(['Cd Cd', 'Cd Se', 'Se Se'], dtype='<U5')

        >>> sort_param(param2)
        array(['Cd Cd Cd', 'Cd Cd Se', 'Se Se Se'], dtype='<U8')


    Parameters
    ----------
    param : array-like
        The to-be sorted parameters.
    seperator : :class:`str`
        The seperator used for splitting the atoms.
    check_duplicates : :class:`bool`
        Whether to check for duplicate elements after sorting the array.

    Returns
    -------
    :class:`np.ndarray[np.str_] <numpy.ndarray>`
        A new array with the atoms sorted within each parameter.

    Raises
    ------
    :exc:`ValueError`
        Raised when ``check_duplicates is True`` and duplicate parameters are present
        in the to-be returned array.

    """
    atoms: NDArray[np.str_] = np.asarray(param)
    if atoms.dtype.kind != "U":
        raise TypeError(f"Expected a string array; observed dtype: {atoms.dtype}")
    elif atoms.size == 0:
        return atoms if atoms is not param else atoms.copy()
    atoms_split = np.array(np.char.split(atoms, seperator).tolist())

    # Sort the atoms whenever dealing with atom-pair/triplet-based parameters
    n = atoms_split.shape[-1]
    if n == 1:
        ret = atoms if atoms is not param else atoms.copy()
    else:
        if n == 2:
            atoms_split.sort(axis=-1)
        elif n == 3:
            atoms_split[..., ::2].sort(axis=-1)
        else:
            raise NotImplementedError(
                f"Sorting parameters consisting of {n} atoms is not supported"
            )

        iterator = (seperator.join(i) for i in atoms_split.reshape(-1, n))
        ret = np.fromiter(iterator, dtype=atoms.dtype, count=atoms.size)
        ret.shape = atoms.shape

    # Check for duplicates
    if not check_duplicates:
        return ret

    unique, idx, counts = np.unique(ret, return_index=True, return_counts=True)
    is_duplicate = counts != 1
    if is_duplicate.any():
        duplicates = unique[is_duplicate]
        raise ValueError(f"Duplicate parameters: {duplicates}")
    return ret
