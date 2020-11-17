"""A module for parsing wildcards (``"X"``) specified in a CHARMM parameter set.

Index
-----
.. currentmodule:: FOX.ff.parse_wildcards
.. autosummary::
    parse_wildcards

API
---
.. autofunction:: parse_wildcards

"""

from types import MappingProxyType
from typing import Collection, Mapping, Callable, Set, Sequence, TypeVar, Any
from itertools import product, permutations

import numpy as np
import pandas as pd

__all__ = ['parse_wildcards']

_T = TypeVar("_T")
_IntersectionFunc = Callable[[Sequence[Any], Set[Sequence[Any]]], bool]


def parse_wildcards(df: pd.DataFrame, symbols: Collection[str], prm_type: str) -> None:
    """Replace any wildcards (``"X"``) in **df** with explicit references to atom types.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        A DataFrame with parameters.
        Atom-pairs should be stored in its (multi-)index.

    symbols : :class:`Collection<collections.abc.Collection>` [:class:`str`]
        A collection with all unique atom types within the molecule of interest.
        Used for substituting ``"X"``.

    prm_type : :class:`str`

    """
    # Check if a wildcard is present
    idx_list = df.index.tolist()
    idx_array = np.array(idx_list)
    if 'X' not in idx_array:
        return

    try:
        is_in_index = INTERSECTION_MAPPING[prm_type]
    except KeyError as ex:
        raise ValueError(f"Invalid value ({repr(prm_type)}) for 'prm_type'; allowed values: "
                         f"{tuple(INTERSECTION_MAPPING.keys())} ").with_traceback(ex.__traceback__)

    index = set(df.index)
    for tup in idx_list:
        if 'X' not in tup:  # Search for wildcards
            continue

        # Replace wildcards with explicit references to atom types
        seq = [([i] if i != 'X' else symbols) for i in tup]
        seq_product = product(*seq)
        for i in seq_product:
            if not is_in_index(i, index):  # Do not add any duplicates
                df.loc[i] = df.loc[tup]
                index.add(i)


def _invert(tup: Sequence[_T], tup_set: Set[Sequence[_T]]) -> bool:
    """Check if **tup_set** contains ``tup`` or ``tup[::-1]``."""
    return bool(tup_set.intersection([tup, tup[::-1]]))


def _all_comb(tup: Sequence[_T], tup_set: Set[Sequence[_T]]) -> bool:
    """Check if **tup_set** any combination of ``tup[0]`` and a permutation of ``tup[1:]``."""
    i_0, *i_n = tup
    iterable = ((i_0,) + i for i in permutations(i_n))
    return bool(tup_set.intersection(iterable))


INTERSECTION_MAPPING: Mapping[str, _IntersectionFunc] = MappingProxyType({
    'bonds': _invert,
    'angles': _invert,
    'urey_bradley': _invert,
    'dihedrals': _invert,
    'impropers': _all_comb
})
