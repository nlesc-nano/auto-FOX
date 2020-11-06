"""A set of functions for creating histograms.

Examples
--------
A workflow for creating histograms of non-bonded interactions.

.. code:: python

    >>> import pandas as pd
    >>> from scm.plams import Units
    >>> from FOX import MultiMolecule, PSFContainer, get_non_bonded
    >>> from FOX.recipes plot_descriptor

    >>> mol = MoltiMolecule.read_xyz(...)
    >>> psf = PSFContainer.read(...)
    >>> prm = PRMContainer.read(...)

    # CP2K Settings can, optionally, be passed to overwrite .psf/.prm parameters
    >>> elstat_df, lj_df = get_non_bonded(mol, psf, prm=prm, cp2k_settings=None)

    >>> total_df: pd.DataFrame = elstat_df + lj_df  # Total interaction between n and m particles
    >>> total_df *= Units.conversion_ratio('au', 'kcal/mol')

    >>> for at1, at2 in total_df.keys():  # Average interaction between 1 and m particles
    ...     at1_count = np.count_nonzero(psf.atom_type == at1)
    ...     total_df[at1, at2] /= at1_count

    >>> plot_descriptor(total_df, sharex=False, sharey=True, kind='hist', bins=50)


Only keep atom pairs within the core (``"Cd"`` and ``"Se"``).

.. code:: python

    >>> ...
    >>> core_set = {'Cd', 'Se'}
    >>> core_df: pd.DataFrame = filter_atoms(total_df, core_set)

    >>> plot_descriptor(core_df, sharex=False, sharey=True, kind='hist', bins=50)


Or all ligand/core pairs.

.. code:: python

    >>> ...
    >>> core_lig_df: pd.DataFrame = filter_atoms(total_df, atom_set, filter_func='between_sets')

    >>> plot_descriptor(lig_core_df, sharex=False, sharey=True, kind='hist', bins=50)


Sum the interactions.

.. code:: python

    >>> ...

    >>> from itertools import chain

    >>> columns = list(set(chain.from_iterable(total_df.keys())))
    >>> total_df_sum = pd.DataFrame(0.0, index=total_df.index.copy(), columns=columns)
    >>> for (at1, at2), series in total_df.items():
    ...     total_df_sum[at1] += series
    ...     total_df_sum[at2] += series

    >>> plot_descriptor(total_df_sum, sharex=False, sharey=True, kind='hist', bins=50)

Index
-----
.. currentmodule:: FOX.recipes
.. autosummary::
    filter_atoms

API
---
.. autofunction:: filter_atoms

"""

import copy
from types import MappingProxyType
from typing import Iterable, Union, Set, TypeVar, Callable, Mapping

import numpy as np
import pandas as pd

__all__ = ['filter_atoms']

T = TypeVar('T')
FilterFunc = Callable[[Set[T], Iterable[T]], bool]


def within_set(atom_set: Set[str], atom_pair: Iterable[str]) -> bool:
    """Check if **atom_set** is a superset of **atom_pair**."""
    return atom_set.issuperset(atom_pair)


def between_sets(atom_set: Set[str], atom_pair: Iterable[str], n: int = 1) -> bool:
    """Check if :math:`n` atoms within in atom_set **atom_set** are part of **atom_pair**."""
    return len(atom_set.intersection(atom_pair)) == n


FILTER_MAPPING: Mapping[str, FilterFunc] = MappingProxyType({
    'within_set': within_set,
    'between_sets': between_sets
})


def filter_atoms(df: pd.DataFrame, atoms: Union[str, Iterable[str]],
                 filter_func: Union[str, FilterFunc] = 'within_set') -> pd.DataFrame:
    """Return a new DataFrame with all columns from **df** whose keys are not part of **atoms_keep**.

    Examples
    --------
    .. code:: python

        >>> from itertools import combinations_with_replacement
        >>> import pandas as pd

        >>> columns = pd.MultiIndex.from_tuples(
        ...     combinations_with_replacement(sorted(['Cd', 'Se', 'O', 'C']), r=2)
        ... )

        >>> df = pd.DataFrame(True, index=[0], columns=columns)
        >>> print(df)
              C                      Cd                 O          Se
              C    Cd     O    Se    Cd     O    Se     O    Se    Se
        0  True  True  True  True  True  True  True  True  True  True

        >>> atoms = {'Cd', 'Se'}
        >>> df_new = filter_atoms(df, atoms)
        >>> print(df_new)
             Cd          Se
             Cd    Se    Se
        0  True  True  True

    Parameters
    ----------
    df : :class:`pd.DataFrame` or :class:`MutableMapping<collections.abc.MutableMapping>`
        A DataFrame or a Mutable mapping.
        Iterating over its :meth:`keys<dict.keys>` should yield n-tuples with atom types.

    keep_atoms : :class:`str` or :class:`Iterable<collections.abc.Iterable>` [:class:`str`]
        An atomt ype or iterable of multiple atom types.
        Values are deleted from **df** if key is encountered whose elements are not all present
        in **atoms_keep**.

    filter_func : :class:`str`
        The function for filtering keys.
        Accepted values are ``"within_set"`` and ``"between_sets"``.

    Returns
    -------
    :class:`pd.DataFrame`
        A copy of **df** with all keys not part of **atoms_keep** removed.

    """  # noqa
    filter_func = _validate_filter_func(filter_func)

    if isinstance(atoms, (str, np.str)):
        atom_set = {atoms}
    else:
        atom_set = set(atoms)

    ret = copy.copy(df)
    for at_pair in df.keys():
        if not filter_func(atom_set, at_pair):
            del ret[at_pair]
    return ret


def _validate_filter_func(filter_func: Union[str, FilterFunc]) -> FilterFunc:
    """Validate the **filter_func** parameter for :func:`filter_atoms`."""
    if callable(filter_func):  # Is it a callable?
        return filter_func

    try:  # It is probably a string
        return FILTER_MAPPING[filter_func.lower()]

    except (AttributeError, TypeError) as ex:  # Guess it's not a string after all
        cls_name = filter_func.__class__.__name__
        raise TypeError("'filter_func' expected a string or callable; observed type: "
                        f"'{cls_name}'").with_traceback(ex.__traceback__)

    except KeyError as ex:  # A string with an incorrect value
        accepted_values = tuple(sorted(FILTER_MAPPING.keys()))
        name = filter_func.lower()
        raise ValueError(f"{repr(name)} is not a valid value for 'filter_func'; "
                         f"accepted values: {accepted_values}").with_traceback(ex.__traceback__)
