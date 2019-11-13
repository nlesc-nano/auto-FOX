"""A module with functions related to manipulating atomic charges."""

import functools
from types import MappingProxyType
from typing import (
    Callable, Hashable, Optional, Collection, Mapping, Container, List, Dict, Union, Iterable,
    Tuple
)

import numpy as np
import pandas as pd

__all__ = ['update_charge']


def get_net_charge(df: pd.DataFrame, index: Optional[Collection] = None,
                   columns: Hashable = ('param', 'count')) -> float:
    """Calculate the total charge in **df**.

    Returns the (summed) product of the ``"param"`` and ``"count"`` columns in **df**.

    Parameters
    ----------
    df : |pd.DataFrame|_
        A dataframe with atomic charges.
        Charges should be stored in the ``"param"`` column and atom counts
        in the ``"count"`` column (see **key**).

    index : slice
        An object for slicing the index of **df**.

    columns : |Tuple|_ [|Hashable|_]
        The name of the columns holding the atomic charges and number of atoms (per atom type).

    Returns
    -------
    |float|_:
        The total charge in **df**.

    """
    if index is None:
        return df.loc[:, columns].product(axis=1).sum()
    return df.loc[index, columns].product(axis=1).sum()


def update_charge(atom: str, value: float, df: pd.DataFrame,
                  constrain_dict: Optional[Mapping] = None, constrain_tot: bool = True) -> None:
    """Set the atomic charge of **at** equal to **charge**.

    The atomic charges in **df** are furthermore exposed to the following constraints:

        * The total charge remains constant.
        * Optional constraints specified in **constrain_dict**
          (see :func:`.update_constrained_charge`).

    Performs an inplace update of the *param* column in **df**.

    Examples
    --------
    .. code:: python

        >>> print(df)
            param  count
        Br   -1.0    240
        Cs    1.0    112
        Pb    2.0     64

    Parameters
    ----------
    atom : str
        An atom type such as ``"Se"``, ``"Cd"`` or ``"OG2D2"``.

    charge : float
        The new charge associated with **at**.

    df : |pd.DataFrame|_
        A dataframe with atomic charges.
        Charges should be stored in the *param* column and atom counts in the *count* column.

    constrain_dict : dict
        A dictionary with charge constrains.

    """
    value_summed = get_net_charge(df)
    df.at[atom, 'param'] = value

    if not constrain_dict or atom in constrain_dict:
        exclude = constrained_update(atom, df, constrain_dict)
    else:
        exclude = [atom]

    if constrain_tot:
        unconstrained_update(value_summed, df, exclude)


def constrained_update(at1: str, df: pd.DataFrame,
                       constrain_dict: Optional[Mapping] = None) -> List[str]:
    """Perform a constrained update of atomic charges.

    Performs an inplace update of the ``"param"`` column in **df**.

    Parameters
    ----------
    at1 : str
        An atom type such as ``"Se"``, ``"Cd"`` or ``"OG2D2"``.

    df : |pd.DataFrame|_
        A dataframe with atomic charges.

    constrain_dict : dict
        A dictionary with charge constrains (see :func:`.get_charge_constraints`).

    Returns
    -------
    |list|_ [|str|_]:
        A list of atom types with updated atomic charges.

    """
    charge = df.at[at1, 'param']
    exclude = [at1]
    if not constrain_dict:
        return exclude
    exclude_append = exclude.append

    func1 = invert_ufunc(constrain_dict[at1]['func'])
    i = constrain_dict[at1]['arg']

    # Perform a constrained charge update
    for at2, values in constrain_dict.items():
        if at2 == at1:
            continue
        exclude_append(at2)

        # Unpack values
        func2 = values['func']
        j = values['arg']

        # Update the charges
        df.at[at2, 'param'] = func2(func1(charge, i), j)

    return exclude


def unconstrained_update(net_charge: float, df: pd.DataFrame,
                         exclude: Optional[Container] = None) -> None:
    """Perform an unconstrained update of atomic charges.

    The total charge in **df** is kept equal to **net_charge**.

    Performs an inplace update of the ``"param"`` column in **df**.

    Parameters
    ----------
    net_charge : float
        The total charge of the system.

    df : |pd.DataFrame|_
        A dataframe with atomic charges.

    exclude : list [str]
        A list of atom types whose atomic charges should not be updated.

    """
    exclude = exclude or ()
    include = np.array([i not in exclude for i in df.index])
    if not include.any():
        return

    i = net_charge - get_net_charge(df, np.invert(include))
    i /= get_net_charge(df, include)
    df.loc[include, 'param'] *= i


def assign_constraints(constraints: Union[str, Iterable[str]], param: pd.DataFrame, idx_key: str):
    # Parse integers and floats
    constraints = [constraints] if isinstance(constraints, str) else constraints
    constrain_list = [i.split() for i in constraints]
    for i in constrain_list:
        for j, k in enumerate(i):
            try:
                i[j] = float(k)
            except ValueError:
                pass

    # Set values in **param**
    for constrain in constrain_list:
        if '==' in i:
            pass
        else:
            _gt_lt_constraints(constrain, param, idx_key)


_INVERT = MappingProxyType({'max': 'min', 'min': 'max'})
_OPPERATOR_MAPPING = MappingProxyType({'<': 'min', '=<': 'min', '>': 'max', '>=': 'max'})


def _gt_lt_constraints(constrain: list, param: pd.DataFrame, idx_key: str) -> None:
    for i, j in enumerate(constrain):
        if j not in _OPPERATOR_MAPPING:
            continue

        operator, value, at = _OPPERATOR_MAPPING[j], constrain[i-1], constrain[i+1]
        if isinstance(at, float):
            at, value = value, at
            operator = _INVERT[operator]
        param.at[(idx_key, at), operator] = value


def _find_float(iterable: Iterable[str]) -> Tuple[str, float]:
    i, j = iterable
    try:
        return j, float(i)
    except ValueError:
        return i, float(j)


def _eq_constraints(constrain: list, param: pd.DataFrame, idx_key: str) -> None:
    ret: Dict[str, functools.partial] = {}
    constrain = ''.join(i for i in constrain).split('==')
    iterator = iter(constrain)

    # Set the first item
    item = next(iterator).split('*')
    if len(item) == 1:
        at = item[0]
        multiplier = 1.0
    elif len(item) == 2:
        at, multiplier = _find_float(item)
        multiplier **= -1
    ret[at] = functools.partial(np.multiply, 1.0)

    for item in iterator:
        at, i = _find_float(item)
        i *= multiplier
        ret[at] = functools.partial(np.multiply, i)


def invert_ufunc(ufunc: Callable) -> Callable:
    """Invert a NumPy universal function.

    Addition will be turned into substraction and multiplication into division.

    Examples
    --------
    .. code:: python

        >>> ufunc = np.add
        >>> ufunc_invert = invert_ufunc(ufunc)
        >>> print(ufunc_invert)
        <ufunc 'subtract'>

        >>> ufunc = np.multiply
        >>> ufunc_invert = invert_ufunc(ufunc)
        >>> print(ufunc_invert)
        <ufunc 'true_divide'>

    Parameters
    ----------
    ufunc : |Callable|_
        A NumPy universal function (ufunc).
        Currently accepted ufuncs are ``np.add`` and ``np.multiply``.

    Returns
    -------
    |Callable|_:
        An inverted NumPy universal function.

    """
    invert_dict = {
        np.add: np.subtract,
        np.multiply: np.divide,
    }

    try:
        return invert_dict[ufunc]
    except KeyError as ex:
        err = "'{}' is not a supported ufunc. Supported ufuncs consist of: 'add' & 'multiply'"
        raise ValueError(err.format(ufunc.__name__)).with_traceback(ex.__traceback__)
