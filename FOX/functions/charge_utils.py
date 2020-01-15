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
                  constrain_dict: Optional[Mapping] = None, charge: bool = True) -> None:
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
    net_charge = get_net_charge(df)
    df.at[atom, 'param'] = value

    if not constrain_dict or atom in constrain_dict:
        exclude = constrained_update(atom, df, constrain_dict)
    else:
        exclude = [atom]

    if charge:
        unconstrained_update(net_charge, df, exclude)


def constrained_update(at1: str, df: pd.DataFrame,
                       constrain_dict: Optional[Mapping[Hashable, Callable]] = None) -> List[str]:
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

    # Perform a constrained charge update
    func1 = invert_partial_ufunc(constrain_dict[at1])
    for at2, func2 in constrain_dict.items():
        if at2 == at1:
            continue
        exclude_append(at2)

        # Update the charges
        df.at[at2, 'param'] = func2(func1(charge))

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


def invert_partial_ufunc(ufunc: functools.partial) -> Callable:
    """Invert a NumPy universal function embedded within a :class:`functools.partial` instance."""
    func = ufunc.func
    x2 = ufunc.args[0]
    return functools.partial(func, x2**-1)


def assign_constraints(constraints: Union[str, Iterable[str]],
                       param: pd.DataFrame, idx_key: str) -> None:
    operator_set = {'>', '<', '>=', '<=', '*', '=='}

    # Parse integers and floats
    if isinstance(constraints, str):
        constraints = [constraints]

    constrain_list = []
    for item in constraints:
        intersect = operator_set.intersection(item)  # Identify all operators
        if not intersect:
            continue

        for i in intersect:  # Sanitize all operators; ensure they are surrounded by spaces
            item = item.replace(i, f' {i} ')

        item_list = item.split()
        for i, j in enumerate(item_list):  # Convert strings to floats where possible
            try:
                float_j = float(j)
            except ValueError:
                pass
            else:
                item_list[i] = float_j

        constrain_list.append(item_list)

    # Set values in **param**
    for constrain in constrain_list:
        if '==' in constrain:
            _eq_constraints(constrain, param, idx_key)
        else:
            _gt_lt_constraints(constrain, param, idx_key)


#: Map ``"min"`` to ``"max"`` and *vice versa*.
_INVERT = MappingProxyType({'max': 'min', 'min': 'max'})

#: Map :math:`>`, :math:`<`, :math:`\ge` and :math:`\le` to either ``"min"`` or ``"max"``.
_OPPERATOR_MAPPING = MappingProxyType({'<': 'min', '<=': 'min', '>': 'max', '>=': 'max'})


def _gt_lt_constraints(constrain: list, param: pd.DataFrame, idx_key: str) -> None:
    r"""Parse :math:`>`, :math:`<`, :math:`\ge` and :math:`\le`-type constraints."""
    for i, j in enumerate(constrain):
        if j not in _OPPERATOR_MAPPING:
            continue

        operator, value, at = _OPPERATOR_MAPPING[j], constrain[i-1], constrain[i+1]
        if isinstance(at, float):
            at, value = value, at
            operator = _INVERT[operator]
        param.at[(idx_key, at), operator] = value


def _find_float(iterable: Tuple[str, str]) -> Tuple[str, float]:
    """Take an iterable of 2 strings and identify which element can be converted into a float."""
    i, j = iterable
    try:
        return j, float(i)
    except ValueError:
        return i, float(j)


def _eq_constraints(constrain: list, param: pd.DataFrame, idx_key: str) -> None:
    """Parse :math:`a = i * b`-type constraints."""
    constrain_dict: Dict[str, functools.partial] = {}
    constrain = ''.join(str(i) for i in constrain).split('==')
    iterator = iter(constrain)

    # Set the first item; remove any prefactor and compensate al other items if required
    item = next(iterator).split('*')
    if len(item) == 1:
        at = item[0]
        multiplier = 1.0
    elif len(item) == 2:
        at, multiplier = _find_float(item)
        multiplier **= -1
    constrain_dict[at] = functools.partial(np.multiply, 1.0)

    # Assign all other constraints
    for item in iterator:
        item = item.split('*')
        at, i = _find_float(item)
        i *= multiplier
        constrain_dict[at] = functools.partial(np.multiply, i)

    # Update the dataframe
    param['constraints'] = None
    for at, _ in param.loc[idx_key].iterrows():
        param.at[(idx_key, at), 'constraints'] = constrain_dict
