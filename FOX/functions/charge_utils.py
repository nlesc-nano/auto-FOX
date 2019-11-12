"""A module with functions related to manipulating atomic charges."""

import functools
from typing import Callable, Hashable, Optional, Collection, Mapping, Container, List, Dict

import numpy as np
import pandas as pd

from scm.plams import Settings

__all__ = ['update_charge', 'get_charge_constraints']


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


def get_charge_constraints(constrain: str) -> Dict[str, functools.partial]:
    r"""Construct a set of charge constraints out of a string.

    Take a string containing a set of interdependent charge constraints and translate
    it into a dictionary containing all arguments and operators.

    The currently supported operators are:

    ================= =======
     Operation         Symbol
    ================= =======
     addition_        ``+``
     multiplication_  ``*``
    ================= =======

    .. _addition: https://docs.scipy.org/doc/numpy/reference/generated/numpy.add.html
    .. _multiplication: https://docs.scipy.org/doc/numpy/reference/generated/numpy.multiply.html

    Examples
    --------
    An example where :math:`q_{Cd} = -0.5*q_{O} = -q_{Se}`:

    .. code:: python

        >>> constrain = 'Cd = -0.5 * O = -1 * Se'
        >>> get_charge_constraints(constrain)
        Cd:
            arg: 	1.0
            func: 	<ufunc 'multiply'>
        O:
            arg:   -0.5
            func: 	<ufunc 'multiply'>
        Se:
            arg:   -1.0
            func: 	<ufunc 'multiply'>

    Another example where the following (nonensical) constraint is applied:
    :math:`q_{Cd} = q_{H} - 1 = q_{O} + 1.5 = 0.5 * q_{Se}`:

    .. code:: python

        >>> constrain = 'Cd = H + -1 = O + 1.5 = 0.5 * Se'
        >>> get_charge_constraints(constrain)
        Cd:
            arg: 	1.0
            func: 	<ufunc 'multiply'>
        H:
            arg:   -1.0
            func: 	<ufunc 'add'>
        O:
            arg: 	1.5
            func: 	<ufunc 'add'>
        Se:
            arg: 	0.5
            func: 	<ufunc 'multiply'>

    Parameters
    ----------
    item : str
        A string with all charge constraints.

    Returns
    -------
    |plams.Settings|_:
        A Settings object with all charge constraints.

    """
    def _loop(i, operator_dict):
        for operator in operator_dict:
            split = i.split(operator)
            if len(split) == 2:
                return split[0], split[1], operator
        return split[0], 1.0, '*'

    operator_dict = {'*': np.multiply}
    list_ = [i for i in constrain.split('==') if i]

    ret = {}
    for i in list_:
        # Seperate the operator from its arguments
        arg1, arg2, operator = _loop(i, operator_dict)

        # Identify keys and values
        try:
            arg = float(arg1)
            key = arg2.split()[0]
        except ValueError:
            arg = float(arg2)
            key = arg1.split()[0]

        # Prepare and return the arguments and operators
        ret[key] = functools.partial(operator_dict[operator], arg)
    return ret


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
