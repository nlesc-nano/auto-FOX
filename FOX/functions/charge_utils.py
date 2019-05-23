"""A module with functions related to manipulating atomic charges."""

from typing import Callable

import numpy as np
import pandas as pd

from scm.plams import Settings

__all__ = ['update_charge', 'get_charge_constraints']


def get_net_charge(df: pd.DataFrame,
                    index_slice=None,
                    charge: str = 'param',
                    atom_count: str = 'count') -> float:
    """Calculate the total charge in **df**.

    Returns the (summed) product of the **charge** and **atom_count** columns in **df**.

    :parameter df: A dataframe with atomic charges.
        Charges should be stored in the *param* column and atom counts in the *count* column.
    :type df: pd.DataFrame
    :parameter object index_slice: An object for slicing the index of **df**.
    :parameter str charge: The name of the column holding the atomic charges (per atom type).
    :parameter str atom_count: The name of the column holding the number of atoms (per atom type).
    :return: The total charge in **df**.
    :rtype: |float|_
    """
    if index_slice is None:
        return df.loc[:, [charge, atom_count]].product(axis=1).sum()
    return df.loc[index_slice, [charge, atom_count]].product(axis=1).sum()


def update_charge(at: str,
                  charge: float,
                  df: pd.DataFrame,
                  constrain_dict: dict = {}) -> None:
    """Set the atomic charge of **at** equal to **charge**.

    The atomic charges in **df** are furthermore exposed to the following constraints:

        * The total charge remains constant.
        * Optional constraints specified in **constrain_dict**
          (see :func:`.update_constrained_charge`).

    Performs an inplace update of the *param* column in **df**.

    Example **df**:

    .. code:: python

        >>> print(df)
            param  count
        Br   -1.0    240
        Cs    1.0    112
        Pb    2.0     64

    :parameter str at: An atom type such as *Se*, *Cd* or *OG2D2*.
    :parameter float charge: The new charge associated with **at**.
    :parameter df: A dataframe with atomic charges.
        Charges should be stored in the *param* column and atom counts in the *count* column.
    :type df: |pd.DataFrame|_
    :parameter dict constrain_dict: A dictionary with charge constrains.
    """
    net_charge = get_net_charge(df)
    df.at[at, 'param'] = charge

    if at in constrain_dict or not constrain_dict:
        exclude = update_constrained_charge(at, df, constrain_dict)
        update_unconstrained_charge(net_charge, df, exclude)
    else:
        exclude = [at]
        update_unconstrained_charge(net_charge, df, exclude)
        condition = [at in exclude for at in df.index]
        charge = net_charge - get_net_charge(df, condition)
        q = find_q(df, charge, constrain_dict)

        key, value = next(iter(constrain_dict.items()))
        func = invert_ufunc(value['func'])
        df.at[key, 'param'] = func(q, value['arg'])
        update_constrained_charge(key, df, constrain_dict)


def update_constrained_charge(at1: str,
                              df: pd.DataFrame,
                              constrain_dict: dict = {}) -> list:
    """Perform a constrained update of atomic charges.

    Performs an inplace update of the *charge* column in **df**.

    :parameter str at1: An atom type such as *Se*, *Cd* or *OG2D2*.
    :parameter df: A dataframe with atomic charges.
    :type df: |pd.DataFrame|_
    :parameter dict constrain_dict: A dictionary with charge constrains.
    :return: A list of atom types with updated atomic charges.
    :rtype: |list|_ [|str|_]
    """
    charge = df.at[at1, 'param']
    exclude = [at1]
    if not constrain_dict:
        return exclude

    func1 = invert_ufunc(constrain_dict[at1]['func'])
    i = constrain_dict[at1]['arg']

    # Perform a constrained charge update
    for at2, values in constrain_dict.items():
        if at2 == at1:
            continue
        exclude.append(at2)

        # Unpack values
        func2 = values['func']
        j = values['arg']

        # Update the charges
        df.at[at2, 'param'] = func2(func1(charge, i), j)

    return exclude


def update_unconstrained_charge(net_charge: float,
                                df: pd.DataFrame,
                                exclude: list = []) -> None:
    """Perform an unconstrained update of atomic charges.

    Performs an inplace update of the *charge* column in **df**.

    :parameter float net_charge: The total charge of the system.
    :parameter df: A dataframe with atomic charges.
    :type df: |pd.DataFrame|_
    :parameter dict constrain_dict: A list of atom types whose atomic charges should not be updated.
    """
    include = np.array([i not in exclude for i in df.index])
    if not include.any():
        return
    i = net_charge - get_net_charge(df, np.invert(include))
    i /= get_net_charge(df, include)
    df.loc[include, 'param'] *= i


def find_q(df: pd.DataFrame,
           Q: float = 0.0,
           constrain_dict: dict = {}) -> float:
    r"""Calculate the atomic charge :math:`q` given the total charge :math:`Q`.

    Atom subsets are denoted by :math:`m` & :math:`n`, with :math:`a` & :math:`b`
    being subset-dependent constants.

    .. math::

        Q = \sum_{i} (q * a_{i}) * \sum_{m_{i}} m_{i} + \sum_{j} (q + b_{j}) * \sum_{n_{j}} n_{j}

        q = \frac{Q + \sum_{j} (q * b_{j}) * \sum_{n_{j}} n_{j}}
            {\sum_{i} (q * a_{i}) * \sum_{m_{i}} m_{i} + \sum_{j, n_{j}} n_{j}}

    :parameter float Q: The sum of all atomic charges.
    :parameter df: A dataframe with atomic charges.
    :type df: |pd.DataFrame|_
    :parameter dict constrain_dict: A dictionary with charge constrains.
    :return: A list of atom types with updated atomic charges.
    :rtype: |float|_
    """
    A = Q
    B = 0.0

    for key, value in constrain_dict.items():
        at_count = df.at[key, 'count']
        if value['func'] == np.add:
            A += at_count * value.arg
            B += at_count
        elif value['func'] == np.multiply:
            B += at_count * value.arg
    return A / B


def get_charge_constraints(constrain: str) -> Settings:
    r"""Take a string containing a set of interdependent charge constraints and translate
    it into a dictionary containing all arguments and operators.

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

    The currently supported operators are:

    ================= =======
     Operation         Symbol
    ================= =======
     addition_        ``+``
     multiplication_  ``*``
    ================= =======

    :parameter str item: A string with all charge constraints.
    :return: A Settings object with all charge constraints.
    :rtype: |plams.Settings|_

    .. _addition: https://docs.scipy.org/doc/numpy/reference/generated/numpy.add.html
    .. _multiplication: https://docs.scipy.org/doc/numpy/reference/generated/numpy.multiply.html
    """
    def _loop(i, operator_dict):
        for operator in operator_dict:
            split = i.split(operator)
            if len(split) == 2:
                return split[0], split[1], operator
        return split[0], 1.0, '*'

    operator_dict = {'+': np.add, '*': np.multiply}
    list_ = [i for i in constrain.split('=') if i]
    ret = Settings()
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
        ret[key].arg = arg
        ret[key].func = operator_dict[operator]
    return ret


def invert_ufunc(ufunc: Callable) -> Callable:
    """Invert a NumPyuniversal function.

    Addition will be turned into substraction and multiplication into division.

    :parameter ufunc: A NumPy universal function (ufunc).
    :type ufunc: |type|_
    :return: An inverted NumPy universal function.
    :rtype: |type|_
    """
    invert_dict = {
        np.add: np.subtract,
        np.multiply: np.divide,
    }

    try:
        return invert_dict[ufunc]
    except KeyError:
        raise KeyError("'{}' is not a supported ufunc. Supported ufuncs consist of: "
                       "'add' & 'multiply'".format(ufunc.__name__))
