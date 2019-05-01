""" A module with functions related to manipulating atomic charges. """

__all__ = ['update_charge', 'get_charge_constraints']

import numpy as np

from scm.plams import Settings


def update_charge(at, charge, df, constrain_dict={}, exclude=[]):
    """ Set the atomic charge of **at** to **charge**.
    The atomic charges in **df** are furthermore exposed to the following constraints:

        * The total charge remains constant.
        * Optional constraints specified in **constrain_dict**
          (see :func:`.get_charge_constraints`).

    Performs an inplace update of the *charge* column in **df**.

    :parameter str at: An atom type such as *Se*, *Cd* or *OG2D2*.
    :parameter float charge: The new charge associated with **at**.
    :parameter df: A dataframe with atomic charges.
    :type df: |pd.DataFrame|_
    :parameter dict constrain_dict: A dictionary with charge constrains.
    :parameter list exclude: A list of excluded atom types. The charges of atoms with matching atom
        types will not be altered.
    """
    net_charge = df['charge'].sum()
    df.loc[df['atom type'] == at, 'charge'] = charge

    if at in constrain_dict or not constrain_dict:
        exclude += update_constrained_charge(at, df, constrain_dict)
        update_unconstrained_charge(net_charge, df, exclude)
    else:
        update_unconstrained_charge(net_charge, df, exclude)
        exclude += update_constrained_charge(at, df, constrain_dict)
        update_constrained_charge(at, df, constrain_dict)


def update_constrained_charge(at, df, constrain_dict={}):
    """ Perform a constrained update of atomic charges.
    Performs an inplace update of the *charge* column in **df**.

    :parameter str at: An atom type such as *Se*, *Cd* or *OG2D2*.
    :parameter df: A dataframe with atomic charges.
    :type df: |pd.DataFrame|_
    :parameter dict constrain_dict: A dictionary with charge constrains.
    :return: A list of atom types with updated atomic charges.
    :rtype: |list|_ [|str|_]
    """
    charge = df.loc[df['atom type'] == at, 'charge'].iloc[0]
    exclude = []
    func1 = constrain_dict[at].func
    i = constrain_dict[at].arg

    # Perform a constrained charge update
    for at2, values in constrain_dict[at].items():
        exclude.append(at2)
        if at2 == at:
            continue

        # Slice the dataframe
        df_slice = df.loc[df['atom type'] == at2, 'charge']

        # Unpack values
        func2 = invert_ufunc(values.func)
        j = values.arg

        # Update the charges
        func1(charge, i, out=df_slice)
        func2(charge, j, out=df_slice)

    return exclude


def find_q(df, Q=0.0, constrain_dict={}):
    r""" Calculates the atomic charge :math:`q` given the total charge :math:`Q`. Atom subsets are
    denoted by :math:`m` & :math:`n`, with :math:`a` & :math:`b` being subset-dependent constants.

    .. math::

        Q = \sum_{i, m_{i}} m_{i} * (q * a_{i}) + \sum_{j, n_{j}} n_{j} * (q + b_{j})

        q = \frac{\sum_{j, n_{j}} n_{j} * (q * b_{j})}
            {\sum_{i, m_{i}} m_{i} * (q * a_{i}) + \sum_{j, n_{j}} n_{j}}

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
        at_count = len(df.loc[df['atom type'] == key])
        if value.ufunc == np.add:
            A += at_count * value.arg
            B += at_count
        elif value.ufunc == np.multiply:
            B += at_count * value.arg

    return A / B


def update_unconstrained_charge(net_charge, df, exclude=[]):
    """ Perform an unconstrained update of atomic charges.
    Performs an inplace update of the *charge* column in **df**.

    :parameter float net_charge: The total charge of the system.
    :parameter df: A dataframe with atomic charges.
    :type df: |pd.DataFrame|_
    :parameter dict constrain_dict: A list of atom types whose atomic charges should not be updated.
    """
    include = np.array([i not in exclude for i in df['atom type']])
    i = net_charge - df.loc[~include, 'charge'].sum()
    i /= df.loc[include, 'charge'].sum()
    df.loc[include, 'charge'] *= i


def get_charge_constraints(constrain):
    """ Take a string containing a set of interdependent charge constraints and translate
    it into a dictionary containing all arguments and operators.

    An example where :math:`q_{Cd} = -\\frac{q_{O}}{0.5} = -q_{Se}`:

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

    Another example where the following (nonensical) constrain is applied:
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

    operator_dict = {
        '+': np.add, '/': np.divide, '**': np.power, '*': np.multiply, '-': np.subtract
    }

    list_ = [i for i in constrain.split('=') if i]
    ret = Settings()
    for i in list_:
        # Seperate the operator from its arguments
        arg1, arg2, operator = _loop(i, operator_dict)

        # Identify keys and values
        try:
            arg = float(arg1)
        except ValueError:
            arg = float(arg2)
        key = arg

        # Prepare and return the arguments and operators
        ret[key].arg = arg
        ret[key].func = operator_dict[operator]
    return ret


def invert_ufunc(ufunc):
    """ Invert a universal function, turning addition into substraction,
    multiplication into division and exponentiation into recipropal exponentiation.

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
                       "'add' & 'multiply' ".format(ufunc.__name__))


def recipropal_power(x1, x2, **kwarg):
    """ First array elements raised to powers of the recipropal from second array, element-wise.

    :parameter x1: The bases.
    :parameter x2: The exponents.
    :return: :math:`{x_{1}}^{1/x_{2}}`
    """
    return np.power(x1, 1/x2, **kwarg)
