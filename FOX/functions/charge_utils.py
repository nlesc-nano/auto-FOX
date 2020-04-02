"""A module with functions related to manipulating atomic charges."""

import functools
from types import MappingProxyType
from itertools import repeat
from typing import (
    Hashable, Optional, Collection, Mapping, Container, List, Dict, Union, Iterable, Tuple, Any,
    SupportsFloat
)

import numpy as np
import pandas as pd

__all__ = ['update_charge']

ConstrainDict = Mapping[Hashable, functools.partial]


class ChargeError(ValueError):
    """A :exc:`ValueError` subclass for charge-related errors."""

    reference: float
    value: float
    tol: Hashable

    def __init__(self, *args: Any, reference: Optional[SupportsFloat] = None,
                 value: Optional[SupportsFloat] = None,
                 tol: SupportsFloat = 0.001) -> None:
        """Initialize an instance."""
        super().__init__(*args)
        self.reference = float(reference)
        self.value = float(value)
        self.tol = float(tol)


def get_net_charge(param: pd.Series, count: pd.Series,
                   index: Optional[Collection] = None) -> float:
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
    index_ = slice(None) if index is None else index
    ret = param[index_] * count[index_]
    return ret.sum()


def update_charge(atom: str, value: float, param: pd.Series, count: pd.Series,
                  constrain_dict: Optional[ConstrainDict] = None,
                  prm_min: Optional[Iterable[float]] = None,
                  prm_max: Optional[Iterable[float]] = None,
                  net_charge: Optional[float] = None) -> Optional[ChargeError]:
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
    param_backup = param.copy()
    param[atom] = value

    if constrain_dict is None or atom in constrain_dict:
        exclude = constrained_update(atom, param, constrain_dict)
    else:
        exclude = {atom}

    if net_charge is not None:
        try:
            unconstrained_update(net_charge, param, count,
                                 prm_min=prm_min,
                                 prm_max=prm_max,
                                 exclude=exclude)
        except ChargeError as ex:
            param[:] = param_backup
            return ex


def constrained_update(at1: str, param: pd.Series,
                       constrain_dict: Optional[ConstrainDict] = None) -> List[str]:
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
    charge = param[at1]
    exclude = [at1]
    if constrain_dict is None:
        return exclude

    # Perform a constrained charge update
    func1 = invert_partial_ufunc(constrain_dict[at1])
    for at2, func2 in constrain_dict.items():
        if at2 == at1:
            continue
        exclude.add(at2)

        # Update the charges
        param[at2] = func2(func1(charge))

    return exclude


def unconstrained_update(net_charge: float, param: pd.Series, count: pd.Series,
                         prm_min: Optional[Iterable[float]] = None,
                         prm_max: Optional[Iterable[float]] = None,
                         exclude: Optional[Container[str]] = None) -> None:
    """Perform an unconstrained update of atomic charges."""
    exclude = exclude or ()
    include = np.array([i not in exclude for i in param.keys()])
    if not include.any():
        return

    v_new = v_new_clip = 0
    i = net_charge - get_net_charge(param, count, ~include)
    i /= get_net_charge(param, count, include)

    min_iter = prm_min if prm_min is not None else repeat(-np.inf)
    max_iter = prm_max if prm_max is not None else repeat(np.inf)
    iterator = enumerate(zip(param.items(), min_iter, max_iter))

    for j, ((atom, prm), prm_min, prm_max) in iterator:
        if atom in exclude:
            continue
        elif v_new != v_new_clip:
            i = net_charge - get_net_charge(param, count, ~include)
            i /= get_net_charge(param, count, include)

        v_new = i * prm
        v_new_clip = np.clip(v_new, prm_min, prm_max)
        param[atom] = v_new_clip
        include[j] = False

    net_charge_new = get_net_charge(param, count)
    if abs(net_charge - net_charge_new) > 0.001:
        msg = f"Failed to conserve the net charge ({net_charge:.4f}): {net_charge_new:.4f}"
        raise ChargeError(msg, reference=net_charge, value=net_charge_new, tol=0.001)


def invert_partial_ufunc(ufunc: functools.partial) -> functools.partial:
    """Invert a NumPy universal function embedded within a :class:`functools.partial` instance."""
    func = ufunc.func
    x2 = ufunc.args[0]
    return functools.partial(func, x2**-1)


def assign_constraints(constraints: Union[str, Iterable[str]],
                       param: pd.DataFrame, idx_key: str) -> None:
    operator_set = {'>', '<', '*', '=='}

    # Parse integers and floats
    if isinstance(constraints, str):
        constraints = [constraints]

    constrain_list = []
    for item in constraints:
        for i in operator_set:  # Sanitize all operators; ensure they are surrounded by spaces
            item = item.replace(i, f'~{i}~')

        item_list = [i.strip().rstrip() for i in item.split('~')]
        if len(item_list) == 1:
            continue

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
        if (idx_key, at) not in param.index:
            raise KeyError(f"Assigning invalid constraint '({' '.join(str(i) for i in constrain)})'"
                           f"; no parameter available of type ({repr(idx_key)}, {repr(at)})")
        param.at[(idx_key, at), operator] = value


def _find_float(iterable: Tuple[str, str]) -> Tuple[str, float]:
    """Take an iterable of 2 strings and identify which element can be converted into a float."""
    try:
        i, j = iterable
    except ValueError:
        return iterable[0], 1.0

    try:
        return j, float(i)
    except ValueError:
        return i, float(j)


def _eq_constraints(constrain_: list, param: pd.DataFrame, idx_key: str) -> None:
    """Parse :math:`a = i * b`-type constraints."""
    constrain_dict: Dict[str, functools.partial] = {}
    constrain = ''.join(str(i) for i in constrain_).split('==')
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
    for k in constrain_dict:
        if (idx_key, k) not in param.index:
            raise KeyError(f"Assigning invalid constraint '({' '.join(str(i) for i in constrain_)})"
                           f"'; no parameter available of type ({repr(idx_key)}, {repr(at)})")
    for at, _ in param.loc[idx_key].iterrows():
        param.at[(idx_key, at), 'constraints'] = constrain_dict
