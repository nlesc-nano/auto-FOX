"""A module with functions related to manipulating atomic charges.

Index
-----
.. currentmodule:: FOX.functions.charge_utils
.. autosummary::
    update_charge

API
---
.. autofunction:: update_charge

"""

from __future__ import annotations

from itertools import chain
from collections.abc import Hashable, Collection, Container
from typing import SupportsFloat, TypeVar, Generic

import numpy as np
import pandas as pd

from nanoutils import TypedDict

__all__ = ['update_charge']

T = TypeVar('T')
KT = TypeVar('KT', bound=Hashable)
ST = TypeVar('ST', bound='ChargeError')


class _StateDict(TypedDict):
    """A dictionary representing the keyword-only arguments of :exc:`ChargeError`."""

    reference: None | float
    value: None | float
    tol: None | float


class ChargeError(ValueError, Generic[T]):
    """A :exc:`ValueError` subclass for charge-related errors."""

    __slots__ = ('__weakref__', 'reference', 'value', 'tol')

    reference: None | float
    value: None | float
    tol: None | float
    args: tuple[T, ...]

    def __init__(
        self,
        *args: T,
        reference: None | SupportsFloat = None,
        value: None | SupportsFloat = None,
        tol: None | SupportsFloat = 0.001,
    ) -> None:
        """Initialize an instance."""
        super().__init__(*args)
        self.reference = float(reference) if reference is not None else None
        self.value = float(value) if value is not None else None
        self.tol = float(tol) if tol is not None else None

    def __reduce__(self: ST) -> tuple[type[ST], tuple[T, ...], _StateDict]:
        """Helper for :mod:`pickle`."""
        cls = type(self)
        kwargs = _StateDict(reference=self.reference, value=self.value, tol=self.tol)
        return cls, self.args, kwargs

    def __setstate__(self, state: _StateDict) -> None:
        """Helper for :meth:`pickle`; handles the setting of keyword arguments."""
        for k, v in state.items():
            setattr(self, k, v)


def get_net_charge(
    param: pd.Series,
    count: pd.Series,
    index: None | Collection[Hashable] = None,
) -> float:
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
    idx = slice(None) if index is None else index
    ret = param[idx] * count[idx]
    return ret.sum()


def update_charge(
    atom: KT,
    value: float,
    param: pd.Series,
    count: pd.Series,
    atom_coefs: None | Collection[pd.Series] = None,
    prm_min: None | pd.Series = None,
    prm_max: None | pd.Series = None,
    exclude: None | Collection[KT] = None,
    net_charge: None | float = None,
) -> None | ChargeError:
    """Set the atomic charge of **at** equal to **charge**.

    The atomic charges in **df** are furthermore exposed to the following constraints:

        * The total charge remains constant.param
        * Optional constraints specified in **constrain_dict**
          (see :func:`.update_constrained_charge`).

    Performs an inplace update of the *param* column in **df**.

    """
    if net_charge is None:
        min_value = prm_min.at[atom] if prm_min is not None else -np.inf
        max_value = prm_max.at[atom] if prm_max is not None else np.inf
        param.at[atom] = np.clip(value, min_value, max_value)
        return None

    if exclude is None:
        exclude_set = set()
    else:
        exclude_set = set(exclude)

    if atom_coefs is not None:
        try:
            constrained_update(atom, value, param, count, atom_coefs, prm_min, prm_max, exclude_set)
        except ChargeError as ex:
            return ex
        exclude_set.update(chain.from_iterable(s.index for s in atom_coefs))

    exclude_set.add(atom)
    param[atom] = value
    unconstrained_update(
        net_charge, param, count, prm_min=prm_min,
        prm_max=prm_max, exclude=exclude_set,
    )

    try:
        _check_net_charge(param, count, net_charge)
    except ChargeError as ex:
        return ex
    else:
        return None


def constrained_update(
    atom: KT,
    value: float,
    param: pd.Series,
    count: pd.Series,
    atom_coefs: Collection[pd.Series],
    param_min: pd.Series,
    param_max: pd.Series,
    exclude: None | set[KT] = None,
) -> None:
    """Perform a constrained update of atomic charges.

    Performs an inplace update **param**.

    Parameters
    ----------
    atom : str
        An atom type such as ``"Se"``, ``"Cd"`` or ``"OG2D2"``.
    value : :class:`float`
        The value to be assigned to **atom**.
    param : :class:`pandas.Series`
        A Series with parameter values.
    count : :class:`pandas.Series`
        A Series with the number of atoms per parameter.
    atom_coefs : :class:`Collection[pandas.Series]<typing.Collection>`
        A collection of (disjoint) coefficient Series.
    param_min : :class:`pandas.Series`
        A Series with the minimum values for all parameters in **param**.
    param_min : :class:`pandas.Series`
        A Series with the maximum values for all parameters in **param**.


    :rtype: :data:`None`

    """
    exclude_set = _update_1st_charge(atom, value, param, param_min, param_max, exclude)

    # Identify the charge of the moved atoms set of constraints
    for ref_coef in atom_coefs:
        if atom in ref_coef.index:
            break
    else:
        return None
    idx = ref_coef.index
    idx_ref = idx.tolist()
    net_charge: float = (param.loc[idx] * ref_coef.loc[idx]).sum()

    df = pd.DataFrame({'param': param, 'count': 0,
                       'prm_min': param_min, 'prm_max': param_max})

    # Update the charge of all other charge-constraint blocks
    coef_iterator = (coef for coef in atom_coefs if coef is not ref_coef)
    with pd.option_context('mode.chained_assignment', None):
        for coef in coef_iterator:
            # Identify the to-be considered slice
            idx = coef.index
            idx_ref += idx.tolist()

            # Update the charges
            df_kwargs = df.loc[idx]
            df_kwargs["count"] = coef
            unconstrained_update(net_charge, exclude=exclude_set, **df_kwargs)
            param.loc[idx] = df_kwargs['param']
            exclude_set.update(idx.intersection(idx_ref))


def _update_1st_charge(
    atom: KT,
    value: float,
    param: pd.Series,
    param_min: pd.Series,
    param_max: pd.Series,
    exclude: None | set[KT] = None,
) -> set[KT]:
    """Helper function for :func:`constrained_update`."""
    if exclude is not None:
        exclude_set = exclude.copy()
        idx_ = pd.Series(True, index=param.index)
        idx_[list(exclude_set)] = False
    else:
        exclude_set = set()
        idx_ = slice(None)
    exclude_set.add(atom)

    value_old = param[atom]
    x = value / value_old

    # Scale all parameters (correct them afterwards)
    param.loc[idx_] *= x
    param.clip(param_min, param_max, inplace=True)
    return exclude_set


def unconstrained_update(
    net_charge: float,
    param: pd.Series,
    count: pd.Series,
    prm_min: None | pd.Series = None,
    prm_max: None | pd.Series = None,
    exclude: None | Container[Hashable] = None,
) -> None:
    """Perform an unconstrained update of atomic charges."""
    if exclude is None:
        include = param.astype(bool, copy=True)
        include.loc[:] = True
    else:
        include = pd.Series([i not in exclude for i in param.keys()], index=param.index)
    if not include.any():
        return

    # Identify the multplicative factor that yields a net-neutral charge
    i = net_charge - get_net_charge(param, count, ~include)
    i /= get_net_charge(param, count, include)

    # Define the minimum and maximum values
    s_min = prm_min if prm_min is not None else -np.inf
    s_max = prm_max if prm_max is not None else np.inf

    # Identify which parameters are closest to their extreme values
    s = param * i
    s_clip = np.clip(s, s_min, s_max).loc[include]
    s_delta = abs(s_clip - s.loc[include])
    s_delta.sort_values(ascending=False, inplace=True)

    start = -len(s_delta) + 1
    for j, atom in enumerate(s_delta.index, start=start):
        param[atom] = s_clip[atom]
        include[atom] = False

        if j and s_clip[atom] != s[atom]:
            i = net_charge - get_net_charge(param, count, ~include)
            i /= get_net_charge(param, count, include)

            s = param * i
            s_clip = np.clip(s, s_min, s_max).loc[include]


def _check_net_charge(
    param: pd.Series,
    count: pd.Series,
    net_charge: float,
    tolerance: float = 0.001,
) -> None:
    """Check if the net charge is actually conserved."""
    net_charge_new = get_net_charge(param, count)
    condition = abs(net_charge - net_charge_new) > tolerance

    if not condition:
        return

    raise ChargeError(
        f"Failed to conserve the net charge: ref = {net_charge:.4f}; {net_charge_new:.4f} != ref",
        reference=net_charge, value=net_charge_new, tol=tolerance
    )
