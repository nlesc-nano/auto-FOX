"""A module for testing :mod:`FOX.functions.charge_utils` ."""

import numpy as np
import pandas as pd
from assertionlib import assertion

from FOX.functions.charge_utils import update_charge

PARAM = pd.Series({
    'Cd': 0.9768,
    'Se': -0.9768,
    'O': -0.47041,
    'C': 0.4524,
    'H': 0.0
})
COUNT = pd.Series({'Cd': 68, 'Se': 55, 'C': 26, 'H': 26, 'O': 52})
PRM_MIN = pd.Series({"Cd": 0, "Se": -2, "O": -1, "C": -np.inf, "H": -np.inf})
PRM_MAX = pd.Series({"Cd": 2, "Se": 0, "O": 0, "C": np.inf, "H": np.inf})
EXCLUDE = {"H"}
ATOM_COEFS = (
    pd.Series({'Cd': 1}),
    pd.Series({"O": -4, "C": -2, "H": -2})
)


def _validate(atom: str, value: float, param: pd.Series) -> None:
    """Validate the moves of :func:`test_constraint_blocks`."""
    ident = f"{atom}={value}"
    assertion.isclose(param[atom], value, abs_tol=0.001, message=f'{atom} ({ident})')
    for min_, max_, (key, v) in zip(PRM_MIN, PRM_MAX, param.items()):
        assertion.le(min_, v, message=f'{key} ({ident})')
        assertion.ge(max_, v, message=f'{key} ({ident})')

    assertion.isclose((param * COUNT).sum(), 0, abs_tol=0.001, message=f"net charge ({ident})")
    for key in EXCLUDE:
        assertion.eq(param[key], PARAM[key], message=f'{key} ({ident})')

    idx1 = ATOM_COEFS[0].index
    idx2 = ATOM_COEFS[1].index
    v1 = (param[idx1] * ATOM_COEFS[0]).sum()
    v2 = (param[idx2] * ATOM_COEFS[1]).sum()
    assertion.isclose(v1, v2, abs_tol=0.001, message=f"{idx1} and {idx2} ({ident})")


def test_constraint_blocks() -> None:
    """Test updates for the following constraints: :math:`Cd = -2 * (2*O + C + H)`."""
    atoms = ("Cd", "Se", "C", "O")
    values = np.arange(0.5, 1.55, 0.05)
    iterator = ((at, PARAM[at] * i) for at in atoms for i in values)

    param = PARAM.copy()
    for atom, value in iterator:
        ex = update_charge(
            atom, value,
            param=param,
            count=COUNT,
            atom_coefs=ATOM_COEFS,
            prm_min=PRM_MIN,
            prm_max=PRM_MAX,
            exclude=EXCLUDE,
            net_charge=0,
        )
        if ex is not None:
            raise ex
        _validate(atom, value, param)
