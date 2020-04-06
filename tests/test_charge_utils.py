"""A module for testing :mod:`FOX.functions.charge_utils` ."""

import functools
from pathlib import Path

import numpy as np
from assertionlib import assertion

from FOX.functions.charge_utils import assign_constraints

PATH = Path('tests') / 'test_files'


def test_assign_constraints() -> None:
    """Test :func:`assign_constraints`."""
    constraints = [
        'H < 1',
        'C >2.0',
        '3.0> N',
        '4<O',
        '1 < F< 2.0',
        '2 > P >1.0',
        'S == 2 * Cl == 0.5*Br == 1* I',
        '1 < H H < 2'
    ]

    partial: dict
    extremite, partial = assign_constraints(constraints)  # type: ignore

    extremite_ref = {
        ('H', 'max'): 1.0,
        ('C', 'min'): 2.0,
        ('N', 'max'): 3.0,
        ('O', 'min'): 4.0,
        ('F', 'min'): 1.0,
        ('F', 'max'): 2.0,
        ('P', 'max'): 2.0,
        ('P', 'min'): 1.0,
        ('H H', 'min'): 1.0,
        ('H H', 'max'): 2.0
    }
    assertion.eq(extremite, extremite_ref)

    partial_ref = {'S': functools.partial(np.multiply, 1.0),
                   'Cl': functools.partial(np.multiply, 2.0),
                   'Br': functools.partial(np.multiply, 0.5),
                   'I': functools.partial(np.multiply, 1.0)}

    assertion.eq(partial.keys(), partial_ref.keys())
    iterator = ((v, partial_ref[k]) for k, v in partial.items())
    for v1, v2 in iterator:
        assertion.isinstance(v1, functools.partial)
        assertion.is_(v1.func, v2.func)
        assertion.eq(v1.args, v2.args)
        assertion.eq(v1.keywords, v2.keywords)
