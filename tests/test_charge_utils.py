"""A module for testing :mod:`FOX.functions.charge_utils` ."""

from pathlib import Path
import functools

import numpy as np
import pandas as pd
from assertionlib import assertion

from FOX.functions.charge_utils import assign_constraints

PATH = Path('tests') / 'test_files'


def test_assign_constraints() -> None:
    """Test :func:`assign_constraints`."""
    df = pd.DataFrame(index=pd.MultiIndex.from_product(
        [['key'], ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'H H']]
    ))
    df['constraints'] = None
    df['min'] = -np.inf
    df['max'] = np.inf

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

    assign_constraints(constraints, df, 'key')

    inf = np.inf
    min_ar = np.array([-inf, 2.0, -inf, 4.0, 1.0, 1.0, -inf, -inf, -inf, -inf, 1.0])
    max_ar = np.array([1.0, inf, 3.0, inf, 2.0, 2.0, inf, inf, inf, inf, 2.0])
    np.testing.assert_allclose(df['min'], min_ar)
    np.testing.assert_allclose(df['max'], max_ar)

    partial_dict = df.loc[('key', 'H'), 'constraints']
    partial_ref = {'S': functools.partial(np.multiply, 1.0),
                   'Cl': functools.partial(np.multiply, 2.0),
                   'Br': functools.partial(np.multiply, 0.5),
                   'I': functools.partial(np.multiply, 1.0)}

    for k, v1 in partial_dict.items():
        v2 = partial_ref[k]
        assertion.isinstance(v1, functools.partial)
        assertion.is_(v1.func, v2.func)
        assertion.eq(v1.args, v2.args)

    assertion.assert_(assign_constraints, ['bob == bub'], df, 'key', exception=KeyError)
    assertion.assert_(assign_constraints, ['bob > 1.0'], df, 'key', exception=KeyError)
