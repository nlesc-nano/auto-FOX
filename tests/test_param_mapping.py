"""A module for testing :mod:`FOX.armc.param_mapping`."""

import warnings
import numpy as np
import pandas as pd

import pytest
from assertionlib import assertion
from FOX.armc import ParamMapping

_DATA = {
    ('charge', 'charge', 'C1'): 0.4524,
    ('charge', 'charge', 'C2'): 0.2262,
    ('charge', 'charge', 'Cd'): 0.9768,
    ('charge', 'charge', 'H'): 0.0,
    ('charge', 'charge', 'O'): -0.4704,
    ('charge', 'charge', 'Se'): -0.9768,
    ('lj', 'sigma', 'C1'): 0.4524,
    ('lj', 'sigma', 'C2'): 0.4524,
    ('lj', 'sigma', 'Cd'): 0.9768,
    ('lj', 'sigma', 'H'): 0.0,
    ('lj', 'sigma', 'O'): -0.4704,
    ('lj', 'sigma', 'Se'): -0.9768,
}

_DF = pd.DataFrame({'param': _DATA})
_DF['count'] = 2 * [26, 0, 68, 26, 52, 55]
_DF['min'] = _DF['param'] - 0.5 * abs(_DF['param'])
_DF['max'] = _DF['param'] + 0.5 * abs(_DF['param'])
_DF['frozen'] = 2 * [False, False, False, True, False, False]

_CONSTRAINTS1 = {
    ('charge', 'charge'): [{'Cd': 1.0}, {'O': -4.0, 'C1': -2.0, 'H': -2.0}]
}

PARAM1 = ParamMapping(_DF, constraints=_CONSTRAINTS1)

PARAM2 = PARAM1.copy(deep=True)
PARAM2.param[1] = PARAM2.param[0].copy()
for (_, k), v in PARAM2.metadata.items():
    PARAM2.metadata[1, k] = v.copy()
PARAM2.metadata.at[("charge", "charge", "C1"), (1, "count")] = 0
PARAM2.metadata.at[("charge", "charge", "C2"), (1, "count")] = 26 * 2
PARAM2.metadata.at[("charge", "charge", "H"), (1, "count")] = 26 * 3
PARAM2.constraints["charge", "charge"] += (pd.Series({'O': -4.0, 'C2': -4.0, 'H': -6.0}),)
PARAM2._net_charge = [0, 0]


@pytest.mark.parametrize("input_param", [PARAM1, PARAM2], ids=[0, 1])
def test_call(input_param: ParamMapping):
    """Test :meth:`ParamMapping.__call__`."""
    param = input_param.copy(deep=True)
    ref: pd.Series = np.sum(
        param.param.loc['charge', :] *
        param.metadata.swaplevel(0, 1, axis=1).loc['charge', 'count']
    )

    failed_iterations = []
    for i in range(1000):
        ex = param()
        if isinstance(ex, Exception):
            failed_iterations.append(i)
            ex_backup = ex

        value: pd.Series = np.sum(
            param.param.loc['charge', :] *
            param.metadata.swaplevel(0, 1, axis=1).loc['charge', 'count']
        )
        np.testing.assert_allclose(value, ref, atol=0.001, err_msg=f"iteration{i}")

        try:
            assertion.le(
                param.param,
                param.metadata.swaplevel(0, 1, axis=1)['max'],
                post_process=np.all,
                message=f"iteration{i}",
            )
        except AssertionError as ex:
            df = param.param.copy()
            df.columns = pd.MultiIndex.from_tuples([(j, "param") for j in df.columns])
            for j in param.metadata.columns.levels[0]:
                df[j, 'max'] = param.metadata[j, "max"]
            df.sort_index(axis=1, inplace=True)
            msg = f"iteration{i}\n{df.round(2)}"
            raise AssertionError(msg) from ex

        try:
            assertion.ge(
                param.param,
                param.metadata.swaplevel(0, 1, axis=1)['min'],
                post_process=np.all,
                message=f"iteration{i}",
            )
        except AssertionError as ex:
            df = param.param.copy()
            df.columns = pd.MultiIndex.from_tuples([(j, "param") for j in df.columns])
            for j in param.metadata.columns.levels[0]:
                df[j, 'max'] = param.metadata[j, "min"]
            df.sort_index(axis=1, inplace=True)
            msg = f"iteration{i}\n{df.round(2)}"
            raise AssertionError(msg) from ex

    if failed_iterations:
        warning = RuntimeWarning("Failed to conserve the net charge during "
                                 f"{len(failed_iterations)} iterations: {failed_iterations!r}")
        warning.__cause__ = ex_backup
        warnings.warn(warning)
