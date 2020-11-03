"""A module for testing :mod:`FOX.armc.param_mapping`."""
import warnings
import pandas as pd

from assertionlib import assertion

from FOX.armc import ParamMapping
from FOX.testing_utils import validate_mapping

_DATA = {
    ('charge', 'charge', 'C'): 0.4524,
    ('charge', 'charge', 'Cd'): 0.9768,
    ('charge', 'charge', 'H'): 0.0,
    ('charge', 'charge', 'O'): -0.4704,
    ('charge', 'charge', 'Se'): -0.9768,
    ('lj', 'sigma', 'C'): 0.4524,
    ('lj', 'sigma', 'Cd'): 0.9768,
    ('lj', 'sigma', 'H'): 0.0,
    ('lj', 'sigma', 'O'): -0.4704,
    ('lj', 'sigma', 'Se'): -0.9768,
}

_DF = pd.DataFrame({'param': _DATA})
_DF['count'] = 2 * [26, 68, 26, 52, 55]
_DF['min'] = _DF['param'] - 0.5 * abs(_DF['param'])
_DF['max'] = _DF['param'] + 0.5 * abs(_DF['param'])
_DF['constant'] = 2 * [False, False, True, False, False]

_CONSTRAINTS = {
    ('charge', 'charge'): [{'Cd': 1.0}, {'O': -4.0, 'C': -2.0, 'H': -2.0}]
}

PARAM = ParamMapping(_DF, constraints=_CONSTRAINTS)


def test_mapping():
    """Test the :class:`~collections.abc.Mapping` implementation of :class:`ParamMapping`."""
    validate_mapping(PARAM, key_type=str, value_type=(pd.DataFrame, pd.Series))


def test_call():
    """Test :meth:`ParamMapping.__call__`."""
    param = PARAM.copy(deep=True)
    ref = sum(param['param'][0].loc['charge'] * param['count'].loc['charge'])

    failed_iterations = []
    for i in range(1000):
        ex = param()
        if isinstance(ex, Exception):
            failed_iterations.append(i)
            ex_backup = ex

        value = sum(param['param'][0].loc['charge'] * param['count'].loc['charge'])
        assertion.isclose(value, ref, abs_tol=0.001)

        try:
            assert (param['param'][0] <= param['max']).all()
        except AssertionError as ex:
            df = pd.DataFrame({'param': param['param'][0], 'max': param['max']})
            msg = f"iteration{i}\n{df.round(2)}"
            raise AssertionError(msg) from ex

        try:
            assert (param['param'][0] >= param['min']).all()
        except AssertionError as ex:
            df = pd.DataFrame({'param': param['param'][0], 'max': param['min']})
            msg = f"iteration{i}\n{df.round(2)}"
            raise AssertionError(msg) from ex

    if failed_iterations:
        warning = RuntimeWarning("Failed to conserve the net charge in the "
                                 f"following iterations: {failed_iterations!r}")
        warning.__cause__ = ex_backup
        warnings.warn(warning)
