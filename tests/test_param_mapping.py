"""A module for testing :mod:`FOX.armc.param_mapping`."""
import warnings
import pandas as pd
import numpy as np

from assertionlib import assertion

from FOX.armc import ParamMapping
from FOX.test_utils import validate_mapping

_DATA = {
    ('charge', 'charge', 'Cd'): 0.9768,
    ('charge', 'charge', 'Se'): -0.9768,
    ('charge', 'charge', 'O_1'): -0.4704,
    ('charge', 'charge', 'C_1'): 0.4524
}
_DF = pd.DataFrame({'param': _DATA})
_DF['count'] = [68, 55, 52, 26]
_DF['min'] = _DF['param'] - abs(_DF['param'] * 0.2)
_DF.at[('charge', 'charge', 'C_1'), 'min'] = -np.inf
_DF['max'] = _DF['param'] + abs(_DF['param'] * 0.2)
_DF.at[('charge', 'charge', 'C_1'), 'max'] = np.inf
PARAM = ParamMapping(_DF)


def test_mapping():
    """Test the :class:`~collections.abc.Mapping` implementation of :class:`ParamMapping`."""
    validate_mapping(PARAM, key_type=str, value_type=pd.Series)


def test_call():
    """Test :meth:`ParamMapping.__call__`."""
    param = PARAM.copy(deep=True)
    ref = sum(param['param'] * param['count'])

    for i in range(1000):
        ex = param()
        if isinstance(ex, Exception):
            warning = RuntimeWarning(f"iteration {i}: {ex}")
            warning.__cause__ = ex
            warnings.warn(warning)

        value = sum(param['param'] * param['count'])
        assertion.isclose(value, ref, abs_tol=0.001)

        try:
            assert (param['param'].iloc[:-1] <= param['max'].iloc[:-1]).all()
        except AssertionError as ex:
            msg = f"iteration{i}\n{pd.DataFrame({'param': param['param'], 'max': param['max']})}"
            raise AssertionError(msg).with_traceback(ex.__tarceback__)

        try:
            assert (param['param'].iloc[:-1] >= param['min'].iloc[:-1]).all()
        except AssertionError as ex:
            msg = f"iteration{i}\n{pd.DataFrame({'param': param['param'], 'min': param['min']})}"
            raise AssertionError(msg).with_traceback(ex.__tarceback__)
