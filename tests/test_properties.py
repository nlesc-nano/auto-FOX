"""Tests for :mod:`FOX.properties`."""

import copy
import types
import pickle
import inspect
from typing import Callable, Any, List
from pathlib import Path
from weakref import WeakKeyDictionary

import pytest
import numpy as np
from assertionlib import assertion
from qmflows import cp2k_mm
from qmflows.packages.cp2k_mm import CP2KMM_Result

from FOX.properties import FromResult, get_pressure, get_bulk_modulus, get_attr
from FOX.properties.pressure import get_pressure as _get_pressure

# Fix for a precision issue in older numpy versions
NP_VERSION = tuple(int(i) for i in np.__version__.split('.')[:2])
NP_15 = NP_VERSION == (1, 15)

PATH = Path('tests') / 'test_files'

RESULT = CP2KMM_Result(
    None, None, 'md',
    dill_path=PATH / 'properties' / 'md' / 'md.002.dill',
    plams_dir=PATH / 'properties',
    work_dir=PATH / 'properties',
)

FROM_RESULT_DICT = {
    get_pressure: PATH / 'pressure.npy',
    get_bulk_modulus: PATH / 'bulk_modulus.npy',
}

GET_ATTR_TUP = (
    'volume',
    'lattice',
    'coordinates',
    'temperature',
    'forces',
)


class _FromResultTest(FromResult):
    def from_result(self, result, reduce=None, axis=None, **kwargs):
        raise NotImplementedError


def test_from_result_abc() -> None:
    """Tests for :class:`FOX.properties.CP2KMM_Result`."""
    assertion.assert_(hash, get_pressure)
    assertion.eq(get_pressure, get_pressure)
    assertion.ne(get_pressure, get_bulk_modulus)

    assertion.eq(get_pressure, copy.deepcopy(get_pressure))
    assertion.eq(get_pressure, pickle.loads(pickle.dumps(get_pressure)))

    NoneType = type(None)
    attr_dict = {
        '__doc__': (str, NoneType),
        '__name__': str,
        '__qualname__': str,
        '__module__': str,
        '__annotations__': types.MappingProxyType,
        '__signature__': (inspect.Signature, NoneType),
        '__text_signature__': (str, NoneType),
        '__closure__': (tuple, NoneType),
        '__defaults__': (tuple, NoneType),
        '__globals__': types.MappingProxyType,
        '__kwdefaults__': types.MappingProxyType,
        '__call__': Callable,
        '_cache': WeakKeyDictionary,
    }

    func_list: List[Callable[..., Any]] = [_get_pressure, len, len.__call__, cp2k_mm, lambda n: n]
    for func in func_list:
        obj = _FromResultTest(func, func.__name__)
        for name, typ in attr_dict.items():
            assertion.isinstance(getattr(obj, name), typ,
                                 message=f'func={func!r}, name={name!r}')

        if hasattr(func, '__get__'):
            assertion.isinstance(obj.__get__, types.MethodWrapperType,
                                 message=f'func={func!r}, name="__get__"')
        if hasattr(func, '__code__'):
            assertion.isinstance(obj.__code__, types.CodeType,
                                 message=f'func={func!r}, name="__code__"')


@pytest.mark.parametrize('func', FROM_RESULT_DICT.keys())
@pytest.mark.xfail(NP_15, reason="Precision issues in numpy 1.15")
def test_from_result(func: FromResult) -> None:
    """Tests for :class:`FOX.properties.CP2KMM_Result` subclasses."""
    prop1 = func.from_result(RESULT)
    ref1 = np.load(FROM_RESULT_DICT[func])
    np.testing.assert_allclose(prop1, ref1)

    prop2 = func.from_result(RESULT, reduce='mean', axis=0)
    ref2 = ref1.mean(axis=0)
    np.testing.assert_allclose(prop2, ref2)

    prop3 = func.from_result(RESULT, reduce=np.linalg.norm)
    ref3 = np.linalg.norm(ref1)
    np.testing.assert_allclose(prop3, ref3)


@pytest.mark.parametrize('name', GET_ATTR_TUP)
def test_get_attr(name: str) -> None:
    """Tests for :func:`FOX.properties.get_attr`."""
    prop1 = get_attr(RESULT, name)
    ref1 = np.load(PATH / f'{name}.npy')
    np.testing.assert_allclose(prop1, ref1)

    prop2 = get_attr(RESULT, name, reduce='mean', axis=0)
    ref2 = ref1.mean(axis=0)
    np.testing.assert_allclose(prop2, ref2)

    prop3 = get_attr(RESULT, name, reduce=np.linalg.norm)
    ref3 = np.linalg.norm(ref1)
    np.testing.assert_allclose(prop3, ref3)
