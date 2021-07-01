"""Tests for :mod:`FOX.properties`."""

from __future__ import annotations

import copy
import types
import pickle
import inspect
import weakref
from typing import Callable, Any
from pathlib import Path

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


class TestABC:
    """Tests for :class:`FromResult`."""

    ATTR_DICT = {
        '__doc__': (str, type(None)),
        '__name__': str,
        '__qualname__': str,
        '__module__': str,
        '__annotations__': types.MappingProxyType,
        '__signature__': (inspect.Signature, type(None)),
        '__text_signature__': (str, type(None)),
        '__closure__': (tuple, type(None)),
        '__defaults__': (tuple, type(None)),
        '__globals__': types.MappingProxyType,
        '__kwdefaults__': types.MappingProxyType,
        '__call__': Callable,
    }
    FUNC_TUP = (_get_pressure, len, len.__call__, cp2k_mm, lambda n: n)

    @pytest.mark.parametrize("name,typ", ATTR_DICT.items(), ids=ATTR_DICT.keys())
    @pytest.mark.parametrize("func", FUNC_TUP)
    def test_attr(self, func: Callable[..., Any], name: str, typ: type | tuple[type, ...]) -> None:
        obj = _FromResultTest(func, func.__name__)
        assertion.isinstance(getattr(obj, name), typ)

    @pytest.mark.parametrize("func", FUNC_TUP)
    def test_attr2(self, func: Callable[..., Any]) -> None:
        obj = _FromResultTest(func, func.__name__)
        if hasattr(func, '__get__'):
            assertion.isinstance(obj.__get__, types.MethodWrapperType)
        if hasattr(func, '__code__'):
            assertion.isinstance(obj.__code__, types.CodeType)

    def test_misc(self) -> None:
        assertion.assert_(hash, get_pressure)
        assertion.eq(get_pressure, get_pressure)
        assertion.ne(get_pressure, get_bulk_modulus)

        assertion.eq(get_pressure, copy.deepcopy(get_pressure))
        assertion.eq(get_pressure, pickle.loads(pickle.dumps(get_pressure)))
        assertion.eq(get_pressure, weakref.ref(get_pressure)())


@pytest.mark.xfail(NP_15, reason="Precision issues in numpy 1.15")
@pytest.mark.parametrize('func', FROM_RESULT_DICT.keys())
class TestFromResult:
    def test_no_reduce(self, func: FromResult) -> None:
        prop = func.from_result(RESULT)
        ref = np.load(FROM_RESULT_DICT[func])
        np.testing.assert_allclose(prop, ref)

    def test_reduce_mean(self, func: FromResult) -> None:
        prop = func.from_result(RESULT, reduce='mean', axis=0)
        ref = np.load(FROM_RESULT_DICT[func]).mean(axis=0)
        np.testing.assert_allclose(prop, ref)

    def test_reduce_norm(self, func: FromResult) -> None:
        prop = func.from_result(RESULT, reduce=np.linalg.norm)
        ref = np.linalg.norm(np.load(FROM_RESULT_DICT[func]))
        np.testing.assert_allclose(prop, ref)


@pytest.mark.parametrize('name', GET_ATTR_TUP)
class TestGetAttr:
    def test_no_reduce(self, name: str) -> None:
        prop = get_attr(RESULT, name)
        ref = np.load(PATH / f'{name}.npy')
        np.testing.assert_allclose(prop, ref)

    def test_reduce_mean(self, name: str) -> None:
        prop = get_attr(RESULT, name, reduce='mean', axis=0)
        ref = np.load(PATH / f'{name}.npy').mean(axis=0)
        np.testing.assert_allclose(prop, ref)

    def test_reduce_norm(self, name: str) -> None:
        prop = get_attr(RESULT, name, reduce=np.linalg.norm)
        ref = np.linalg.norm(np.load(PATH / f'{name}.npy'))
        np.testing.assert_allclose(prop, ref)
