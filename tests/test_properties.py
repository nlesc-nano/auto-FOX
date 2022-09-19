"""Tests for :mod:`FOX.properties`."""

from __future__ import annotations

import copy
import pickle
import weakref
import inspect
from typing import Callable
from pathlib import Path

import pytest
import numpy as np
from assertionlib import assertion
from qmflows.packages import CP2KMM_Result

from FOX.properties import FromResult, get_pressure, get_bulk_modulus, get_attr

# Fix for a precision issue in older numpy versions
NP_VERSION = tuple(int(i) for i in np.__version__.split('.')[:2])
NP_15 = NP_VERSION == (1, 15)

PATH = Path('tests') / 'test_files'

RESULT = CP2KMM_Result(
    None, None, 'md',
    dill_path=PATH / 'properties' / 'md.002.dill',
    plams_dir=PATH / 'properties',
    work_dir=PATH / 'properties',
)


class TestFromResultType:
    """Tests for :class:`FromResult`."""

    @pytest.mark.parametrize("func", [
        lambda i: pickle.loads(pickle.dumps(i)),
        lambda i: weakref.ref(i)(),
        copy.copy,
        copy.deepcopy,
    ], ids=["pickle", "weakref", "copy", "deepcopy"])
    def test_eq(self, func: Callable[[FromResult], FromResult]) -> None:
        obj = func(get_pressure)
        assertion.eq(obj, get_pressure)
        assertion.is_(obj, get_pressure)
        assertion.ne(obj, 1)
        assertion.ne(obj, get_bulk_modulus)

    def test_hash(self) -> None:
        assertion.assert_(hash, get_pressure)

    def test_dir(self) -> None:
        ref = object.__dir__(get_pressure) + dir(get_pressure.__call__)
        ref_sorted = sorted(set(ref))

        assertion.eq(ref_sorted, dir(get_pressure))
        for name in ref_sorted:
            assertion.hasattr(get_pressure, name)

    @pytest.mark.parametrize(
        "name", ["__call__", "__code__", "bob", "__module__", "__annotations__"]
    )
    def test_settatr(self, name: str) -> None:
        with pytest.raises(AttributeError):
            setattr(get_pressure, name, None)

    @pytest.mark.parametrize(
        "name", ["__call__", "__code__", "bob", "__module__", "__annotations__"]
    )
    def test_delattr(self, name: str) -> None:
        with pytest.raises(AttributeError):
            delattr(get_pressure, name)

    def test_getattr(self) -> None:
        assertion.truth(get_pressure.__call__)
        assertion.is_(get_pressure.__code__, get_pressure.__call__.__code__)
        with pytest.raises(AttributeError):
            get_pressure.bob

    @pytest.mark.parametrize("func", [repr, str], ids=["repr", "str"])
    def test_repr(self, func: Callable[[FromResult], str]) -> None:
        ref = (
            "<FromResult instance FOX.properties.pressure.get_pressure("
            "forces: 'ArrayLike', "
            "coords: 'ArrayLike', "
            "volume: 'ArrayLike', "
            "temp: 'float' = 298.15, "
            "*, "
            "forces_unit: 'str' = 'ha/bohr', "
            "coords_unit: 'str' = 'bohr', "
            "volume_unit: 'str' = 'bohr', "
            "return_unit: 'str' = 'ha/bohr^3'"
            ") -> 'NDArray[f8]'>"
        )
        assertion.str_eq(get_pressure, ref, str_converter=func)

    def test_signature(self) -> None:
        sgn1 = inspect.signature(get_pressure)
        sgn2 = inspect.signature(get_pressure.__call__)
        assertion.eq(sgn1, sgn2)

    @pytest.mark.xfail(NP_15, reason="Precision issues in numpy 1.15")
    @pytest.mark.parametrize('func,ref', [
        (get_pressure, np.load(PATH / 'pressure.npy')),
        (get_bulk_modulus, np.load(PATH / 'bulk_modulus.npy')),
    ], ids=["get_pressure", "get_bulk_modulus"])
    class TestFromResult:
        def test_no_reduce(self, func: FromResult, ref: np.ndarray) -> None:
            prop = func.from_result(RESULT)
            np.testing.assert_allclose(prop, ref, rtol=1e-06)

        def test_reduce_mean(self, func: FromResult, ref: np.ndarray) -> None:
            prop = func.from_result(RESULT, reduce='mean', axis=0)
            np.testing.assert_allclose(prop, ref.mean(axis=0))

        def test_reduce_norm(self, func: FromResult, ref: np.ndarray) -> None:
            prop = func.from_result(RESULT, reduce=np.linalg.norm)
            np.testing.assert_allclose(prop, np.linalg.norm(ref))

    @pytest.mark.parametrize('name,ref', [
        ('volume', np.load(PATH / 'volume.npy')),
        ('lattice', np.load(PATH / 'lattice.npy')),
        ('coordinates', np.load(PATH / 'coordinates.npy')),
        ('temperature', np.load(PATH / 'temperature.npy')),
        ('forces', np.load(PATH / 'forces.npy')),
    ], ids=['volume', 'lattice', 'coordinates', 'temperature', 'forces'])
    class TestGetAttr:
        def test_no_reduce(self, name: str, ref: np.ndarray) -> None:
            prop = get_attr(RESULT, name)
            np.testing.assert_allclose(prop, ref)

        def test_reduce_mean(self, name: str, ref: np.ndarray) -> None:
            prop = get_attr(RESULT, name, reduce='mean', axis=0)
            np.testing.assert_allclose(prop, ref.mean(axis=0))

        def test_reduce_norm(self, name: str, ref: np.ndarray) -> None:
            prop = get_attr(RESULT, name, reduce=np.linalg.norm)
            np.testing.assert_allclose(prop, np.linalg.norm(ref))
