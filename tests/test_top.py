from __future__ import annotations

import copy
import pickle
import weakref
from pathlib import Path
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest
import h5py
import numpy as np
import pandas as pd
from FOX import TOPContainer
from FOX.io import TOPDirectiveWarning
from assertionlib import assertion

if TYPE_CHECKING:
    import _pytest

PATH = Path('tests') / 'test_files'


class TestTOPContainer:
    DF_NAMES = ("mass", "atom", "bond", "impr", "angles", "dihe")

    @pytest.fixture(scope="class", autouse=True)
    def top(self, request: _pytest.fixtures.SubRequest) -> Iterator[TOPContainer]:
        with pytest.warns(TOPDirectiveWarning):
            top = TOPContainer.from_file(PATH / "test.top")
        yield top

    @pytest.mark.parametrize("name", TOPContainer.DF_DTYPES.keys())
    def test_attribute(self, top: TOPContainer, name: str) -> None:
        df: pd.DataFrame = getattr(top, name)
        with h5py.File(PATH / "test_ref.hdf5", "r") as f:
            if (dtype := TOPContainer.DF_DTYPES.get(name)) is not None:
                pass
            else:
                raise AssertionError
            ref = f[f"test_top/TestTOPContainer/test_attribute/{name}"][...].astype(dtype)
        assertion.eq(tuple(df.columns), ref.dtype.names)

        for field_name, series in df.items():
            ref_array = i if (i := ref[field_name]).dtype != np.object_ else i.astype(np.str_)
            if issubclass(series.dtype.type, np.inexact):
                np.testing.assert_allclose(series.values, ref_array, err_msg=field_name)
            else:
                np.testing.assert_array_equal(series.values, ref_array, err_msg=field_name)

    @pytest.mark.parametrize(
        "name,func",
        [(k, i) for k, dct in TOPContainer.DF_DICT_DTYPES.items() for i in dct],
    )
    def test_dict_attribute(self, top: TOPContainer, name: str, func: int) -> None:
        df: None | pd.DataFrame = getattr(top, name).get(func)
        if df is None:
            return

        with h5py.File(PATH / "test_ref.hdf5", "r") as f:
            dtype = TOPContainer.DF_DICT_DTYPES[name][func]
            ref = f[f"test_top/TestTOPContainer/test_attribute/{name}/{func}"][...].astype(dtype)
        assertion.eq(tuple(df.columns), ref.dtype.names)

        for field_name, series in df.items():
            ref_array = i if (i := ref[field_name]).dtype != np.object_ else i.astype(np.str_)
            if issubclass(series.dtype.type, np.inexact):
                np.testing.assert_allclose(series.values, ref_array, err_msg=field_name)
            else:
                np.testing.assert_array_equal(series.values, ref_array, err_msg=field_name)

    def test_to_hdf5_dict(self, top: TOPContainer) -> None:
        dct = top._to_hdf5_dict()
        assertion.isinstance(dct, dict)
        for v in dct.values():
            assertion.isinstance(v, np.ndarray)
            assertion.is_not(v.dtype.fields, None)

    def test_eq(self, top: TOPContainer) -> None:
        assertion.eq(top, top)
        assertion.eq(top, copy.copy(top))
        assertion.eq(top, copy.deepcopy(top))
        assertion.eq(top, pickle.loads(pickle.dumps(top)))
        assertion.eq(top, weakref.ref(top)())

    def test_ne(self, top: TOPContainer) -> None:
        top2 = copy.deepcopy(top)
        top2.atomtypes["mass"] += 1
        assertion.ne(top, top2)
        assertion.ne(top, copy.copy(top2))
        assertion.ne(top, copy.deepcopy(top2))
        assertion.ne(top, pickle.loads(pickle.dumps(top2)))

    def test_repr(self, top: TOPContainer) -> None:
        assertion.contains(repr(top), type(top).__name__)
        assertion.contains(str(top), type(top).__name__)

    def test_hash(self, top: TOPContainer) -> None:
        with pytest.raises(TypeError):
            hash(top)

    def test_to_file(self, top: TOPContainer, tmp_path: Path) -> None:
        top.to_file(tmp_path / "test.top")
        assertion.isfile(tmp_path / "test.top")
        top2 = TOPContainer.from_file(tmp_path / "test.top")
        assertion.assert_(top2.isclose, top2, rtol=0, atol=0.0001)
