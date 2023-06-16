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
from FOX import RTFContainer
from assertionlib import assertion

if TYPE_CHECKING:
    import _pytest

PATH = Path('tests') / 'test_files'


class TestRTFContainer:
    DF_NAMES = ("mass", "atom", "bond", "impr", "angles", "dihe")

    @pytest.fixture(scope="class", autouse=True)
    def rtf(self, request: _pytest.fixtures.SubRequest) -> Iterator[RTFContainer]:
        rtf = RTFContainer.from_file(PATH / "ola.rtf")
        rtf.auto_to_explicit()
        yield rtf

    def test_collapse_charges(self, rtf: RTFContainer) -> None:
        ref = {
            'C321': -0.18,
            'C331': -0.27,
            'C2D1': -0.15,
            'C2O3': 0.62,
            'O2D2': -0.76,
            'HGA4': 0.15,
            'HGA3': 0.09,
            'HGA2': 0.09,
        }
        dct = rtf.collapse_charges()
        assertion.eq(dct, ref)

    @pytest.mark.parametrize("name", DF_NAMES)
    def test_attribute(self, rtf: RTFContainer, name: str) -> None:
        name = name.upper()
        df: pd.DataFrame = getattr(rtf, name.lower()).reset_index(inplace=False, drop=False)
        with h5py.File(PATH / "test_ref.hdf5", "r") as f:
            dtype = rtf.DTYPES[name]
            ref = f[f"test_rtf/TestRTFContainer/test_attribute/{name}"][...].astype(dtype)
        assertion.eq(tuple(df.columns), ref.dtype.names)

        for field_name, series in df.items():
            if issubclass(series.dtype.type, np.inexact):
                np.testing.assert_allclose(series.values, ref[field_name], err_msg=field_name)
            else:
                np.testing.assert_array_equal(series.values, ref[field_name], err_msg=field_name)

    def test_residues(self, rtf: RTFContainer) -> None:
        assertion.eq(set(rtf.residues), {"UNL"})

    def test_version(self, rtf: RTFContainer) -> None:
        assertion.eq(rtf.charmm_version, (22, 0))

    def test_to_hdf5_dict(self, rtf: RTFContainer) -> None:
        dct = rtf._to_hdf5_dict()
        assertion.isinstance(dct, dict)
        for v in dct.values():
            assertion.isinstance(v, np.ndarray)
            assertion.is_not(v.dtype.fields, None)

    def test_eq(self, rtf: RTFContainer) -> None:
        assertion.eq(rtf, rtf)
        assertion.eq(rtf, copy.copy(rtf))
        assertion.eq(rtf, copy.deepcopy(rtf))
        assertion.eq(rtf, pickle.loads(pickle.dumps(rtf)))
        assertion.eq(rtf, weakref.ref(rtf)())

    def test_ne(self, rtf: RTFContainer) -> None:
        rtf2 = copy.deepcopy(rtf)
        rtf2.mass["mass"] += 1
        assertion.ne(rtf, rtf2)
        assertion.ne(rtf, copy.copy(rtf2))
        assertion.ne(rtf, copy.deepcopy(rtf2))
        assertion.ne(rtf, pickle.loads(pickle.dumps(rtf2)))

    def test_repr(self, rtf: RTFContainer) -> None:
        assertion.contains(repr(rtf), type(rtf).__name__)
        assertion.contains(str(rtf), type(rtf).__name__)

    def test_hash(self, rtf: RTFContainer) -> None:
        with pytest.raises(TypeError):
            hash(rtf)
