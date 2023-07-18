from __future__ import annotations

import copy
import pickle
import weakref
from pathlib import Path
from collections.abc import Iterator, Callable
from typing import TYPE_CHECKING, Any

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
    @pytest.fixture(scope="class", autouse=True)
    def top(self, request: _pytest.fixtures.SubRequest) -> Iterator[TOPContainer]:
        with pytest.warns(TOPDirectiveWarning):
            top = TOPContainer.from_file(PATH / "test.top")
        yield top

    @pytest.mark.parametrize("name", TOPContainer.DF_DTYPES.keys())
    def test_attribute(self, top: TOPContainer, name: str) -> None:
        df: pd.DataFrame = getattr(top, name)
        with h5py.File(PATH / "test_ref.hdf5", "r+") as f:
            if (dtype := TOPContainer.DF_DTYPES.get(name)) is None:
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

    EQ_CALLBACKS: dict[str, Callable[[TOPContainer], None | TOPContainer]] = {
        "copy_method_shallow": lambda i: i.copy(deep=False),
        "copy_method_deep": lambda i: i.copy(deep=True),
        "copy_copy": lambda i: copy.copy(i),
        "copy_deepcopy": lambda i: copy.deepcopy(i),
        "pickle": lambda i: pickle.loads(pickle.dumps(i)),
        "weakref": lambda i: weakref.ref(i)(),
    }

    @pytest.mark.parametrize("callback", EQ_CALLBACKS.values(), ids=EQ_CALLBACKS.keys())
    def test_eq(self, top: TOPContainer, callback: Callable[[TOPContainer], TOPContainer]) -> None:
        assertion.eq(top, callback(top))

    @pytest.mark.parametrize("callback", EQ_CALLBACKS.values(), ids=EQ_CALLBACKS.keys())
    def test_ne(self, top: TOPContainer, callback: Callable[[TOPContainer], TOPContainer]) -> None:
        top2 = copy.deepcopy(top)
        top2.atomtypes["mass"] += 1
        assertion.ne(top, callback(top2))

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
        assertion.assert_(top2.allclose, top2, rtol=0, atol=0.0001)

    def test_generate_pairs(self, top: TOPContainer) -> None:
        pairs_ref = top.pairs.sort_values(
            ["molecule", "atom1", "atom2"], ignore_index=True,
        )

        top = top.copy()
        top.pairs = top.pairs.loc[[], :]
        top.generate_pairs()

        for field_name, series in top.pairs.items():
            ref = pairs_ref[field_name]
            if np.issubdtype(series.dtype, np.inexact):
                np.testing.assert_allclose(series.values, ref, err_msg=field_name)
            else:
                np.testing.assert_array_equal(series.values, ref, err_msg=field_name)


class TestTOPConcat:
    @pytest.fixture(scope="class", autouse=True)
    def top(self, request: _pytest.fixtures.SubRequest) -> Iterator[TOPContainer]:
        with pytest.warns(TOPDirectiveWarning):
            top = TOPContainer.from_file(PATH / "test.top")
        yield top

    PARAM = {
        "atomtypes": (
            "atomtypes",
            {"atnum": 21, "atom_type": "SC4", "sigma": 0.41, "epsilon": 2.35},
            ["atnum", "atom_type"],
        ),
        "atoms": (
            "atoms",
            {
                "molecule": "molecule9",
                "res_num": 1,
                "res_name": "TO",
                "atom_type": "TC5",
                "atom_name": "R3",
            },
            ["molecule", "atom1"],
        ),
        "pairs": (
            "pairs",
            {"molecule": "molecule5", "atom1": 21, "atom2": 22, "func": 1, },
            ["molecule", "atom1", "atom2"],
        ),
    }

    PARAM_DICT = {
        "nonbond_params": (
            "nonbond_params",
            1,
            {"atom1": "TC5", "atom2": "SC4", "func": 1, "sigma": 0.365, "epsilon": 1.91},
            ["atom1", "atom2"],
        ),
    }

    @pytest.mark.parametrize("name,kwargs,sort_fields", PARAM.values(), ids=PARAM.keys())
    def test(
        self,
        top: TOPContainer,
        name: str,
        kwargs: dict[str, Any],
        sort_fields: list[str],
    ) -> None:
        top = top.copy()
        ref_df: pd.DataFrame = getattr(top, name).sort_values(sort_fields, ignore_index=True)

        setattr(top, name, getattr(top, name).iloc[:-1, :])
        concatenate = getattr(top.concatenate, name)
        concatenate(**kwargs)

        df: pd.DataFrame = getattr(top, name)
        for field_name, series in df.items():
            ref = ref_df[field_name]
            if np.issubdtype(series.dtype, np.inexact):
                np.testing.assert_allclose(series, ref, err_msg=field_name, rtol=0, atol=0.001)
            else:
                np.testing.assert_array_equal(series, ref, err_msg=field_name)

    @pytest.mark.parametrize(
        "name,func,kwargs,sort_fields",
        PARAM_DICT.values(), ids=PARAM_DICT.keys(),
    )
    def test_dict(
        self,
        top: TOPContainer,
        name: str,
        func: int,
        kwargs: dict[str, Any],
        sort_fields: list[str],
    ) -> None:
        top = top.copy()
        df_dict: dict[int, pd.DataFrame] = getattr(top, name)
        ref_df = df_dict[func].sort_values(sort_fields, ignore_index=True)

        df_dict[func] = df_dict[func].iloc[:-1, :]
        concatenate = getattr(top.concatenate, name)
        concatenate(**kwargs)

        df: pd.DataFrame = df_dict[func]
        for field_name, series in df.items():
            ref = ref_df[field_name]
            if np.issubdtype(series.dtype, np.inexact):
                np.testing.assert_allclose(series, ref, err_msg=field_name, rtol=0, atol=0.001)
            else:
                np.testing.assert_array_equal(series, ref, err_msg=field_name)
