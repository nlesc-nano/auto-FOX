"""A module for testing files in the :mod:`FOX.utils` module,"""

from pathlib import Path
from types import MappingProxyType

import pandas as pd
import numpy as np

from assertionlib import assertion

from FOX.utils import (
    get_move_range, array_to_index, assert_error, serialize_array, read_str_file,
    get_shape, slice_str, get_atom_count, get_importable, group_by_values,
    read_rtf_file, fill_diagonal_blocks, split_dict, as_nd_array, prepend_exception,
    VersionInfo
)

PATH = Path('tests') / 'test_files'


def test_assert_error() -> None:
    """Test :func:`FOX.utils.assert_error`."""
    msg = 'test error {}'

    @assert_error(msg)
    def test_func():
        pass

    assertion.assert_(test_func, exception=ModuleNotFoundError)


def test_serialize_array() -> None:
    """Test :func:`FOX.utils.serialize_array`."""
    zeros = np.zeros((10, 2), dtype=bool)
    ref = ('         0         0         0         0         0         0         0         '
           '0\n         0         0         0         0         0         0         0         '
           '0\n         0         0         0         0')
    assertion.eq(serialize_array(zeros), ref)


def test_read_str_file() -> None:
    """Test :func:`FOX.utils.read_str_file`."""
    at, charge = read_str_file(PATH / 'ligand.str')
    assertion.eq(at, ('CG2O3', 'HGR52', 'OG2D2', 'OG2D2'))
    assertion.eq(charge, (0.52, 0.0, -0.76, -0.76))


def test_get_shape() -> None:
    """Test :func:`FOX.utils.get_shape`."""
    a = np.random.rand(100, 10)
    b = a[0:15, 0].tolist()
    c = 5

    assertion.eq(get_shape(a), (100, 10))
    assertion.eq(get_shape(b), (15,))
    assertion.eq(get_shape(c), (1,))


def test_array_to_index() -> None:
    """Test :func:`FOX.utils.array_to_index`."""
    ar1 = np.ones(5, dtype=bytes)
    ar2 = np.ones(5)
    ar3 = np.ones((2, 5))
    ar4 = np.ones((5, 5, 5))

    idx2 = pd.Index(5 * [1])
    idx1 = idx2.astype(str)
    idx3 = pd.MultiIndex.from_product([[1], 5 * [1]])

    np.testing.assert_array_equal(array_to_index(ar1), idx1)
    np.testing.assert_array_equal(array_to_index(ar2), idx2)
    np.testing.assert_array_equal(array_to_index(ar3), idx3)
    assertion.assert_(array_to_index, ar4, exception=ValueError)


def test_get_move_range() -> None:
    """Test :func:`FOX.utils.get_move_range`."""
    ar1 = get_move_range()
    ar2 = get_move_range(start=0.001)
    ar3 = get_move_range(step=0.001)
    ar4 = get_move_range(stop=0.01)
    ar5 = get_move_range(ratio=[1, 2, 4, 8])

    ref1 = np.load(PATH / 'get_move_range.1.npy')
    ref2 = np.load(PATH / 'get_move_range.2.npy')
    ref3 = np.load(PATH / 'get_move_range.3.npy')
    ref4 = np.load(PATH / 'get_move_range.4.npy')
    ref5 = np.load(PATH / 'get_move_range.5.npy')

    np.testing.assert_allclose(ar1, ref1)
    np.testing.assert_allclose(ar2, ref2)
    np.testing.assert_allclose(ar3, ref3)
    np.testing.assert_allclose(ar4, ref4)
    np.testing.assert_allclose(ar5, ref5)

    assertion.assert_(get_move_range, stop=10, step=1, exception=ValueError)
    assertion.assert_(get_move_range, ratio=[1, 99], exception=ValueError)


def test_slice_str() -> None:
    """Test :func:`FOX.utils.slice_str`."""
    intervals = [None, 3, 6, None]
    string = '12 456789'

    str_list1 = slice_str(string, intervals)
    str_list2 = slice_str(string, intervals, strip_spaces=False)
    assertion.eq(str_list1, ['12', '456', '789'])
    assertion.eq(str_list2, ['12 ', '456', '789'])


def test_get_atom_count() -> None:
    """Test :func:`FOX.utils.get_atom_count`."""
    iterable = ['H', 'C', 'O', 'N']
    count = {'H': 6, 'C': 2, 'O': 1}
    lst = get_atom_count(iterable, count)

    assertion.eq(lst, [6, 2, 1, None])


def test_get_importable() -> None:
    """Test :func:`FOX.utils.get_importable`."""

    def _validate(dct: dict):
        return not isinstance(dct, dict)

    dict_type = get_importable('builtins.dict')
    assertion.is_(dict_type, dict)

    assertion.assert_(get_importable, b'bob', exception=TypeError)
    assertion.assert_(get_importable, 'bob', exception=ModuleNotFoundError)
    assertion.assert_(get_importable, 'builtins.dict', validate=_validate, exception=RuntimeError)


def test_group_by_values() -> None:
    """Test :func:`FOX.utils.group_by_values`."""
    str_list = ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b']

    dct1 = group_by_values(enumerate(str_list))
    dct2 = group_by_values(enumerate(str_list), mapping_type=MappingProxyType)
    ref = {'a': [0, 1, 2, 3, 4], 'b': [5, 6, 7]}

    assertion.eq(dct1, ref)
    assertion.eq(dct2, ref)
    assertion.isinstance(dct2, MappingProxyType)


def test_read_rtf_file() -> None:
    """Test :func:`FOX.utils.read_rtf_file`."""
    at, charge = read_rtf_file(PATH / 'ligand.rtf')
    at_ref = ('C331', 'C321', 'C2O3', 'O2D2', 'O2D2', 'HGA2', 'HGA2', 'HGA3', 'HGA3', 'HGA3')
    charge_ref = (-0.27, -0.28, 0.62, -0.76, -0.76, 0.09, 0.09, 0.09, 0.09, 0.09)

    assertion.eq(at, at_ref)
    assertion.eq(charge, charge_ref)


def test_fill_diagonal_blocks() -> None:
    """Test :func:`FOX.utils.fill_diagonal_blocks`."""
    i = 2
    j = 3
    ar = np.zeros((10, 15), dtype=int)
    fill_diagonal_blocks(ar, i, j, val=1)

    ref = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])
    np.testing.assert_array_equal(ar, ref)

    assertion.assert_(fill_diagonal_blocks, ar, 0, 1, exception=ValueError)
    assertion.assert_(fill_diagonal_blocks, ar, 1, 0, exception=ValueError)


def test_split_dict() -> None:
    """Test :func:`FOX.utils.split_dict`."""
    keep_keys = {'a', 'b'}
    dct1 = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    dct2 = dct1.copy()
    out1 = split_dict(dct1, keep_keys)
    out2 = split_dict(dct2, keep_keys, keep_order=False)

    assertion.eq(dct1, dct2)
    assertion.eq(dct1, {'a': 1, 'b': 2})
    assertion.eq(out1, out2)
    assertion.eq(out1, {'c': 3, 'd': 4})


def test_as_nd_array() -> None:
    """Test :func:`FOX.utils.as_nd_array`."""
    a = as_nd_array(1, dtype=int)
    b = as_nd_array(1, dtype=float)
    c = as_nd_array([1], dtype=int)
    d = as_nd_array([1], dtype=float)
    e = as_nd_array(iter([1]), dtype=int)
    f = as_nd_array(iter([1]), dtype=float)

    ref1 = np.array(1)
    ref2 = np.array(1.0)
    ref3 = np.array([1])
    ref4 = np.array([1.0])

    np.testing.assert_array_equal(a, ref1)
    np.testing.assert_array_equal(b, ref2)
    np.testing.assert_array_equal(c, ref3)
    np.testing.assert_array_equal(d, ref4)
    np.testing.assert_array_equal(e, ref3)
    np.testing.assert_array_equal(f, ref4)

    assertion.assert_(as_nd_array, None, dtype=int, exception=TypeError)


def test_prepend_exception() -> None:
    """Test :func:`FOX.utils.prepend_exception`."""

    @prepend_exception('custom message: ', exception=TypeError)
    def func():
        raise TypeError('test')

    try:
        func()
    except TypeError as ex1:
        assertion.eq(str(ex1), "custom message: test")
    except Exception as ex2:
        raise AssertionError("Failed to raise an 'AssertionError'") from ex2
    else:
        raise AssertionError("Failed to raise an 'AssertionError'")


def test_version_info() -> None:
    """Test :class:`FOX.utils.VersionInfo`."""
    tup1 = VersionInfo(0, 1, 2)
    tup2 = VersionInfo.from_str('0.1.2')
    assertion.eq(tup1, (0, 1, 2))
    assertion.eq(tup2, (0, 1, 2))

    assertion.assert_(VersionInfo.from_str, b'0.1.2', exception=TypeError)
    assertion.assert_(VersionInfo.from_str, '0.1.2a', exception=ValueError)
    assertion.assert_(VersionInfo.from_str, '0.1.2.3.4', exception=ValueError)
