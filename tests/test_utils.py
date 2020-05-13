"""A module for testing files in the :mod:`FOX.functions.utils` module,"""

from os.path import join

import pandas as pd
import numpy as np

from scm.plams import Settings
from assertionlib import assertion

from FOX.utils import (
    assert_error, serialize_array, read_str_file, get_shape
)

PATH = join('tests', 'test_files')


def test_assert_error():
    """Test :func:`FOX.functions.utils.assert_error`,"""
    msg = 'test error {}'
    @assert_error(msg)
    def test_func():
        pass

    assertion.assert_(test_func, exception=ModuleNotFoundError)


def test_serialize_array():
    """Test :func:`FOX.functions.utils.serialize_array`,"""
    zeros = np.zeros((10, 2), dtype=bool)
    ref = ('         0         0         0         0         0         0         0         '
           '0\n         0         0         0         0         0         0         0         '
           '0\n         0         0         0         0')
    assertion.eq(serialize_array(zeros), ref)


def test_read_str_file():
    """Test :func:`FOX.functions.utils.read_str_file`,"""
    at, charge = read_str_file(join(PATH, 'ligand.str'))
    assertion.eq(at, ('CG2O3', 'HGR52', 'OG2D2', 'OG2D2'))
    assertion.eq(charge, (0.52, 0.0, -0.76, -0.76))


def test_get_shape():
    """Test :func:`FOX.functions.utils.get_shape`,"""
    a = np.random.rand(100, 10)
    b = a[0:15, 0].tolist()
    c = 5

    assertion.eq(get_shape(a), (100, 10))
    assertion.eq(get_shape(b), (15,))
    assertion.eq(get_shape(c), (1,))
