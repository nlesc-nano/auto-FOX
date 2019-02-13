""" A module for testing Auto-FOX functionality. """

__all__ = []

from itertools import chain

import numpy as np
import pandas as pd

from FOX.functions.read_xyz import read_multi_xyz
from FOX.functions.radial_distribution import get_all_radial
from FOX.examples.example_xyz import get_example_xyz

# Path to the test multi-xyz file
xyz_file = get_example_xyz()


def test_read_xyz(xyz_file):
    """ Test the FOX.functions.read_xyz module. """
    # Run functions
    xyz_array, idx_dict = read_multi_xyz(xyz_file)
    xyz_array2 = read_multi_xyz(xyz_file, ret_idx_dict=False)

    # Check xyz_array
    assert isinstance(xyz_array, np.ndarray)
    assert xyz_array.shape == (4905, 227, 3)
    assert xyz_array.dtype == np.float64

    # Check xyz_array2
    assert isinstance(xyz_array2, np.ndarray)
    assert xyz_array2.shape == xyz_array.shape
    assert xyz_array2.dtype == np.float64

    # Compare xyz_array & xyz_array2
    assert (xyz_array - xyz_array2).sum() == 0.0

    # Check idx_dict
    at_tup = ('Cd', 'Se', 'O', 'C', 'H')
    values = list(chain.from_iterable(idx_dict.values()))
    assert isinstance(idx_dict, dict)
    assert len(values) == xyz_array.shape[1]
    assert max(values) == xyz_array.shape[1] - 1
    for key in idx_dict:
        assert isinstance(key, str)
        assert isinstance(idx_dict[key], list)
        assert isinstance(idx_dict[key][0], int)
        assert key in at_tup


def test_radial_distribution(xyz_file):
    """ Test the FOX.functions.radial_distribution module. """
    # Set constants
    dr = 0.05
    r_max = 12.0
    atoms = ('Cd', 'Se', 'O')

    # Infer dimensions
    x = 1 + int(r_max / dr)
    y = np.math.factorial(len(atoms))

    # Run functions
    xyz_array, idx_dict = read_multi_xyz(xyz_file)
    df = get_all_radial(xyz_array, idx_dict, dr=dr, r_max=r_max, atoms=atoms)

    # Check df
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (x, y)
    assert df.index.min() == 0.0
    assert df.index.max() == r_max
    assert (df.iloc[0].values - np.zeros(y)).sum() == 0.0
    assert df.columns.name == 'Atom pairs'
    assert df.index.name == 'r  /  Ångström'
    for dtype in df.dtypes.values:
        assert dtype.name == 'float64'
    for key in df.keys():
        at1, at2 = key.split()
        assert at1 in atoms
        assert at2 in atoms
