""" A module for testing files in the :mod:`FOX.io.read_prm` module. """

__all__ = []

from os import remove
from os.path import join

import pandas as pd
import numpy as np

from FOX.io.read_prm import (read_prm, write_prm, rename_atom_types, update_dtype)


REF_DIR = 'test/test_files'


def test_read_prm():
    """ Test :func:`FOX.io.read_prm.read_prm`. """
    prm_dict = read_prm(join(REF_DIR, 'test_param1.prm'))
    nonbonded = 'NONBONDED nbxmod  5 atom cdiel fshift vatom vdistance vfswitch -\n'
    nonbonded += 'cutnb 14.0 ctofnb 12.0 ctonnb 10.0 eps 1.0 e14fac 1.0 wmin 1.5\n'
    shape_dict = {
        'ATOMS': (159, 3),
        'BONDS': (526, 2),
        'ANGLES': (1639, 4),
        'DIHEDRALS': (4132, 3),
        nonbonded: (159, 6),
        'NBFIX': (75, 2),
        'HBOND CUTHB 0.5': (2, 0),
        'IMPROPERS': (128, 3)
    }

    for key in prm_dict:
        assert prm_dict[key].shape == shape_dict[key]


def test_write_prm():
    """ Test :func:`FOX.io.read_prm.write_prm`. """
    param_ref = join(REF_DIR, 'test_param2.prm')
    param_tmp = join(REF_DIR, 'param.tmp')
    prm_dict = read_prm(join(REF_DIR, 'test_param1.prm'))

    write_prm(prm_dict, param_tmp)
    with open(param_tmp, 'r') as a, open(param_ref, 'r') as b:
        for i, j in zip(a, b):
            assert i == j

    remove(param_tmp)


def test_rename_atom_types():
    """ Test :func:`FOX.io.read_prm.rename_atom_types`. """
    prm_dict = read_prm(join(REF_DIR, 'test_param1.prm'))

    rename_dict = {'CG2O3': 'C_1', 'HGR52': 'H_1', 'OG2D2': 'O_1'}
    ignore = ('HBOND CUTHB 0.5', 'NBFIX')
    rename_atom_types(prm_dict, rename_dict)

    for prm, df in prm_dict.items():
        if prm in ignore:
            continue
        for key, value in rename_dict.items():
            idx = np.array(df.index.tolist())
            assert key not in idx
            assert value in idx


def test_update_dtype():
    """ Test :func:`FOX.functions.read_prm.update_dtype`. """
    df = pd.DataFrame(np.random.rand(10, 3))
    df[1] = 1
    df[2] = 'test'
    float_blacklist = [1]

    update_dtype(df, float_blacklist)
    assert df[0].dtype == np.dtype('float64')
    assert df[1].dtype == np.dtype('int64')
    assert df[2].dtype == np.dtype('object')


def test_reorder_column_dict():
    """ Test :func:`FOX.io.read_prm.reorder_column_dict`. """
    df = read_prm(join(REF_DIR, 'test_param1.prm'))['ATOMS']
    assert (df.columns == pd.Index(['MASS', '-1', 'mass'], name='parameters')).all()
    assert df.index.name == 'Atom 1'
