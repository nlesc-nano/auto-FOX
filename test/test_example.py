""" A module for testing files in the FOX/examples directory. """

__all__ = []

from os.path import (join, isfile)

import numpy as np

import FOX


REF_DIR = 'test/test_files'


def test_input():
    """ Test :mod:`FOX.examples.input`. """
    path = join(FOX.__path__[0], 'examples/input.py')
    assert isfile(path)
    with open(path, 'r') as f:
        exec(f.read())

    ref_rdf = np.load(join(REF_DIR, 'rdf.npy'))
    ref_rmsf = np.load(join(REF_DIR, 'rmsf.npy'))
    ref_rmsd = np.load(join(REF_DIR, 'rmsd.npy'))

    np.testing.assert_allclose(rdf.values, ref_rdf)
    np.testing.assert_allclose(rmsf.values, ref_rmsf)
    np.testing.assert_allclose(rmsd.values, ref_rmsd)
