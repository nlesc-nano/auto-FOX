""" A module for testing example input files in the FOX/examples directory. """

__all__ = []

from os import remove
from os.path import join

import numpy as np
import pandas as pd

import FOX


MOL = FOX.MultiMolecule.from_xyz(FOX.get_example_xyz())
REF_DIR = 'test/test_files'


def test_input():
    """ Test :mod:`FOX.examples.input`. """
    global_dict = {}
    local_dict = {}
    path = join(FOX.__path__[0], 'examples/input.py')
    with open(path, 'r') as f:
        exec(f.read(), global_dict, local_dict)

    ref_rdf = np.load(join(REF_DIR, 'rdf.npy'))
    ref_rmsf = np.load(join(REF_DIR, 'rmsf.npy'))
    ref_rmsd = np.load(join(REF_DIR, 'rmsd.npy'))

    np.testing.assert_allclose(local_dict['rdf'].values, ref_rdf)
    np.testing.assert_allclose(local_dict['rmsf'].values, ref_rmsf)
    np.testing.assert_allclose(local_dict['rmsd'].values, ref_rmsd)


def test_cp2k_md():
    """ Test :mod:`FOX.examples.cp2k_md`. """
    global_dict = {}
    local_dict = {}
    path = join(FOX.__path__[0], 'examples/cp2k_md.py')
    with open(path, 'r') as f:
        exec(f.read(), global_dict, local_dict)
    remove('mol.psf')

    mol = MOL.copy()
    psf = {
        'filename': 'mol.psf',
        'atoms': pd.read_csv(join(REF_DIR, 'psf_atoms.csv'), float_precision='high', index_col=0),
        'bonds': np.load(join(REF_DIR, 'bonds.npy')),
        'angles': np.load(join(REF_DIR, 'angles.npy')),
        'dihedrals': np.load(join(REF_DIR, 'dihedrals.npy')),
        'impropers': np.load(join(REF_DIR, 'impropers.npy')),
        }

    np.testing.assert_allclose(local_dict['mol'], mol)
    assert local_dict['psf']['filename'] == psf['filename']
    np.testing.assert_allclose(local_dict['psf']['bonds'], psf['bonds'])
    np.testing.assert_allclose(local_dict['psf']['angles'], psf['angles'])
    np.testing.assert_allclose(local_dict['psf']['dihedrals'], psf['dihedrals'])
    np.testing.assert_allclose(local_dict['psf']['impropers'], psf['impropers'])
    for key in local_dict['psf']['atoms']:
        if not local_dict['psf']['atoms'][key].dtype.name == 'object':
            np.testing.assert_allclose(local_dict['psf']['atoms'][key], psf['atoms'][key])
        else:
            np.testing.assert_array_equal(local_dict['psf']['atoms'][key], psf['atoms'][key])

    armc = local_dict['armc']
    assert armc.phi.phi == 1.0
    assert armc.phi.kwarg == {}
    assert armc.phi.func == np.add

    assert armc.armc.a_target == 0.25
    assert armc.armc.gamma == 2.0
    assert armc.armc.iter_len == 50000
    assert armc.armc.sub_iter_len == 100
