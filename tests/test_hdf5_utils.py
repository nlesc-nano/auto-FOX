"""A module for testing functions in the :mod:`FOX.io.hdf5_utils` module."""

from pathlib import Path
from math import inf

import h5py
import yaml
import numpy as np
import pandas as pd

from scm.plams import Settings
from assertionlib import assertion
from nanoutils import delete_finally, recursive_items, UniqueLoader

import FOX
from FOX.armc import dict_to_armc
from FOX.io.hdf5_utils import create_hdf5, to_hdf5, from_hdf5, create_xyz_hdf5

PATH = Path('tests') / 'test_files'


@delete_finally(PATH / 'test.hdf5')
def test_create_hdf5():
    """Test :meth:`FOX.io.hdf5_utils.create_hdf5`."""
    yaml_file = PATH / 'armc.yaml'
    with open(yaml_file, 'r') as f:
        armc, _ = dict_to_armc(yaml.load(f.read(), Loader=UniqueLoader))
    armc.hdf5_file = hdf5_file = PATH / 'test.hdf5'

    ref_dict = Settings()
    ref_dict['/acceptance'].shape = 500, 100, 1
    ref_dict['/acceptance'].dtype = np.bool_
    ref_dict['/aux_error'].shape = 500, 100, 1, 1
    ref_dict['/aux_error'].dtype = np.float64
    ref_dict['/validation/aux_error'].shape = 500, 100, 1, 0
    ref_dict['/validation/aux_error'].dtype = np.float64
    ref_dict['/aux_error_mod'].shape = 500, 100, 1, 16
    ref_dict['/aux_error_mod'].dtype = np.float64
    ref_dict['/param'].shape = 500, 100, 1, 15
    ref_dict['/param'].dtype = np.float64
    ref_dict['/param_metadata'].shape = (1, 15)
    ref_dict['/param_metadata'].dtype = np.void
    ref_dict['/phi'].shape = 500, 1
    ref_dict['/phi'].dtype = np.float64
    ref_dict['/rdf.0'].shape = 500, 100, 241, 6
    ref_dict['/rdf.0'].dtype = np.float64
    ref_dict['/rdf.0.ref'].shape = 241, 6
    ref_dict['/rdf.0.ref'].dtype = np.float64

    create_hdf5(hdf5_file, armc)
    with h5py.File(hdf5_file, 'r') as f:
        for key, value in recursive_items(f):
            assertion.shape_eq(value, ref_dict[key].shape, message=key)
            assertion.eq(value.dtype.type, ref_dict[key].dtype, message=key)


@delete_finally(PATH / 'test.hdf5', PATH / 'test.xyz.hdf5')
def test_to_hdf5():
    """Test :meth:`FOX.io.hdf5_utils.to_hdf5`."""
    yaml_file = PATH / 'armc.yaml'
    with open(yaml_file, 'r') as f:
        armc, _ = dict_to_armc(yaml.load(f.read(), Loader=UniqueLoader))
    armc.hdf5_file = hdf5_file = PATH / 'test.hdf5'

    kappa = 5
    omega = 15
    hdf5_dict = {
        'xyz': [FOX.MultiMolecule.from_xyz(FOX.example_xyz)],
        'phi': np.array([5.0]),
        'param': np.array(np.arange(15, dtype=float), ndmin=2),
        'acceptance': True,
        'aux_error': np.array([2.0]),
    }
    hdf5_dict['rdf.0'] = hdf5_dict['xyz'][0].init_rdf(atom_subset=['Cd', 'Se', 'O']).values
    hdf5_dict['xyz'] = armc.molecule
    hdf5_dict['aux_error_mod'] = np.hstack([
        hdf5_dict['param'], hdf5_dict['phi'][None, ...]
    ])

    create_hdf5(hdf5_file, armc)
    create_xyz_hdf5(hdf5_file, armc.molecule, 100, phi=[1.0])
    to_hdf5(hdf5_file, hdf5_dict, kappa, omega)

    with h5py.File(hdf5_file, 'r') as f:
        for key, value in hdf5_dict.items():
            # Prepare slices
            if key == 'xyz':
                continue
            elif key == 'phi':
                dset_slice = (kappa, )
            else:
                dset_slice = (kappa, omega)

            # Assert
            try:
                assert value == f[key][dset_slice]
            except ValueError:
                np.testing.assert_allclose(value, f[key][dset_slice])


PARAM_METADATA = pd.DataFrame.from_dict({
    'min': {
        ('charge', 'charge', 'CG2O3'): -inf,
        ('charge', 'charge', 'Cd'): 0.5,
        ('charge', 'charge', 'HGR52'): -inf,
        ('charge', 'charge', 'OG2D2'): -inf,
        ('charge', 'charge', 'Se'): -inf,
        ('lennard_jones', 'epsilon', 'Cd Cd'): -inf,
        ('lennard_jones', 'epsilon', 'Cd OG2D2'): -inf,
        ('lennard_jones', 'epsilon', 'Cd Se'): -inf,
        ('lennard_jones', 'epsilon', 'OG2D2 Se'): -inf,
        ('lennard_jones', 'epsilon', 'Se Se'): -inf,
        ('lennard_jones', 'sigma', 'Cd Cd'): -inf,
        ('lennard_jones', 'sigma', 'Cd OG2D2'): -inf,
        ('lennard_jones', 'sigma', 'Cd Se'): -inf,
        ('lennard_jones', 'sigma', 'OG2D2 Se'): -inf,
        ('lennard_jones', 'sigma', 'Se Se'): -inf,
    },
    'max': {
        ('charge', 'charge', 'CG2O3'): inf,
        ('charge', 'charge', 'Cd'): 1.5,
        ('charge', 'charge', 'HGR52'): inf,
        ('charge', 'charge', 'OG2D2'): 0.0,
        ('charge', 'charge', 'Se'): -0.5,
        ('lennard_jones', 'epsilon', 'Cd Cd'): inf,
        ('lennard_jones', 'epsilon', 'Cd OG2D2'): inf,
        ('lennard_jones', 'epsilon', 'Cd Se'): inf,
        ('lennard_jones', 'epsilon', 'OG2D2 Se'): inf,
        ('lennard_jones', 'epsilon', 'Se Se'): inf,
        ('lennard_jones', 'sigma', 'Cd Cd'): inf,
        ('lennard_jones', 'sigma', 'Cd OG2D2'): inf,
        ('lennard_jones', 'sigma', 'Cd Se'): inf,
        ('lennard_jones', 'sigma', 'OG2D2 Se'): inf,
        ('lennard_jones', 'sigma', 'Se Se'): inf,
    },
    'count': {
        ('charge', 'charge', 'CG2O3'): 26,
        ('charge', 'charge', 'Cd'): 68,
        ('charge', 'charge', 'HGR52'): 26,
        ('charge', 'charge', 'OG2D2'): 52,
        ('charge', 'charge', 'Se'): 55,
        ('lennard_jones', 'epsilon', 'Cd Cd'): 2278,
        ('lennard_jones', 'epsilon', 'Cd OG2D2'): 3536,
        ('lennard_jones', 'epsilon', 'Cd Se'): 3740,
        ('lennard_jones', 'epsilon', 'OG2D2 Se'): 2860,
        ('lennard_jones', 'epsilon', 'Se Se'): 1485,
        ('lennard_jones', 'sigma', 'Cd Cd'): 2278,
        ('lennard_jones', 'sigma', 'Cd OG2D2'): 3536,
        ('lennard_jones', 'sigma', 'Cd Se'): 3740,
        ('lennard_jones', 'sigma', 'OG2D2 Se'): 2860,
        ('lennard_jones', 'sigma', 'Se Se'): 1485,
    },
    'frozen': {
        ('charge', 'charge', 'CG2O3'): True,
        ('charge', 'charge', 'Cd'): False,
        ('charge', 'charge', 'HGR52'): True,
        ('charge', 'charge', 'OG2D2'): False,
        ('charge', 'charge', 'Se'): False,
        ('lennard_jones', 'epsilon', 'Cd Cd'): False,
        ('lennard_jones', 'epsilon', 'Cd OG2D2'): False,
        ('lennard_jones', 'epsilon', 'Cd Se'): False,
        ('lennard_jones', 'epsilon', 'OG2D2 Se'): False,
        ('lennard_jones', 'epsilon', 'Se Se'): False,
        ('lennard_jones', 'sigma', 'Cd Cd'): False,
        ('lennard_jones', 'sigma', 'Cd OG2D2'): False,
        ('lennard_jones', 'sigma', 'Cd Se'): False,
        ('lennard_jones', 'sigma', 'OG2D2 Se'): False,
        ('lennard_jones', 'sigma', 'Se Se'): False,
    },
    'guess': {
        ('charge', 'charge', 'CG2O3'): False,
        ('charge', 'charge', 'Cd'): False,
        ('charge', 'charge', 'HGR52'): False,
        ('charge', 'charge', 'OG2D2'): False,
        ('charge', 'charge', 'Se'): False,
        ('lennard_jones', 'epsilon', 'Cd Cd'): False,
        ('lennard_jones', 'epsilon', 'Cd OG2D2'): False,
        ('lennard_jones', 'epsilon', 'Cd Se'): False,
        ('lennard_jones', 'epsilon', 'OG2D2 Se'): False,
        ('lennard_jones', 'epsilon', 'Se Se'): False,
        ('lennard_jones', 'sigma', 'Cd Cd'): False,
        ('lennard_jones', 'sigma', 'Cd OG2D2'): False,
        ('lennard_jones', 'sigma', 'Cd Se'): False,
        ('lennard_jones', 'sigma', 'OG2D2 Se'): False,
        ('lennard_jones', 'sigma', 'Se Se'): False,
    },
    'unit': {
        ('charge', 'charge', 'CG2O3'): '',
        ('charge', 'charge', 'Cd'): '',
        ('charge', 'charge', 'HGR52'): '',
        ('charge', 'charge', 'OG2D2'): '',
        ('charge', 'charge', 'Se'): '',
        ('lennard_jones', 'epsilon', 'Cd Cd'): 'kjmol',
        ('lennard_jones', 'epsilon', 'Cd OG2D2'): 'kjmol',
        ('lennard_jones', 'epsilon', 'Cd Se'): 'kjmol',
        ('lennard_jones', 'epsilon', 'OG2D2 Se'): 'kjmol',
        ('lennard_jones', 'epsilon', 'Se Se'): 'kjmol',
        ('lennard_jones', 'sigma', 'Cd Cd'): 'nm',
        ('lennard_jones', 'sigma', 'Cd OG2D2'): 'nm',
        ('lennard_jones', 'sigma', 'Cd Se'): 'nm',
        ('lennard_jones', 'sigma', 'OG2D2 Se'): 'nm',
        ('lennard_jones', 'sigma', 'Se Se'): 'nm',
    },
})
PARAM_METADATA.columns = pd.MultiIndex.from_tuples([(0, k) for k in PARAM_METADATA.columns])


@delete_finally(PATH / 'test.hdf5', PATH / 'test.xyz.hdf5')
def test_from_hdf5():
    """Test :meth:`FOX.io.hdf5_utils.from_hdf5`."""
    yaml_file = PATH / 'armc.yaml'
    with open(yaml_file, 'r') as f:
        armc, _ = dict_to_armc(yaml.load(f.read(), Loader=UniqueLoader))
    armc.hdf5_file = hdf5_file = PATH / 'test.hdf5'

    kappa = 0
    omega = 0
    hdf5_dict = {
        'xyz': [FOX.MultiMolecule.from_xyz(FOX.example_xyz)],
        'phi': np.array([5.0]),
        'param': np.array([np.arange(15, dtype=float)]),
        'acceptance': True,
        'aux_error': np.array([2.0]),
    }
    hdf5_dict['rdf.0'] = hdf5_dict['xyz'][0].init_rdf(atom_subset=['Cd', 'Se', 'O']).values
    hdf5_dict['aux_error_mod'] = np.append(hdf5_dict['param'], hdf5_dict['phi'])
    hdf5_dict['xyz'] = armc.molecule

    create_hdf5(hdf5_file, armc)
    create_xyz_hdf5(hdf5_file, armc.molecule, 100, phi=[1.0])
    to_hdf5(hdf5_file, hdf5_dict, kappa, omega)
    out = from_hdf5(hdf5_file)

    assertion.eq(hdf5_dict['acceptance'], out['acceptance'].loc[0, 0])
    assertion.eq(hdf5_dict['aux_error'], out['aux_error']['rdf.0'][0])
    assertion.eq(hdf5_dict['phi'], out['aux_error_mod'].values[:, -1])
    assertion.eq(hdf5_dict['phi'][0], out['phi'].loc[0, 0])
    np.testing.assert_allclose(hdf5_dict['param'], out['param'][0].T)
    np.testing.assert_allclose(hdf5_dict['rdf.0'], out['rdf.0'][0].values)
    np.testing.assert_array_equal(PARAM_METADATA, out['param_metadata'])
