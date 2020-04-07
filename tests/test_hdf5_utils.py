"""A module for testing functions in the :mod:`FOX.io.hdf5_utils` module."""

from os import remove
from os.path import join, isfile

import h5py
import yaml
import numpy as np

from scm.plams import Settings
from assertionlib import assertion

import FOX
from FOX.armc import dict_to_armc
from FOX.io.hdf5_utils import create_hdf5, to_hdf5, from_hdf5, create_xyz_hdf5

PATH: str = join('tests', 'test_files')


def test_create_hdf5():
    """Test :meth:`FOX.io.hdf5_utils.create_hdf5`."""
    yaml_file = join(PATH, 'armc.yaml')
    with open(yaml_file, 'r') as f:
        armc, _ = dict_to_armc(yaml.load(f.read(), Loader=yaml.FullLoader))
    armc.hdf5_file = hdf5_file = join(PATH, 'test.hdf5')

    ref_dict = Settings()
    ref_dict.acceptance.shape = 500, 100
    ref_dict.acceptance.dtype = bool
    ref_dict.aux_error.shape = 500, 100, 1, 1
    ref_dict.aux_error.dtype = np.float
    ref_dict.aux_error_mod.shape = 500, 100, 15
    ref_dict.aux_error_mod.dtype = np.float
    ref_dict.param.shape = 500, 100, 1, 14
    ref_dict.param.dtype = np.float
    ref_dict.phi.shape = 500, 1
    ref_dict.phi.dtype = np.float
    ref_dict['rdf.0'].shape = 500, 100, 241, 6
    ref_dict['rdf.0'].dtype = np.float
    ref_dict['rdf.0.ref'].shape = 241, 6
    ref_dict['rdf.0.ref'].dtype = np.float

    try:
        create_hdf5(hdf5_file, armc)
        with h5py.File(hdf5_file, 'r') as f:
            for key, value in f.items():
                assertion.shape_eq(value, ref_dict[key].shape)
                assertion.isinstance(value[:].item(0), ref_dict[key].dtype)
    finally:
        remove(hdf5_file) if isfile(hdf5_file) else None


def test_to_hdf5():
    """Test :meth:`FOX.io.hdf5_utils.to_hdf5`."""
    yaml_file = join(PATH, 'armc.yaml')
    with open(yaml_file, 'r') as f:
        armc, _ = dict_to_armc(yaml.load(f.read(), Loader=yaml.FullLoader))
    armc.hdf5_file = hdf5_file = join(PATH, 'test.hdf5')

    kappa = 5
    omega = 15
    hdf5_dict = {
        'xyz': [FOX.MultiMolecule.from_xyz(FOX.example_xyz)],
        'phi': np.array([5.0]),
        'param': np.array([np.arange(14, dtype=float)]),
        'acceptance': True,
        'aux_error': np.array([2.0]),
    }
    hdf5_dict['rdf.0'] = hdf5_dict['xyz'][0].init_rdf(atom_subset=['Cd', 'Se', 'O']).values
    hdf5_dict['aux_error_mod'] = np.append(hdf5_dict['param'], hdf5_dict['phi'])
    hdf5_dict['xyz'] = armc.molecule

    try:
        create_hdf5(hdf5_file, armc)
        create_xyz_hdf5(hdf5_file, armc.molecule, 100)
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
    finally:
        xyz_hdf5 = hdf5_file.replace('.hdf5', '.xyz.hdf5')
        remove(hdf5_file) if isfile(hdf5_file) else None
        remove(xyz_hdf5) if isfile(xyz_hdf5) else None


def test_from_hdf5():
    """Test :meth:`FOX.io.hdf5_utils.from_hdf5`."""
    yaml_file = join(PATH, 'armc.yaml')
    with open(yaml_file, 'r') as f:
        armc, _ = dict_to_armc(yaml.load(f.read(), Loader=yaml.FullLoader))
    armc.hdf5_file = hdf5_file = join(PATH, 'test.hdf5')

    kappa = 0
    omega = 0
    hdf5_dict = {
        'xyz': [FOX.MultiMolecule.from_xyz(FOX.example_xyz)],
        'phi': np.array([5.0]),
        'param': np.array([np.arange(14, dtype=float)]),
        'acceptance': True,
        'aux_error': np.array([2.0]),
    }
    hdf5_dict['rdf.0'] = hdf5_dict['xyz'][0].init_rdf(atom_subset=['Cd', 'Se', 'O']).values
    hdf5_dict['aux_error_mod'] = np.append(hdf5_dict['param'], hdf5_dict['phi'])
    hdf5_dict['xyz'] = armc.molecule

    try:
        create_hdf5(hdf5_file, armc)
        create_xyz_hdf5(hdf5_file, armc.molecule, 100)
        to_hdf5(hdf5_file, hdf5_dict, kappa, omega)
        out = from_hdf5(hdf5_file)

        assertion.eq(hdf5_dict['acceptance'], out['acceptance'][0])
        assertion.eq(hdf5_dict['aux_error'], out['aux_error']['rdf.0'][0])
        assertion.eq(hdf5_dict['phi'], out['aux_error_mod'].values[0])
        assertion.eq(hdf5_dict['phi'], out['phi'][0][0])
        np.testing.assert_allclose(hdf5_dict['param'], out['param'].values[0])
        np.testing.assert_allclose(hdf5_dict['rdf.0'], out['rdf.0'][0].values)
    finally:
        xyz_hdf5 = hdf5_file.replace('.hdf5', '.xyz.hdf5')
        remove(hdf5_file) if isfile(hdf5_file) else None
        remove(xyz_hdf5) if isfile(xyz_hdf5) else None
