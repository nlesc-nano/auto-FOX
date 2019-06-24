"""A module for testing functions in the :mod:`FOX.io.hdf5_utils` module."""

from os import remove
from os.path import join

import h5py
import numpy as np

from scm.plams import Settings

import FOX
from FOX.io.hdf5_utils import (create_hdf5, to_hdf5, from_hdf5, create_xyz_hdf5)

__all__: list = []

REF_DIR = 'tests/test_files'


def test_create_hdf5():
    """Test :meth:`FOX.io.hdf5_utils.create_hdf5`."""
    hdf5_path = join(REF_DIR, 'test.hdf5')
    try:
        remove(hdf5_path)
    except FileNotFoundError:
        pass

    examples = join(FOX.__path__[0], 'examples')
    s = FOX.get_template('armc.yaml', path=examples)
    s.psf.str_file = join(examples, s.psf.str_file)
    s.molecule = FOX.MultiMolecule.from_xyz(FOX.get_example_xyz())
    armc = FOX.ARMC.from_dict(s)

    ref_dict = Settings()
    ref_dict.acceptance.shape = 500, 100
    ref_dict.acceptance.dtype = bool
    ref_dict.aux_error.shape = 500, 100, 1
    ref_dict.aux_error.dtype = np.float
    ref_dict.aux_error_mod.shape = 500, 100, 15
    ref_dict.aux_error_mod.dtype = np.float
    ref_dict.param.shape = 500, 100, 14
    ref_dict.param.dtype = np.float
    ref_dict.phi.shape = (500, )
    ref_dict.phi.dtype = np.float
    ref_dict.rdf.shape = 500, 100, 241, 6
    ref_dict.rdf.dtype = np.float
    ref_dict['rdf.ref'].shape = 241, 6
    ref_dict['rdf.ref'].dtype = np.float

    create_hdf5(hdf5_path, armc)
    with h5py.File(hdf5_path, 'r') as f:
        for key, value in f.items():
            assert value.shape == ref_dict[key].shape
            assert isinstance(value[:].item(0), ref_dict[key].dtype)
    remove(hdf5_path)


def test_to_hdf5():
    """Test :meth:`FOX.io.hdf5_utils.to_hdf5`."""
    hdf5_path = join(REF_DIR, 'test.hdf5')
    try:
        remove(hdf5_path)
    except FileNotFoundError:
        pass

    examples = join(FOX.__path__[0], 'examples')
    s = FOX.get_template('armc.yaml', path=examples)
    s.psf.str_file = join(examples, s.psf.str_file)
    s.molecule = FOX.MultiMolecule.from_xyz(FOX.get_example_xyz())
    armc = FOX.ARMC.from_dict(s)

    kappa = 5
    omega = 15
    hdf5_dict = {
        'xyz': FOX.MultiMolecule.from_xyz(FOX.get_example_xyz()),
        'phi': 5.0,
        'param': np.arange(14, dtype=float),
        'acceptance': True,
        'aux_error': np.array([2.0], ndmin=1),
    }
    hdf5_dict['rdf'] = hdf5_dict['xyz'].init_rdf(atom_subset=['Cd', 'Se', 'O']).values
    hdf5_dict['aux_error_mod'] = np.append(hdf5_dict['param'], hdf5_dict['phi'])
    hdf5_dict['xyz'] = s.molecule

    create_hdf5(hdf5_path, armc)
    create_xyz_hdf5(hdf5_path, s.molecule.as_Molecule(0)[0], 100)
    to_hdf5(hdf5_path, hdf5_dict, kappa, omega)
    with h5py.File(hdf5_path, 'r') as f:
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
    remove(hdf5_path)
    remove(hdf5_path.replace('.hdf5', '.xyz.hdf5'))


def test_from_hdf5():
    """ Test :meth:`FOX.io.hdf5_utils.from_hdf5`. """
    hdf5_path = join(REF_DIR, 'test.hdf5')
    try:
        remove(hdf5_path)
    except FileNotFoundError:
        pass

    examples = join(FOX.__path__[0], 'examples')
    s = FOX.get_template('armc.yaml', path=examples)
    s.psf.str_file = join(examples, s.psf.str_file)
    s.molecule = FOX.MultiMolecule.from_xyz(FOX.get_example_xyz())
    armc = FOX.ARMC.from_dict(s)

    kappa = 0
    omega = 0
    hdf5_dict = {
        'xyz': FOX.MultiMolecule.from_xyz(FOX.get_example_xyz()),
        'phi': 5.0,
        'param': np.arange(14, dtype=float),
        'acceptance': True,
        'aux_error': np.array([2.0], ndmin=1),
    }
    hdf5_dict['rdf'] = hdf5_dict['xyz'].init_rdf(atom_subset=['Cd', 'Se', 'O']).values
    hdf5_dict['aux_error_mod'] = np.append(hdf5_dict['param'], hdf5_dict['phi'])
    hdf5_dict['xyz'] = s.molecule

    create_hdf5(hdf5_path, armc)
    create_xyz_hdf5(hdf5_path, s.molecule.as_Molecule(0)[0], 100)
    to_hdf5(hdf5_path, hdf5_dict, kappa, omega)
    out = from_hdf5(hdf5_path)

    assert hdf5_dict['acceptance'] == out['acceptance'][0]
    assert hdf5_dict['aux_error'] == out['aux_error']['rdf'][0]
    assert hdf5_dict['phi'] == out['aux_error_mod'].values[0]
    assert hdf5_dict['phi'] == out['phi'][0][0]
    np.testing.assert_allclose(hdf5_dict['param'], out['param'].values[0])
    np.testing.assert_allclose(hdf5_dict['rdf'], out['rdf'][0].values)

    remove(hdf5_path)
    remove(hdf5_path.replace('.hdf5', '.xyz.hdf5'))
