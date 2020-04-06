"""A module for testing example input files in the FOX/examples directory."""

import os
import time
import functools
from os.path import join
from logging import Logger

import yaml
import numpy as np

from assertionlib import assertion
from qmflows import cp2k_mm, Settings as QmSettings

from FOX import MultiMolecule, PSFContainer, example_xyz
from FOX.test_utils import validate_mapping
from FOX.armc import dict_to_armc, ARMC, PhiUpdater

PATH = join('tests', 'test_files')


def _test_input():
    """Test :mod:`FOX.examples.input`."""
    rdf = rmsf = rmsd = None

    # Define the atoms of interest and the .xyz path + filename
    atoms = ('Cd', 'Se', 'O')

    # Optional: start the timer
    start = time.time()

    # Read the .xyz file
    mol = MultiMolecule.from_xyz(example_xyz)

    # Calculate the RDF, RSMF & RMSD
    rdf = mol.init_rdf(atom_subset=atoms)
    adf = mol.init_adf(r_max=8.0, atom_subset=['Cd', 'Se'])
    rmsf = mol.init_rmsf(atom_subset=atoms)
    rmsd = mol.init_rmsd(atom_subset=atoms)

    # Optional: print the results and try to plot them in a graph (if Matplotlib is installed)
    for name, df in {'rdf ': rdf, 'adf ': adf, 'rmsf': rmsf, 'rmsd': rmsd}.items():
        try:
            df.plot(name)
        except Exception as ex:
            print(f'{name} - {repr(ex)}')
    print('run time: {:.2f} sec'.format(time.time() - start))

    ref_rdf = np.load(join(PATH, 'rdf.npy'))
    ref_rmsf = np.load(join(PATH, 'rmsf.npy'))
    ref_rmsd = np.load(join(PATH, 'rmsd.npy'))

    np.testing.assert_allclose(rdf.values, ref_rdf)
    np.testing.assert_allclose(rmsf.values, ref_rmsf)
    np.testing.assert_allclose(rmsd.values, ref_rmsd)


def test_cp2k_md():
    """Test :mod:`FOX.examples.cp2k_md`."""
    file = join(PATH, 'armc.yaml')

    with open(file, 'r') as f:
        dct = yaml.load(f.read(), Loader=yaml.FullLoader)
    armc, job_kwarg = dict_to_armc(dct)

    assertion.isinstance(armc, ARMC)
    validate_mapping(armc, key_type=tuple, value_type=np.ndarray)

    assertion.eq(armc._data, {})
    assertion.eq(armc.hdf5_file, os.path.abspath('tests/test_files/MM_MD_workdir/armc.hdf5'))
    assertion.eq(armc.iter_len, 50000)
    assertion.is_(armc.keep_files, False)
    assertion.isinstance(armc.logger, Logger)

    for mol in armc.molecule:
        assertion.isinstance(mol, MultiMolecule)
        assertion.shape_eq(mol, (4905, 227, 3))
        assertion.eq(mol.dtype, float)

    iterator = (i for v in armc.package_manager.values() for i in v)
    for job_dict in iterator:
        assertion.isinstance(job_dict['settings'], QmSettings)
        np.testing.assert_allclose(np.array(job_dict['molecule']), armc.molecule[0][0])
        assertion.is_(job_dict['type'], cp2k_mm)

    assertion.isinstance(armc.pes['rdf.0'], functools.partial)
    assertion.eq(armc.pes['rdf.0'].keywords, {'atom_subset': ['Cd', 'Se', 'O']})
    assertion.eq(armc.pes['rdf.0'].args, ())
    assertion.eq(armc.pes['rdf.0'].func, MultiMolecule.init_rdf)

    phi_ref = PhiUpdater(a_target=0.25, gamma=2.0, phi=1.0, func=np.add)
    assertion.eq(armc.phi, phi_ref)

    assertion.eq(armc.sub_iter_len, 100)

    for i in job_kwarg['psf']:
        assertion.isinstance(i, PSFContainer)
    assertion.eq(job_kwarg['folder'], 'MM_MD_workdir')
    assertion.assert_(str.endswith, job_kwarg['path'], '/tests/test_files')
    assertion.assert_(str.endswith, job_kwarg['logfile'],
                      '/tests/test_files/MM_MD_workdir/armc.log')
