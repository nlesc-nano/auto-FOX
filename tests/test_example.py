"""A module for testing example input files in the FOX/examples directory."""

import os
import functools
from os.path import join
from logging import Logger

import yaml
import numpy as np

from assertionlib import assertion
from qmflows import cp2k_mm, Settings as QmSettings

from FOX import MultiMolecule, PSFContainer
from FOX.testing_utils import validate_mapping
from FOX.armc import dict_to_armc, ARMC, PhiUpdater

PATH = join('tests', 'test_files')


def test_cp2k_md():
    """Test :mod:`FOX.examples.cp2k_md`."""
    file = join(PATH, 'armc.yaml')

    with open(file, 'r') as f:
        dct = yaml.load(f.read(), Loader=yaml.SafeLoader)
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
    assertion.eq(armc.pes['rdf.0'].args, (MultiMolecule.init_rdf,))

    phi_ref = PhiUpdater(a_target=0.25, gamma=2.0, phi=1.0, func=np.add)
    assertion.eq(armc.phi, phi_ref)

    assertion.eq(armc.sub_iter_len, 100)

    for i in job_kwarg['psf']:
        assertion.isinstance(i, PSFContainer)
    assertion.eq(job_kwarg['folder'], 'MM_MD_workdir')
    assertion.assert_(str.endswith, job_kwarg['path'],
                      join('tests', 'test_files'))
    assertion.assert_(str.endswith, job_kwarg['logfile'],
                      join('tests', 'test_files', 'MM_MD_workdir', 'armc.log'))
