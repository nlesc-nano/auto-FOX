"""A module for testing example input files in the FOX/examples directory."""

import time
import os
from os.path import join

import yaml
import numpy as np
import pandas as pd

from scm.plams import Cp2kJob
from assertionlib import assertion

from FOX import ARMC, MultiMolecule, example_xyz
from FOX.io.read_psf import PSFContainer

PATH = join('tests', 'test_files')


def test_input():
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
    yaml_file = join(PATH, 'armc.yaml')
    armc, job_kwarg = ARMC.from_yaml(yaml_file)

    assertion.eq(armc.a_target, 0.25)
    assertion.is_(armc.apply_move.func, np.multiply)
    assertion.is_(armc.apply_phi, np.add)
    assertion.eq(armc.gamma, 2.0)
    assertion.eq(armc.hdf5_file, 'armc.hdf5')
    assertion.eq(armc.history_dict, {})
    assertion.eq(armc.iter_len, 50000)
    assertion.eq(armc.job_cache, [])

    assertion.is_(armc.job_type.func, Cp2kJob)
    assertion.eq(armc.job_type.keywords, {'name': 'armc'})

    assertion.eq(armc.keep_files, False)
    assertion.isinstance(armc.molecule, tuple)
    assertion.len(armc.molecule)
    for mol in armc.molecule:
        assertion.isinstance(mol, MultiMolecule)

    s = armc.md_settings[0].copy()
    del s.input.force_eval.mm.forcefield.parm_file_name
    del s.input.force_eval.subsys.topology.conn_file_name
    with open(join(PATH, 'armc_md_settings.yaml'), 'r') as f:
        ref = yaml.load(f, Loader=yaml.FullLoader)
        assertion.eq(s, ref)

    np.testing.assert_allclose(
        armc.move_range,
        np.array([0.9, 0.905, 0.91, 0.915, 0.92, 0.925, 0.93, 0.935, 0.94,
                  0.945, 0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985,
                  0.99, 0.995, 1.005, 1.01, 1.015, 1.02, 1.025, 1.03, 1.035,
                  1.04, 1.045, 1.05, 1.055, 1.06, 1.065, 1.07, 1.075, 1.08,
                  1.085, 1.09, 1.095, 1.1])
    )

    param_ref = pd.read_csv(join(PATH, 'armc_param.csv'), index_col=[0, 1], float_precision='high')
    param_ref['constraints'] = None
    param_ref['keys'] = [eval(i) for i in param_ref['keys']]
    for k, v1 in param_ref.items():
        v2 = armc.param[k]
        if k == 'param_old':
            assertion.is_(v1.isna().all(), v2.isna().all())
        elif v1.dtype.name == 'float64':
            np.testing.assert_allclose(v1, v2)
        else:
            np.testing.assert_array_equal(v1, v2)

    assertion.isinstance(armc.pes, dict)
    assertion.contains(armc.pes, 'rdf.0')
    assertion.is_(armc.pes['rdf.0'].func, MultiMolecule.init_rdf)
    assertion.eq(armc.pes['rdf.0'].keywords, {'atom_subset': ['Cd', 'Se', 'O']})

    assertion.eq(armc.phi, 1.0)
    assertion.is_(armc.preopt_settings, None)
    assertion.eq(armc.rmsd_threshold, 10.0)
    assertion.eq(armc.sub_iter_len, 100)

    assertion.eq(job_kwarg.logfile, 'armc.log')
    assertion.eq(job_kwarg.path, os.getcwd())
    assertion.eq(job_kwarg.folder, 'MM_MD_workdir')
    for psf in job_kwarg.psf:
        assertion.isinstance(psf, PSFContainer)
