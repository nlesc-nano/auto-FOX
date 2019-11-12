"""A module for testing example input files in the FOX/examples directory."""

import time
from os.path import join

import numpy as np
import pandas as pd

from assertionlib import assertion

from FOX import ARMC, MultiMolecule, get_example_xyz

PATH = join('tests', 'test_files')
PATH = '/Users/bvanbeek/Documents/GitHub/auto-FOX/tests/test_files'


def test_input():
    """Test :mod:`FOX.examples.input`."""
    rdf = rmsf = rmsd = None

    # Define the atoms of interest and the .xyz path + filename
    atoms = ('Cd', 'Se', 'O')
    example_xyz_filename = get_example_xyz()

    # Optional: start the timer
    start = time.time()

    # Read the .xyz file
    mol = MultiMolecule.from_xyz(example_xyz_filename)

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
    armc, job_kwarg = ARMC.from_yaml(join(PATH, 'armc.yaml'))

