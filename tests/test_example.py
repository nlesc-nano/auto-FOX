""" A module for testing example input files in the FOX/examples directory. """

__all__ = []

import time
from os.path import join

import numpy as np
import pandas as pd

import FOX


REF_DIR = 'tests/test_files'
REF_DIR = '/Users/basvanbeek/Documents/GitHub/auto-FOX/tests/test_files'


def test_input():
    """ Test :mod:`FOX.examples.input`. """
    # Define the atoms of interest and the .xyz path + filename
    atoms = ('Cd', 'Se', 'O')
    example_xyz_filename = FOX.get_example_xyz()

    # Optional: start the timer
    print('')
    start = time.time()

    # Read the .xyz file
    mol = FOX.MultiMolecule.from_xyz(example_xyz_filename)

    # Calculate the RDF, RSMF & RMSD
    rdf = mol.init_rdf(atom_subset=atoms)
    rmsf = mol.init_rmsf(atom_subset=atoms)
    rmsd = mol.init_rmsd(atom_subset=atoms)

    # Optional: print the results and try to plot them in a graph (if Matplotlib is installed)
    print('run time:', '%.2f' % (time.time() - start), 'sec')
    try:
        rdf.plot()
        rmsf.plot()
        rmsd.plot()
    except Exception as ex:
        print(ex)

    ref_rdf = np.load(join(REF_DIR, 'rdf.npy'))
    ref_rmsf = np.load(join(REF_DIR, 'rmsf.npy'))
    ref_rmsd = np.load(join(REF_DIR, 'rmsd.npy'))

    np.testing.assert_allclose(rdf.values, ref_rdf)
    np.testing.assert_allclose(rmsf.values, ref_rmsf)
    np.testing.assert_allclose(rmsd.values, ref_rmsd)


def test_cp2k_md():
    """ Test :mod:`FOX.examples.cp2k_md`. """
    examples = join(FOX.__path__[0], 'examples')
    s = FOX.get_template('armc.yaml', path=examples)
    s.psf.str_file = join(examples, s.psf.str_file)
    s.molecule = FOX.MultiMolecule.from_xyz(FOX.get_example_xyz())
    armc = FOX.ARMC.from_dict(s)

    psf = {
        'atoms': pd.read_csv(join(REF_DIR, 'psf_atoms.csv'), float_precision='high', index_col=0),
        'bonds': np.load(join(REF_DIR, 'bonds.npy')),
        'angles': np.load(join(REF_DIR, 'angles.npy')),
        'dihedrals': np.load(join(REF_DIR, 'dihedrals.npy')),
        'impropers': np.load(join(REF_DIR, 'impropers.npy')),
        }

    np.testing.assert_allclose(armc.job.psf.bonds, psf['bonds'])
    np.testing.assert_allclose(armc.job.psf.angles, psf['angles'])
    np.testing.assert_allclose(armc.job.psf.dihedrals, psf['dihedrals'])
    np.testing.assert_allclose(armc.job.psf.impropers, psf['impropers'])
    for key, value in armc.job.psf.atoms.items():
        if not value.dtype.name == 'object':
            np.testing.assert_allclose(value, psf['atoms'][key])
        else:
            np.testing.assert_array_equal(value, psf['atoms'][key])

    assert armc.phi.phi == 1.0
    assert armc.phi.kwarg == {}
    assert armc.phi.func == np.add

    assert armc.armc.a_target == 0.25
    assert armc.armc.gamma == 2.0
    assert armc.armc.iter_len == 50000
    assert armc.armc.sub_iter_len == 100
