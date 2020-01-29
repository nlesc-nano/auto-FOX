"""A module for testing the :class:`FOX.classes.multi_mol.MultiMolecule` class."""

from os import remove
from os.path import join

import numpy as np

from assertionlib import assertion

from FOX import MultiMolecule, example_xyz
from FOX.functions.utils import get_template

MOL = MultiMolecule.from_xyz(example_xyz)
PATH = join('tests', 'test_files')
PATH = '/Users/basvanbeek/Documents/GitHub/auto-FOX/tests/test_files'


def test_delete_atoms():
    """Test :meth:`.MultiMolecule.delete_atoms`."""
    mol = MOL.copy()

    atoms = ('H', 'C', 'O')
    mol_new = mol.delete_atoms(atom_subset=atoms)
    assertion.eq(mol_new.shape, (4905, 104, 3))
    ref = np.load(join(PATH, 'delete_atoms.npy'))
    np.testing.assert_array_equal(mol_new.symbol, ref)


def test_guess_bonds():
    """Test :meth:`.MultiMolecule.guess_bonds`."""
    mol = MOL.copy()

    atoms = ('H', 'C', 'O')
    mol.guess_bonds(atom_subset=atoms)
    ref = np.load(join(PATH, 'guess_bonds.npy'))
    np.testing.assert_allclose(mol.bonds, ref)


def test_random_slice():
    """Test :meth:`.MultiMolecule.random_slice`."""
    mol = MOL.copy()

    assertion.assert_(mol.random_slice, start=0, stop=None, p=1.0,
                      inplace=True, exception=ValueError)

    mol_new = mol.random_slice(start=0, stop=None, p=0.5)
    assertion.eq(mol_new.shape[1:2], mol.shape[1:2])
    assertion.eq(mol_new.shape[0], mol.shape[0] // 2)


def test_reset_origin():
    """Test :meth:`.MultiMolecule.reset_origin`."""
    mol = MOL.copy()

    mol.reset_origin()
    assertion.allclose(mol.mean(axis=1).sum(), 0.0, abs_tol=10**-8)

    mol_new = mol.reset_origin(mol_subset=1000, inplace=False)
    assertion.allclose(mol_new[0:1000].mean(axis=1).sum(), 0.0, abs_tol=10**-8)

    mol_new = mol.reset_origin(atom_subset=slice(0, 100), inplace=False)
    assertion.allclose(mol_new[:, 0:100].mean(axis=1).sum(), 0.0, abs_tol=10**-8)


def test_sort():
    """Test :meth:`.MultiMolecule.sort`."""
    mol = MOL.copy()

    mol.sort(sort_by='symbol')
    np.testing.assert_array_equal(mol.symbol, np.sort(mol.symbol))

    mol.sort(sort_by='atnum')
    np.testing.assert_allclose(mol.atnum, np.sort(mol.atnum))

    mol.sort(sort_by='mass')
    np.testing.assert_allclose(mol.mass, np.sort(mol.mass))

    mol.sort(sort_by='radius')
    np.testing.assert_allclose(mol.radius, np.sort(mol.radius))

    mol.sort(sort_by='connectors')
    np.testing.assert_allclose(mol.connectors, np.sort(mol.connectors))


def test_residue_argsort():
    """Test :meth:`.MultiMolecule.residue_argsort`."""
    mol = MOL.copy()

    atoms = ('H', 'C', 'O')
    mol.guess_bonds(atom_subset=atoms)
    idx = mol.residue_argsort()
    ref = np.load(join(PATH, 'residue_argsort.npy'))
    np.testing.assert_allclose(idx, ref)


def test_get_center_of_mass():
    """Test :meth:`.MultiMolecule.sort`."""
    mol = MOL.copy()

    center_of_mass = mol.get_center_of_mass()
    mol -= center_of_mass[:, None, :]
    center_of_mass = mol.get_center_of_mass()
    assertion.allclose(np.abs(center_of_mass.mean()), 0.0, abs_tol=10**-8)


def test_get_bonds_per_atom():
    """Test :meth:`.MultiMolecule.get_bonds_per_atom`."""
    mol = MOL.copy()

    atoms = ('H', 'C', 'O')
    mol.guess_bonds(atom_subset=atoms)
    bond_count = mol.get_bonds_per_atom()
    ref = np.load(join(PATH, 'get_bonds_per_atom.npy'))
    np.testing.assert_allclose(bond_count, ref)


def test_power_spectrum():
    """Test :meth:`.MultiMolecule.init_power_spectrum`."""
    mol = MOL.copy()

    p = mol.init_power_spectrum()
    p_ref = np.load(join(PATH, 'power_spectrum.npy'))
    np.testing.assert_allclose(p, p_ref)


def test_vacf():
    """Test :meth:`.MultiMolecule.get_vacf`."""
    mol = MOL.copy()

    vacf = mol.get_vacf()
    vacf_ref = np.load(join(PATH, 'vacf.npy'))
    np.testing.assert_allclose(vacf, vacf_ref, rtol=1e-06)


def test_rdf():
    """Test :meth:`.MultiMolecule.init_rdf`."""
    mol = MOL.copy()

    atoms = ('Cd', 'Se', 'O')
    rdf1 = mol.init_rdf(atom_subset=atoms, mem_level=0).values
    rdf2 = mol.init_rdf(atom_subset=atoms, mem_level=1).values
    rdf3 = mol.init_rdf(atom_subset=atoms, mem_level=2).values
    ref = np.load(join(PATH, 'rdf.npy'))
    np.testing.assert_allclose(rdf1, ref)
    np.testing.assert_allclose(rdf2, ref)
    np.testing.assert_allclose(rdf3, ref)

    rdf4 = mol.init_rdf(mol_subset=slice(None, None, 10), mem_level=1).values
    rdf5 = mol.init_rdf(mol_subset=slice(None, None, 100), mem_level=1).values
    ref4 = np.load(join(PATH, 'rdf_10.npy'))
    ref5 = np.load(join(PATH, 'rdf_100.npy'))
    np.testing.assert_allclose(rdf4, ref4)
    np.testing.assert_allclose(rdf5, ref5)


def test_rmsf():
    """Test :meth:`.MultiMolecule.init_rmsf`."""
    mol = MOL.copy()

    atoms = ('Cd', 'Se', 'O')
    rmsf = mol.init_rmsf(atom_subset=atoms).values
    np.nan_to_num(rmsf, copy=False)
    ref = np.load(join(PATH, 'rmsf.npy'))
    np.nan_to_num(ref, copy=False)
    np.testing.assert_allclose(rmsf, ref)


def test_rmsd():
    """Test :meth:`.MultiMolecule.init_rmsd`."""
    mol = MOL.copy()

    atoms = ('Cd', 'Se', 'O')
    rmsd = mol.init_rmsd(atom_subset=atoms).values
    ref = np.load(join(PATH, 'rmsd.npy'))
    np.testing.assert_allclose(rmsd, ref)


def test_time_averaged_velocity():
    """Test :meth:`.MultiMolecule.init_time_averaged_velocity`."""
    mol = MOL.copy()

    atoms = ('Cd', 'Se', 'O')
    v = mol.init_time_averaged_velocity(atom_subset=atoms).values
    np.nan_to_num(v, copy=False)
    ref = np.load(join(PATH, 'time_averaged_velocity.npy'))
    np.nan_to_num(ref, copy=False)
    np.testing.assert_allclose(v, ref)


def test_average_velocity():
    """Test :meth:`.MultiMolecule.init_average_velocity`."""
    mol = MOL.copy()

    atoms = ('Cd', 'Se', 'O')
    v = mol.init_average_velocity(atom_subset=atoms).values
    ref = np.load(join(PATH, 'average_velocity.npy'))
    np.testing.assert_allclose(v, ref)


def test_adf():
    """Test :meth:`.MultiMolecule.init_adf`."""
    mol = MOL.copy()

    atoms = ('Cd', 'Se')

    adf1 = mol.init_adf(atom_subset=atoms).values
    ref1 = np.load(join(PATH, 'adf_weighted.npy'))
    np.testing.assert_allclose(adf1, ref1)

    adf2 = mol.init_adf(atom_subset=atoms, weight=None).values
    ref2 = np.load(join(PATH, 'adf.npy'))
    np.testing.assert_allclose(adf2, ref2)


def test_shell_search():
    """Test :meth:`.MultiMolecule.init_shell_search`."""
    mol = MOL.copy()

    rmsf, idx_series, rdf = mol.init_shell_search()
    np.nan_to_num(rmsf, copy=False)

    ref_rmsf = np.load(join(PATH, 'shell_rmsf.npy'))
    np.nan_to_num(ref_rmsf, copy=False)
    ref_idx = np.load(join(PATH, 'shell_idx.npy'))
    ref_rdf = np.load(join(PATH, 'shell_rdf.npy'))

    np.testing.assert_allclose(ref_rmsf, rmsf)
    np.testing.assert_allclose(ref_idx, idx_series)
    np.testing.assert_allclose(ref_rdf, rdf)


def test_get_at_idx():
    """Test :meth:`.MultiMolecule.get_at_idx`."""
    mol = MOL.copy()

    rmsf, idx_series, _ = mol.init_shell_search()
    dist = 3.0, 6.5, 10.0
    dist_dict = {'Cd': dist, 'Se': dist, 'O': dist, 'C': dist, 'H': dist}
    dict_ = mol.get_at_idx(rmsf, idx_series, dist_dict)
    ref = get_template('idx_series.yaml', path=PATH)
    for key in dict_:
        assertion.eq(dict_[key], ref[key])


def test_as_mass_weighted():
    """Test :meth:`.MultiMolecule.as_mass_weighted`."""
    mol = MOL.copy()

    mol_new = mol.as_mass_weighted()
    mol *= mol.mass[None, :, None]
    np.testing.assert_allclose(mol, mol_new)


def test_from_mass_weighted():
    """Test :meth:`.MultiMolecule.from_mass_weighted`."""
    mol = MOL.copy()

    mol_new = mol.as_mass_weighted()
    mol_new.from_mass_weighted()
    assertion.allclose(np.abs((mol_new - mol).mean()), 0.0, abs_tol=10**-8)


def test_as_Molecule():
    """Test :meth:`.MultiMolecule.as_Molecule`."""
    mol = MOL.copy()

    mol_list = mol.as_Molecule()
    mol_array = np.array([i.as_array() for i in mol_list])
    np.testing.assert_allclose(mol_array, mol)


def test_from_Molecule():
    """Test :meth:`.MultiMolecule.from_Molecule`."""
    mol = MOL.copy()

    mol_list = mol.as_Molecule()
    mol_new = MultiMolecule.from_Molecule(mol_list)
    np.testing.assert_allclose(mol_new, mol)


def test_as_xyz():
    """Test :meth:`.MultiMolecule.as_xyz`."""
    mol = MOL.copy()

    xyz = join(PATH, 'mol.xyz')
    mol.as_xyz(filename=xyz)
    mol_new = MultiMolecule.from_xyz(xyz)
    remove(xyz)
    np.testing.assert_allclose(mol_new, mol)


def test_from_xyz():
    """Test :meth:`.MultiMolecule.from_xyz`."""
    mol = MOL.copy()

    mol_new = MultiMolecule.from_xyz(example_xyz)
    np.testing.assert_allclose(mol_new, mol)
