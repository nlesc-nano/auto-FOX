"""A module for testing the :class:`FOX.classes.multi_mol.MultiMolecule` class."""

from __future__ import annotations

import os
from os.path import join
from pathlib import Path
from itertools import islice, chain, combinations
from typing import Mapping, Any, Type, Set, Iterable

import pytest
import yaml
import numpy as np

from assertionlib import assertion
from nanoutils import delete_finally, Literal

from FOX import MultiMolecule, example_xyz


def _read_lattice(file: str | bytes | os.PathLike[Any]) -> np.ndarray[Any, np.dtype[np.float64]]:
    with open(file, 'r') as f:
        iterator = chain.from_iterable(i.split()[2:11] for i in islice(f, 1, None))
        return np.fromiter(iterator, dtype=np.float64).reshape(-1, 3, 3)


PATH = Path('tests') / 'test_files'

MOL = MultiMolecule.from_xyz(example_xyz)
MOL.guess_bonds(['C', 'H', 'O'])

MOL_ALIAS = MOL.copy()
MOL_ALIAS.atoms_alias = {"Cd2": ("Cd", np.s_[:10])}

MOL_LATTICE_3D = MultiMolecule.from_xyz(PATH / "md_lattice.xyz")[::10]
MOL_LATTICE_3D.lattice = _read_lattice(PATH / "md_lattice.cell")[::10]

MOL_LATTICE_2D = MOL_LATTICE_3D.copy()
MOL_LATTICE_2D.lattice = MOL_LATTICE_2D.lattice[0]

MOL_LATTICE_3D_ORTH = MOL_LATTICE_3D.copy()
MOL_LATTICE_3D_ORTH.lattice = np.zeros_like(MOL_LATTICE_3D_ORTH.lattice)
MOL_LATTICE_3D_ORTH.lattice[..., 0] = np.linalg.norm(MOL_LATTICE_3D.lattice, axis=-1)

MOL_LATTICE_2D_ORTH = MOL_LATTICE_3D.copy()
MOL_LATTICE_2D_ORTH.lattice = np.zeros_like(MOL_LATTICE_2D_ORTH.lattice)
MOL_LATTICE_2D_ORTH.lattice[..., 0] = np.linalg.norm(MOL_LATTICE_2D.lattice, axis=-1)


@delete_finally(join(PATH, '.tmp.xyz'))
def test_mol_to_file():
    """Test :meth:`MultiMolecule._mol_to_file`."""
    file = join(PATH, '.tmp.xyz')
    MOL._mol_to_file(file, 'xyz')

    mol = MultiMolecule.from_xyz(file)
    np.testing.assert_allclose(MOL[0][None], mol, atol=1e-6)


class TestDeleteAtoms:
    """Test :meth:`.MultiMolecule.delete_atoms`."""

    @pytest.mark.parametrize(
        "name,mol,atoms",
        [
            ("delete_atoms", MOL, {'H', 'C', 'O'}),
            ("delete_atoms_alias", MOL_ALIAS, {'Cd2'}),
            ("delete_atoms_alias2", MOL_ALIAS, {'Cd'}),
        ]
    )
    def test_passes(self, name: str, mol: MultiMolecule, atoms: Set[str]) -> None:
        mol_new = mol.delete_atoms(atom_subset=atoms)
        ref = np.load(PATH / f'{name}.npy')

        assertion.isdisjoint(mol_new.symbol, atoms)
        assertion.isdisjoint(mol_new.atoms_alias, atoms)
        np.testing.assert_array_equal(mol_new.symbol, ref)

        for k1, (k2, _) in mol.atoms_alias.items():
            if k2 in atoms:
                assertion.contains(mol_new.atoms_alias, k1, invert=True)

    def test_raises(self) -> None:
        with pytest.raises(TypeError):
            MOL.delete_atoms(atom_subset=None)


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


class TestRDF:
    """Test :meth:`.MultiMolecule.init_rdf`."""

    @pytest.mark.parametrize(
        "kwargs,filename",
        [
            ({'atom_subset': ('Cd', 'Se', 'O')}, 'rdf.npy'),
            ({'mol_subset': np.s_[::10]}, 'rdf_10.npy'),
            ({'mol_subset': np.s_[::100]}, 'rdf_100.npy')
        ]
    )
    def test_passes(self, kwargs: Mapping[str, Any], filename: str) -> None:
        rdf = MOL.init_rdf(**kwargs)
        ref = rdf.copy()
        ref[:] = np.load(join(PATH, filename))
        np.testing.assert_allclose(rdf, ref)

    @pytest.mark.parametrize("mol", [MOL_LATTICE_3D, MOL_LATTICE_2D])
    @pytest.mark.parametrize(
        "periodic",
        chain([None], combinations("xyz", 1), combinations("xyz", 2), combinations("xyz", 3)),
    )
    def test_lattice(
        self, mol: MultiMolecule, periodic: None | Iterable[Literal["x", "y", "z"]]
    ) -> None:
        assert mol.lattice is not None
        rdf = mol.init_rdf(periodic=periodic)

        if periodic is None:
            filename = f"test_lattice_{mol.lattice.ndim}d.npy"
        else:
            filename = f"test_lattice_{mol.lattice.ndim}d_{''.join(sorted(periodic))}.npy"
        ref = np.load(PATH / filename)

        np.testing.assert_allclose(rdf, ref)

    @pytest.mark.parametrize(
        "mol,kwargs,exc",
        [
            (MOL_LATTICE_3D, {"periodic": "bob"}, ValueError),
            (MOL, {"periodic": "xyz"}, TypeError),
        ]
    )
    def test_raises(
        self, mol: MultiMolecule, kwargs: Mapping[str, Any], exc: Type[Exception]
    ) -> None:
        with pytest.raises(exc):
            mol.init_rdf(**kwargs)


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


class TestADF:
    """Test :meth:`MultiMolecule.init_adf`."""

    @pytest.mark.parametrize(
        "name,mol,kwargs",
        [
            ("adf_weighted", MOL, {"atom_subset": ("Cd", "Se")}),
            ("adf", MOL, {"atom_subset": ("Cd", "Se"), "weight": None}),
            ("adf_periodic_2d", MOL_LATTICE_2D_ORTH, {"atom_subset": ("Pb",), "periodic": "xyz"}),
            ("adf_periodic_3d", MOL_LATTICE_3D_ORTH, {"atom_subset": ("Pb",), "periodic": "xy"}),
        ],
        ids=["adf_weighted", "adf", "adf_periodic_2d", "adf_periodic_3d"],
    )
    def test_passes(self, name: str, mol: MultiMolecule, kwargs: Mapping[str, Any]) -> None:
        adf = mol.init_adf(**kwargs)
        ref = np.load(PATH / f"{name}.npy")
        np.testing.assert_allclose(adf, ref)

    @pytest.mark.parametrize(
        "kwargs,exc",
        [
            ({"periodic": "bob"}, ValueError),
            ({"periodic": "xyz", "r_max": np.inf}, NotImplementedError),
            ({"periodic": "xyz"}, NotImplementedError),
        ]
    )
    @pytest.mark.parametrize("mol", [MOL_LATTICE_2D, MOL_LATTICE_3D])
    def test_raises(
        self, mol: MultiMolecule, kwargs: Mapping[str, Any], exc: Type[Exception]
    ) -> None:
        with pytest.raises(exc):
            mol.init_adf(**kwargs)


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

    dct = mol.get_at_idx(rmsf, idx_series, dist_dict)
    with open(join(PATH, 'idx_series.yaml'), 'r') as f:
        ref = yaml.load(f.read(), Loader=yaml.FullLoader)

    for key, value in dct.items():
        value_ref = ref.get(key)
        assertion.eq(value, value_ref)


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


def test_as_molecule():
    """Test :meth:`.MultiMolecule.as_Molecule`."""
    mol = MOL.copy()

    mol_list = mol.as_Molecule()
    mol_array = np.array([i.as_array() for i in mol_list])
    np.testing.assert_allclose(mol_array, mol)


def test_from_molecule():
    """Test :meth:`.MultiMolecule.from_Molecule`."""
    mol = MOL.copy()

    mol_list = mol.as_Molecule()
    mol_new = MultiMolecule.from_Molecule(mol_list)
    np.testing.assert_allclose(mol_new, mol)


@delete_finally(join(PATH, 'mol.xyz'))
def test_as_xyz():
    """Test :meth:`.MultiMolecule.as_xyz`."""
    mol = MOL.copy()

    xyz = join(PATH, 'mol.xyz')
    mol.as_xyz(filename=xyz)
    mol_new = MultiMolecule.from_xyz(xyz)
    np.testing.assert_allclose(mol_new, mol)


def test_from_xyz():
    """Test :meth:`.MultiMolecule.from_xyz`."""
    mol = MOL.copy()

    mol_new = MultiMolecule.from_xyz(example_xyz)
    np.testing.assert_allclose(mol_new, mol)


def test_properties():
    """Test :class:`MultiMolecule` properties."""
    mol = MOL.copy()

    order_ref1 = mol.bonds[:, 2] / 10.0
    np.testing.assert_array_equal(mol.order, order_ref1)

    order_ref2 = np.arange(len(mol.bonds))
    mol.order = order_ref2
    np.testing.assert_array_equal(mol.order, order_ref2)

    np.testing.assert_array_equal(mol.x, mol[..., 0])
    np.testing.assert_array_equal(mol.y, mol[..., 1])
    np.testing.assert_array_equal(mol.z, mol[..., 2])

    mol.x = 1
    mol.y = 2
    mol.z = 3
    np.testing.assert_array_equal(mol.x, 1)
    np.testing.assert_array_equal(mol.y, 2)
    np.testing.assert_array_equal(mol.z, 3)


def test_loc():
    """Test :attr:`MultiMolecule.loc`."""
    mol = MOL.copy()

    loc1 = mol.loc
    loc2 = mol.loc

    assertion.eq(loc1, loc2)
    assertion.eq(hash(loc1), hash(loc2))
    assertion.ne(loc1, None)
    assertion.assert_(loc1.__reduce__, exception=TypeError)

    assertion.shape_eq(mol.loc['Cd'], (4905, 68, 3))
    assertion.shape_eq(mol.loc['Cd', 'Cd'], (4905, 136, 3))
    assertion.assert_(mol.loc.__getitem__, 1, exception=TypeError)

    mol.loc['Cd'] = 1
    np.testing.assert_array_equal(mol.loc['Cd'], 1)

    assertion.assert_(mol.loc.__delitem__, 'Cd', exception=ValueError)

    mol._atoms["Xx"] = np.array([], dtype=np.intp)
    assertion.shape_eq(mol.loc["Xx"], (4905, 0, 3))


class TestAtoms:
    @pytest.mark.parametrize(
        "name,exc,dct",
        [
            ("atoms", TypeError, {"Cd": [1.0]}),
            ("atoms", ValueError, {"Cd": [[1]]}),
            ("atoms", TypeError, {"Cd": None}),
            ("atoms", TypeError, {"Cd": [True]}),
            ("atoms", ValueError, {"Cd": [1], "Se": [1]}),
            ("atoms_alias", TypeError, {"Cd2": ("Cd", [1.0])}),
            ("atoms_alias", ValueError, {"Cd2": ("Cd", [[1]])}),
            ("atoms_alias", IndexError, {"Cd2": ("Cd", range(200))}),
            ("atoms_alias", TypeError, {"Cd2": ("Cd", None)}),
            ("atoms_alias", TypeError, {"Cd2": ("Cd", [True])}),
            ("atoms_alias", KeyError, {"Cd2": ("Bob", [1])}),
            ("atoms_alias", KeyError, {"Cd": ("Cd2", [1])}),
        ]
    )
    def test_raises(self, name: str, exc: Type[Exception], dct: Mapping[str, Any]) -> None:
        mol = MOL.copy()
        with pytest.raises(exc):
            if name == "atoms":
                mol.atoms = dct
            else:
                mol.atoms_alias = dct


@pytest.mark.parametrize(
    "kwargs",
    [
        {'mol_subset': np.s_[::10]},
        {'atom_subset': ['Cd', 'Se']},
        {'mol_subset': np.s_[10:30], 'atom_subset': ['Cd', 'O']},
        {'masses': MOL.mass},
    ]
)
def test_ase(kwargs) -> None:
    """Test :meth:`MultiMolecule.as_ase` and :meth:`MultiMolecule.from_ase`."""
    mol_ref = MOL[:100]
    ase_mols = mol_ref.as_ase(**kwargs)
    mol = MultiMolecule.from_ase(ase_mols)

    i = kwargs.get('mol_subset', np.s_[:])
    j = kwargs.get('atom_subset')
    if j is None:
        np.testing.assert_allclose(mol, mol_ref[i])
    else:
        np.testing.assert_allclose(mol, mol_ref[i].loc[j])
