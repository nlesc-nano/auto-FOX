"""Tests for :class:`FOX.io.read_psf.PSFContainer`."""

import os
from pathlib import Path
from tempfile import TemporaryFile
from itertools import zip_longest

import pytest
import numpy as np
from scm.plams import Molecule
from assertionlib import assertion

from FOX import PSFContainer

PATH = Path('tests') / 'test_files' / 'psf'
PSF = PSFContainer.read(PATH / 'mol.psf')

MOL = Molecule(PATH / 'mol.pdb')
MOL.guess_bonds(atom_subset=[at for at in MOL if at.symbol in ('C', 'O', 'H')])


def test_write() -> None:
    """Tests for :meth:`PSFContainer.write`."""
    filename1 = PATH / 'mol.psf'
    filename2 = PATH / 'tmp.psf'

    try:
        PSF.write(filename2)
        with open(filename1) as f1, open(filename2) as f2:
            for i, j in zip_longest(f1, f2):
                assertion.eq(i, j)
    finally:
        if os.path.isfile(filename2):
            os.remove(filename2)

    with open(filename1, 'rb') as f1, TemporaryFile() as f2:
        PSF.write(f2, mode='wb', bytes_encoding='utf-8')
        f2.seek(0)

        linesep = os.linesep.encode()
        iterator = ((i.rstrip(linesep), j.rstrip(linesep)) for i, j in zip_longest(f1, f2))
        for i, j in iterator:
            assertion.eq(i, j)


def test_update_atom_charge() -> None:
    """Tests for :meth:`PSFContainer.update_atom_charge`."""
    psf = PSF.copy()
    psf.update_atom_charge('C2O3', -5.0)
    condition = psf.atom_type == 'C2O3'

    assert (psf.charge[condition] == -5.0).all()
    assertion.assert_(psf.update_atom_charge, 'C2O3', 'bob', exception=ValueError)


def test_update_atom_type() -> None:
    """Tests for :meth:`PSFContainer.update_atom_type`."""
    psf = PSF.copy()
    psf.update_atom_type('C2O3', 'C8')
    assertion.contains(psf.atom_type.values, 'C8')


def test_generate_bonds() -> None:
    """Tests for :meth:`PSFContainer.generate_bonds`."""
    psf = PSF.copy()
    ref = np.load(PATH / 'generate_bonds.npy')
    psf.generate_bonds(MOL)
    np.testing.assert_array_equal(ref, psf.bonds)


def test_generate_angles() -> None:
    """Tests for :meth:`PSFContainer.generate_angles`."""
    psf = PSF.copy()
    ref = np.load(PATH / 'generate_angles.npy')
    psf.generate_angles(MOL)
    np.testing.assert_array_equal(ref, psf.angles)


def test_generate_dihedrals() -> None:
    """Tests for :meth:`PSFContainer.generate_dihedrals`."""
    psf = PSF.copy()
    ref = np.load(PATH / 'generate_dihedrals.npy')
    psf.generate_dihedrals(MOL)
    np.testing.assert_array_equal(ref, psf.dihedrals)


def test_generate_impropers() -> None:
    """Tests for :meth:`PSFContainer.generate_impropers`."""
    psf = PSF.copy()
    ref = np.load(PATH / 'generate_impropers.npy')
    psf.generate_impropers(MOL)
    np.testing.assert_array_equal(ref, psf.impropers)


def test_to_atom_alias_dict() -> None:
    """Tests for :meth:`PSFContainer.to_atom_alias_dict`."""
    dct = PSF.to_atom_alias_dict()
    for at1, (at2, idx) in dct.items():
        at2_slice = PSF.atom_type[PSF.atom_name == at2]
        np.testing.assert_array_equal(at2_slice.iloc[idx], at1)


def test_sort_values() -> None:
    """Tests for :meth:`PSFContainer.sort_values`."""
    psf = PSF.copy()
    assertion.is_(psf, psf.sort_values([], inplace=True))

    argsort_ref = np.load(PATH / "sort_values_argsort.npy")
    psf2_ref = PSFContainer.read(PATH / 'sort_values.psf')

    psf2, argsort = psf.sort_values(["residue ID", "mass"], return_argsort=True)
    np.testing.assert_array_equal(argsort, argsort_ref)
    assertion.eq(psf2, psf2_ref)

    with pytest.raises(TypeError):
        psf.sort_values([], axis=0)
