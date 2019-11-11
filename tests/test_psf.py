"""Tests for :class:`FOX.io.read_psf.PSFContainer`."""

import os
from os.path import join
from tempfile import TemporaryFile
from itertools import zip_longest

import numpy as np

from scm.plams import readpdb, Molecule
from assertionlib import assertion

from FOX.io.read_psf import PSFContainer

PATH: str = join('tests', 'test_files', 'psf')
PSF: PSFContainer = PSFContainer.read(join(PATH, 'mol.psf'))
MOL: Molecule = readpdb(join(PATH, 'mol.pdb'))


def test_write() -> None:
    """Tests for :meth:`PSFContainer.write`."""
    filename1 = join(PATH, 'mol.psf')
    filename2 = join(PATH, 'tmp.psf')

    try:
        PSF.write(filename2)
        with open(filename1) as f1, open(filename2) as f2:
            for i, j in zip_longest(f1, f2):
                assertion.eq(i, j)
    finally:
        if os.path.isfile(filename2):
            os.remove(filename2)

    with open(filename1, 'rb') as f1, TemporaryFile() as f2:
        PSF.write(f2, encoding='utf-8')
        f2.seek(0)
        for i, j in zip_longest(f1, f2):
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
    ref = np.load(join(PATH, 'generate_bonds.npy'))
    psf.generate_bonds(MOL)
    np.testing.assert_array_equal(ref, psf.bonds)


def test_generate_angles() -> None:
    """Tests for :meth:`PSFContainer.generate_angles`."""
    psf = PSF.copy()
    ref = np.load(join(PATH, 'generate_angles.npy'))
    psf.generate_angles(MOL)
    np.testing.assert_array_equal(ref, psf.angles)


def test_generate_dihedrals() -> None:
    """Tests for :meth:`PSFContainer.generate_dihedrals`."""
    psf = PSF.copy()
    ref = np.load(join(PATH, 'generate_dihedrals.npy'))
    psf.generate_dihedrals(MOL)
    np.testing.assert_array_equal(ref, psf.dihedrals)


def test_generate_impropers() -> None:
    """Tests for :meth:`PSFContainer.generate_impropers`."""
    psf = PSF.copy()
    ref = np.load(join(PATH, 'generate_impropers.npy'))
    psf.generate_impropers(MOL)
    np.testing.assert_array_equal(ref, psf.impropers)
