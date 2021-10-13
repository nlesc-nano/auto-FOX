"""Tests for :mod:`FOX.recipes.psf`."""

import warnings
from pathlib import Path
from typing import Optional, Mapping, Any, Type

import pytest
import numpy as np
from scm.plams import Molecule, MoleculeError, Atom
from assertionlib import assertion

from FOX import PSFContainer
from FOX.recipes import generate_psf, generate_psf2, extract_ligand
from FOX.recipes.psf import RDKIT_EX

PATH = Path('tests') / 'test_files'
STR_FILE = PATH / 'ligand.str'

QD: Molecule = Molecule(PATH / 'Cd68Se55_26COO_MD_trajec.xyz')
LIGAND = extract_ligand(QD, 4, {'C', 'H', 'O'})

LIGAND2 = LIGAND.copy()
LIGAND2.add_atom(Atom(symbol="Na"))


def test_extract_ligand() -> None:
    """Tests for :func:`extract_ligand`."""
    qd = QD.copy()
    ligand = extract_ligand(qd, 4, {'C', 'H', 'O'})
    ref = np.array([[5.47492504, -3.54070096, -3.91403079],
                    [6.25736402, -3.70922481, -4.71302813],
                    [4.31834322, -3.0473366, -4.20026909],
                    [5.77141553, -3.85039436, -2.74500143]])
    np.testing.assert_allclose(ligand, ref)

    ligand = extract_ligand(qd, 2, 'O')
    ref = np.array([[4.31834322, -3.0473366, -4.20026909],
                    [5.77141553, -3.85039436, -2.74500143]])
    np.testing.assert_allclose(ligand, ref)

    assertion.assert_(extract_ligand, qd, 4, 'bob', exception=MoleculeError)


class TestGeneratePSF:
    """Tests for :func:`extract_ligand`."""

    @pytest.mark.parametrize("lig", ["formate", None])
    def test_passes(self, lig: Optional[str]) -> None:
        qd = QD.copy()
        ref_name = "generate_psf.psf" if lig is None else f"generate_psf_{lig}.psf"
        ref = PSFContainer.read(PATH / ref_name)

        if lig is None:
            psf = generate_psf(qd)
        else:
            psf = generate_psf(qd, LIGAND, str_file=STR_FILE)
        assertion.eq(psf, ref)

    @pytest.mark.parametrize(
        "kwargs,exc",
        [
            ({"qd": QD, "rtf_file": "test"}, TypeError),
            ({"qd": QD, "str_file": "test"}, TypeError),
            ({"qd": QD, "ligand": LIGAND2}, MoleculeError),
        ],
    )
    def test_raises(self, kwargs: Mapping[str, Any], exc: Type[Exception]) -> None:
        with pytest.raises(exc):
            generate_psf(**kwargs)


@pytest.mark.skipif(RDKIT_EX is not None, reason="Requires RDKit")
def test_generate_psf2() -> None:
    """Tests for :func:`extract_ligand`."""
    qd = QD.copy()
    psf = generate_psf2(qd, 'C(=O)[O-]')
    ref = np.load(PATH / 'generate_psf_bonds2.npy')

    np.testing.assert_array_equal(psf.bonds, ref)

    qd = QD.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mol_list = generate_psf2(qd, 'CC(=O)[O-]', ret_failed_lig=True)

    assertion.len_eq(mol_list, 1)
    assertion.eq(mol_list[0].get_formula(), 'C2H2O3')
