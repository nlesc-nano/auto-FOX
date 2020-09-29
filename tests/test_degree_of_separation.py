"""A module for testing the :mod:`FOX.ff.degree_of_separation` module."""

import warnings
from pathlib import Path

import numpy as np
from scipy.sparse import (bsr_matrix, coo_matrix, csr_matrix, csc_matrix,
                          dia_matrix, dok_matrix, lil_matrix)

from scm.plams import Molecule, MoleculeError
from assertionlib import assertion

from FOX.ff.degree_of_separation import degree_of_separation, sparse_bond_matrix

PATH = Path('tests', 'test_files')
MOL = Molecule(PATH / 'Cd68Se55_26COO_MD_trajec.xyz')
MOL.guess_bonds([at for at in MOL if at.symbol in {'C', 'H', 'O'}])


def test_degree_of_separation():
    """Test :func:`degree_of_separation`."""
    ref1 = np.load(PATH / 'degree_of_separation.npy')
    ref2 = ref1.copy()
    ref2[ref1 > 1] = np.inf

    mat1 = degree_of_separation(MOL)
    mat2 = degree_of_separation(MOL, bond_mat=MOL.bond_matrix())
    mat3 = degree_of_separation(MOL, limit=1)

    np.testing.assert_array_equal(mat1, ref1)
    np.testing.assert_array_equal(mat2, ref1)
    np.testing.assert_array_equal(mat3, ref2)

    mol = MOL.copy()
    mol.delete_all_bonds()
    assertion.assert_(degree_of_separation, mol, exception=MoleculeError)


def test_sparse_bond_matrix():
    """Test :func:`sparse_bond_matrix`."""
    ref1 = np.load(PATH / 'sparse_bond_matrix.npy')
    ref2 = ref1.astype(int)
    ref3 = ref1.astype(bool)

    mat1 = sparse_bond_matrix(MOL, dtype=float)
    mat2 = sparse_bond_matrix(MOL, dtype=int)
    mat3 = sparse_bond_matrix(MOL)

    np.testing.assert_array_equal(mat1.toarray(), ref1)
    np.testing.assert_array_equal(mat2.toarray(), ref2)
    np.testing.assert_array_equal(mat3.toarray(), ref3)

    sparse_type_tup = (
        bsr_matrix, coo_matrix, csr_matrix, csc_matrix, dia_matrix, dok_matrix, lil_matrix
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=np.VisibleDeprecationWarning())

        for sparse_type in sparse_type_tup:
            mat_n = sparse_bond_matrix(MOL, sparse_type=sparse_type)
            np.testing.assert_array_equal(mat_n.toarray(), ref3)

    assertion.assert_(sparse_bond_matrix, MOL, dtype='bob', exception=TypeError)
    assertion.assert_(sparse_bond_matrix, MOL, sparse_type='bob', exception=TypeError)
    assertion.assert_(sparse_bond_matrix, MOL, sparse_type=int, exception=TypeError)
