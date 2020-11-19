"""A set of functions for calculating the degree of separation of atoms within a molecule.

Index
-----
.. currentmodule:: FOX.ff.degree_of_separation
.. autosummary::
    degree_of_separation
    sparse_bond_matrix

API
---
.. autofunction:: degree_of_separation
.. autofunction:: sparse_bond_matrix

"""

from typing import Type, Union, Optional, Any, TypeVar, Tuple, Generator, overload
from itertools import chain

import numpy as np
from scipy.sparse import csr_matrix, spmatrix
from scipy.sparse.csgraph import dijkstra

from scm.plams import Molecule, MoleculeError

__all__ = ['degree_of_separation', 'sparse_bond_matrix']


def degree_of_separation(mol: Molecule, limit: float = np.inf,
                         bond_mat: Optional[np.ndarray] = None) -> np.ndarray:
    r"""Construct a matrix with the degree of separation of all atom-pairs in **mol**.

    Each element :math:`D_{i, j}^{\text{sep}}` in the to-be returned matrix
    :math:`\boldsymbol{D}^{\text{sep}} \in \mathbb{R}_{+}^{n,n}`
    represents the (minimum) number of bonds
    seperating atoms :math:`i` and :math:`j`.

    The degree of seperation is set to ``inf`` if two atoms are disjointed,
    *i.e.* there is no combination of bonds connecting aformentioned atom-pair.

    Parameters
    ----------
    mol : :class:`Molecule` or :class:`MultiMolecule`
        A PLAMS Molecule with bonds.

    limit : :class:`float`
        The maximum degree of separation to calculate, must be >= 0.
        Using a smaller limit will decrease computation time by aborting calculations between
        pairs that are separated by a degree of separation > limit.
        For such pairs, the degree of separation will be equal to ``inf`` (*i.e.* not connected).

    bond_mat : array_like, optional
        An optional bond matrix or other object compatible with SciPy's
        :class:`csr_matrix<scipy.sparse.csr_matrix>`.
        If ``None``, calculate the sparse bond matrix with :func:`get_sparse_bond_matrix`.

    Returns
    -------
    (:math:`n`, :math:`n`) :class:`numpy.ndarray` [:class:`float`]
        A symmetric 2D array containing the degrees of separation between all atom-pairs in **mol**.
        Values are set to ``inf`` if the responsible atoms are disjointed.

    Raises
    ------
    :exc:`MoleculeError<scm.plams.core.errors.MoleculeError>`
        Raised if the passed Molecule has no bonds and **bond_mat** is ``None``.

    """
    len_mol = len(mol)

    if bond_mat is None:
        if not mol.bonds:
            raise MoleculeError("The passed Molecule has no bonds")
        sparse_bond_mat = sparse_bond_matrix(mol, dtype=bool)
    else:
        sparse_bond_mat = csr_matrix(bond_mat, dtype=bool, copy=False, shape=(len_mol, len_mol))

    return dijkstra(sparse_bond_mat,
                    directed=False,
                    limit=limit,
                    indices=np.arange(len_mol),
                    return_predecessors=False)


S = TypeVar('S', bound=spmatrix)


@overload
def sparse_bond_matrix(
    mol: Molecule,
    dtype: Union[None, type, str, np.dtype] = ...,
    sparse_type: Type[csr_matrix] = ...,
    **kwargs: Any,
) -> csr_matrix:
    ...
@overload  # noqa: E302
def sparse_bond_matrix(  # type: ignore[misc]
    mol: Molecule,
    dtype: Union[None, type, str, np.dtype] = ...,
    sparse_type: Type[S] = ...,
    **kwargs: Any,
) -> S:
    ...
def sparse_bond_matrix(mol, dtype=bool, sparse_type=csr_matrix, **kwargs):  # noqa: E302
    r"""Create a sparse bond-matrix of a user-specified data type.

    Parameters
    ----------
    mol : :class:`Molecule`
        A PLAMS Molecule with bonds.

    dtype : :class:`type`, :class:`str` or :class:`dtype<numpy.dtype>`, optional
        The datatype of the to-be returned sparse matrix.

    sparse_type : :class:`type` [:class:`spmatrix<scipy.sparse.spmatrix>`]
        The to-be returned type of sparse matrix.
        Uses a compressed sparse row matrix by default
        (:class:`csr_matrix<scipy.sparse.csr_matrix>`).

    \**kwargs : :data:`Any<typing.Any>`
        Further keyword arguments for the to-be initialized sparse matrix.

    Returns
    -------
    :class:`csr_matrix<scipy.sparse.csr_matrix>`
        A sparse bond matrix.

    """
    dtype_ = np.dtype(dtype)

    # Prepare parameters
    bond_count = 2 * len(mol.bonds)  # Multiply by 2 in order to consider all permutations
    count = 2 * bond_count
    shape = bond_count, 2

    # Construct an array of row and column indices
    iterator1 = chain.from_iterable(_bond_id_generator(mol))
    bond_idx = np.fromiter(iterator1, dtype=int, count=count)
    bond_idx.shape = shape

    # Construct the to-be assigned data
    if dtype_ == bool:
        data = np.ones(bond_count, dtype=dtype_)
    else:
        iterator2 = chain.from_iterable((b.order, b.order) for b in mol.bonds)
        data = np.fromiter(iterator2, count=bond_count, dtype=dtype_)

    # Create and return the sparse matrix
    ret_shape = len(mol), len(mol)
    return _sparse_mat(sparse_type, ret_shape, data, *bond_idx.T, **kwargs)


def _bond_id_generator(mol: Molecule) -> Generator[Tuple[int, int, int, int], None, None]:
    """Yield all permutations of the bond-defining atoms indices in **mol**."""
    mol.set_atoms_id(start=0)
    for b in mol.bonds:
        id1 = b.atom1.id
        id2 = b.atom2.id
        yield id1, id2, id2, id1

    mol.unset_atoms_id()


def _sparse_mat(sparse_type: Type[S], shape: Tuple[int, int], data: np.ndarray,
                rows: np.ndarray, columns: np.ndarray, **kwargs: Any) -> S:
    """Create and return the sparse matrix for :func:`sparse_bond_matrix`."""
    try:
        return sparse_type((data, (rows, columns)), shape=shape, **kwargs)
    except Exception as ex:
        if not issubclass(sparse_type, spmatrix):  # Validate the passed type
            raise TypeError("'sparse_type' expect a subclass of 'spmatrix'; observed type: "
                            f"'{sparse_type.__class__.__name__}'") from ex

        try:  # Try harder
            return sparse_type(csr_matrix((data, (rows, columns)), shape=shape), **kwargs)
        except Exception:
            raise ex  # Give up
