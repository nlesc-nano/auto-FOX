"""A module which expands on the Molecule class of PLAMS.

Index
-----
.. currentmodule:: FOX.functions.molecule_utils
.. autosummary::
    get_bonds
    get_angles
    get_dihedrals
    get_impropers
    residue_argsort

API
---
.. autofunction:: get_bonds
.. autofunction:: get_angles
.. autofunction:: get_dihedrals
.. autofunction:: get_impropers
.. autofunction:: residue_argsort

"""

from typing import List, overload, Callable, Tuple

import numpy as np

from scm.plams import Molecule, Atom, MoleculeError
from nanoutils import Literal

__all__ = ['get_bonds', 'get_angles', 'get_dihedrals', 'get_impropers', 'residue_argsort']


def fix_bond_orders(mol: Molecule) -> None:
    """Attempt to fix bond orders and (formal) atomic charges in this instance."""
    # Set default atomic charges
    for at in mol.atoms:
        if not at.properties.charge:
            at.properties.charge = 0

    # Fix atomic charges and bond orders
    for b1 in mol.bonds:
        at1, at2 = b1
        at1_saturation = sum([b2.order for b2 in at1.bonds])
        at1_saturation += -at1.properties.charge - at1.connectors
        at2_saturation = sum([b3.order for b3 in at2.bonds])
        at2_saturation += -at2.properties.charge - at2.connectors
        if at1_saturation == at2_saturation != 0:
            b1.order += np.abs(at1_saturation)
        else:
            if at1_saturation != 0:
                at1.properties.charge += at1_saturation
            if at2_saturation != 0:
                at2.properties.charge += at2_saturation


def separate_mod(mol: Molecule) -> List[List[int]]:
    """Modified version of the PLAMS Molecule.separate_ method.

    Separates the molecule into connected component as based on its bonds.
    Returns aforementioned components as a nested list of atomic indices.

    .. _Molecule.separate: https://www.scm.com/doc/plams/components/mol_api.html#scm.plams.mol.molecule.Molecule.separate

    Returns
    -------
    |list|_ [|list|_ [|int|_]]:
        A nested list of atomic indices, each sublist representing a set of unconnected
        moleculair fragments.

    """  # noqa
    # Mark atoms
    for i, at in enumerate(mol.atoms):
        at.id = i
        at._visited = False  # type: ignore[attr-defined]

    # Loop through atoms
    def dfs(at1: Atom, m_append: Callable[[int], None]):
        at1._visited = True  # type: ignore[attr-defined]
        m_append(at1.id)
        for bond in at1.bonds:
            at2 = bond.other_end(at1)
            if not at2._visited:  # type: ignore[attr-defined]
                dfs(at2, m_append)

    # Create a nested list of atomic indices
    indices: List[List[int]] = []
    indices_append = indices.append
    for at in mol.atoms:
        if not at._visited:  # type: ignore[attr-defined]
            m: List[int] = []
            dfs(at, m.append)
            indices_append(m)

    return indices


def get_bonds(mol: Molecule) -> np.ndarray:
    """Return an array with the atomic indices defining all bonds in **mol**.

    Parameters
    ----------
    mol : |plams.Molecule|_
        A PLAMS molecule.

    Returns
    -------
    :math:`n*2` |np.ndarray|_ [|np.int64|_]:
        A 2D array with atomic indices defining :math:`n` bonds.

    """
    mol.set_atoms_id()
    bonds = [(b.atom1.id, b.atom2.id) for b in mol.bonds]

    ret = np.array(bonds, dtype=int, ndmin=2)
    mol.unset_atoms_id()
    if not bonds:  # If no angles are found
        return ret

    # Sort horizontally
    mass = np.array([[mol[j].mass for j in i] for i in ret])
    idx1 = np.argsort(mass, axis=1)[:, ::-1]
    ret[:] = np.take_along_axis(ret, idx1, axis=1)

    # Sort and return vertically
    idx2 = np.argsort(ret, axis=0)[:, 0]
    return ret[idx2]


def get_angles(mol: Molecule) -> np.ndarray:
    """Return an array with the atomic indices defining all angles in **mol**.

    Parameters
    ----------
    mol : |plams.Molecule|_
        A PLAMS molecule.

    Returns
    -------
    :math:`n*3` |np.ndarray|_ [|np.int64|_]:
        A 2D array with atomic indices defining :math:`n` angles.

    """
    mol.set_atoms_id()
    angle: List[Tuple[int, int, int]] = []
    angle_append = angle.append

    for at2 in mol.atoms:
        if len(at2.bonds) < 2:
            continue

        at_other = [bond.other_end(at2) for bond in at2.bonds]
        for i, at1 in enumerate(at_other, 1):
            for at3 in at_other[i:]:
                angle_append((at1.id, at2.id, at3.id))

    ret = np.array(angle, dtype=int, ndmin=2)
    mol.unset_atoms_id()
    if not angle:  # If no angles are found
        return ret

    # Sort horizontally
    mass = np.array([[mol[j].mass for j in i] for i in ret[:, 0::2]])
    idx1 = np.argsort(mass, axis=1)[:, ::-1]
    ret[:, ::2] = np.take_along_axis(ret[:, ::2], idx1, axis=1)

    # Sort and return vertically
    idx2 = np.argsort(ret, axis=0)[:, 0]
    return ret[idx2]


def get_dihedrals(mol: Molecule) -> np.ndarray:
    """Return an array with the atomic indices defining all proper dihedral angles in **mol**.

    Parameters
    ----------
    mol : |plams.Molecule|_
        A PLAMS molecule.

    Returns
    -------
    :math:`n*4` |np.ndarray|_ [|np.int64|_]:
        A 2D array with atomic indices defining :math:`n` proper dihedrals.

    """
    mol.set_atoms_id()
    dihed: List[Tuple[int, int, int, int]] = []
    dihed_append = dihed.append

    for b1 in mol.bonds:
        if not (len(b1.atom1.bonds) > 1 and len(b1.atom2.bonds) > 1):
            continue

        at2, at3 = b1
        for b2 in at2.bonds:
            at1 = b2.other_end(at2)
            if at1 == at3:
                continue

            for b3 in at3.bonds:
                at4 = b3.other_end(at3)
                if at4 != at2:
                    dihed_append((at1.id, at2.id, at3.id, at4.id))

    ret = np.array(dihed, dtype=int, ndmin=2)
    mol.unset_atoms_id()
    if not dihed:  # If no dihedrals are found
        return ret

    # Sort horizontally
    mass = np.array([[mol[j].mass for j in i] for i in ret[:, 1:3]])
    idx1 = np.argsort(mass, axis=1)
    ret[:, ::3] = np.take_along_axis(ret[:, ::3], idx1, axis=1)
    ret[:, 1:3] = np.take_along_axis(ret[:, 1:3], idx1, axis=1)

    # Sort and return vertically
    idx2 = np.argsort(ret, axis=0)[:, 0]
    return ret[idx2]


def get_impropers(mol: Molecule) -> np.ndarray:
    """Return an array with the atomic indices defining all improper dihedral angles in **mol**.

    Parameters
    ----------
    mol : |plams.Molecule|_
        A PLAMS molecule.

    Returns
    -------
    :math:`n*4` |np.ndarray|_ [|np.int64|_]:
        A 2D array with atomic indices defining :math:`n` improper dihedrals.

    """
    mol.set_atoms_id()
    impropers: List[Tuple[int, int, int, int]] = []
    impropers_append = impropers.append

    for at1 in mol.atoms:
        order = [bond.order for bond in at1.bonds]
        if len(order) != 3:
            continue

        if 2.0 in order or 1.5 in order:
            at2, at3, at4 = [bond.other_end(at1) for bond in at1.bonds]
            impropers_append((at1.id, at2.id, at3.id, at4.id))

    ret = np.array(impropers, dtype=int, ndmin=2)
    mol.unset_atoms_id()
    if not impropers:  # If no impropers are found
        return ret

    # Sort along the rows of columns 2, 3 & 4 based on atomic mass in descending order
    mass = np.array([[mol[j].mass for j in i] for i in ret[:, 1:]])
    idx1 = np.argsort(mass, axis=1)
    idx1[:, 1:] = idx1[:, 1:][::-1]
    ret[:, 1:] = np.take_along_axis(ret[:, 1:], idx1, axis=1)

    # Sort vertically
    idx2 = np.argsort(ret, axis=0)[:, 0]
    return ret[idx2]


@overload
def residue_argsort(mol: Molecule, concatenate: Literal[True] = ...) -> np.ndarray: ...
@overload
def residue_argsort(mol: Molecule, concatenate: Literal[False] = ...) -> List[List[int]]: ...
def residue_argsort(mol, concatenate=True):  # noqa: E302
    """Return the indices that would sort this instance by residue number.

    Residues are defined based on moleculair fragments based on **self.bonds**.

    Parameters
    ----------
    mol : :class:`~scm.plams.mol.molecule.Molecule`
        A PLAMS Molecule with bonds.
    concatenate : bool
        If ``False``, returned a nested list with atomic indices.
        Each sublist contains the indices of a single residue.

    Returns
    -------
    :class:`numpy.ndarray` [:class:`int`], shape :math:`(n,)` or :class:`list` [:class:`list` [:class:`int`]]
        A 1D array of indices that would sort :math:`n` atoms this instance
        or a nested list of indices.

    """  # noqa: E501
    if not mol.bonds:
        raise MoleculeError("The passed Molecule has no bonds")

    # Define residues
    frags = separate_mod(mol)
    symbol = np.fromiter((at.symbol for at in mol), count=len(mol), dtype='U2')

    # Sort the residues
    core = []
    ligands = []
    for frag in frags:
        if len(frag) == 1:
            core += frag
        else:
            i = np.array(frag)
            argsort = np.argsort(symbol[i])
            ligands.append(i[argsort].tolist())
    core.sort()
    ligands.sort()

    ret = [core] + ligands
    if concatenate:
        return np.concatenate(ret)
    return ret
