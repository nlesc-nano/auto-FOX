"""A set of functions for creating .psf files.

Examples
--------
Example code for generating a .psf file.
Ligand atoms within the ligand .xyz file and the qd .xyz file should be in the *exact* same order.
For example, implicit hydrogen atoms added by the
:func:`~scm.plams.interfaces.molecule.rdkit.from_smiles` functions are not guaranteed
to be ordered, even when using canonical SMILES strings.

.. code:: python

    >>> from scm.plams import Molecule, from_smiles
    >>> from FOX import PSFContainer
    >>> from FOX.recipes import generate_psf

    # Accepts .xyz, .pdb, .mol or .mol2 files
    >>> qd = Molecule(...)
    >>> ligand: Molecule = Molecule(...)
    >>> rtf_file : str = ...
    >>> psf_file : str = ...

    >>> psf: PSFContainer = generate_psf(qd_xyz, ligand_xyz, rtf_file=rtf_file)
    >>> psf.write(psf_file)


Examples
--------
If no ligand .xyz is on hand, or its atoms are in the wrong order, it is possible the
extract the ligand directly from the quantum dot.
This is demonstrated below with oleate (:math:`C_{18} H_{33} O_{2}^{-}`).

.. code:: python

    >>> from scm.plams import Molecule
    >>> from FOX import PSFContainer
    >>> from FOX.recipes import generate_psf, extract_ligand

    >>> qd = Molecule(...)  # Accepts an .xyz, .pdb, .mol or .mol2 file
    >>> rtf_file : str = ...

    >>> ligand_len = 18 + 33 + 2
    >>> ligand_atoms = {'C', 'H', 'O'}
    >>> ligand: Molecule = extract_ligand(qd, ligand_len, ligand_atoms)

    >>> psf: PSFContainer = generate_psf(qd, ligand, rtf_file=rtf_file)
    >>> psf.write(...)


Examples
--------
Example for multiple ligands.

.. code:: python

    >>> from typing import List
    >>> from scm.plams import Molecule
    >>> from FOX import PSFContainer
    >>> from FOX.recipes import generate_psf2

    >>> qd = Molecule(...)  # Accepts an .xyz, .pdb, .mol or .mol2 file
    >>> ligands = ('C[O-]', 'CC[O-]', 'CCC[O-]')
    >>> rtf_files = (..., ..., ...)

    >>> psf: PSFContainer = generate_psf2(qd, *ligands, rtf_file=rtf_files)
    >>> psf.write(...)

If the the psf construction with :func:`generate_psf2` failes to identify a particular ligand,
it is possible to return all (failed) potential ligands with the **ret_failed_lig** parameter.

.. code:: python

    >>> ...

    >>> ligands = ('CCCCCCCCC[O-]', 'CCCCBr')
    >>> failed_mol_list: List[Molecule] = generate_psf2(qd, *ligands, ret_failed_lig=True)


Index
-----
.. currentmodule:: FOX.recipes
.. autosummary::
    generate_psf
    generate_psf2
    extract_ligand

API
---
.. autofunction:: generate_psf
.. autofunction:: generate_psf2
.. autofunction:: extract_ligand

"""

from __future__ import annotations

import math
import warnings
from types import MappingProxyType
from typing import (Union, Iterable, Optional, Callable, Mapping, Type, Iterator, TypeVar,
                    Any, Tuple, List, cast, Sequence, Dict, TYPE_CHECKING)
from itertools import chain
from collections import abc

import numpy as np
from scm.plams import Molecule, Atom, Bond, MoleculeError, PT
from nanoutils import group_by_values, PathType, raise_if

from FOX import PSFContainer
from FOX.io.read_psf import overlay_rtf_file, overlay_str_file
from FOX.functions.molecule_utils import fix_bond_orders
from FOX.armc.sanitization import _assign_residues

try:
    from scm.plams import from_smiles, to_rdmol
except ImportError as ex:
    RDKIT_EX: None | ImportError = ex
else:
    from rdkit.Chem import Mol
    RDKIT_EX = None

    # A somewhat contrived way of loading :exc:`~Boost.Python.ArgumentError`
    _MOL = Molecule()
    _MOL.atoms = [Atom(symbol='H', coords=[0, 0, 0], mol=_MOL)]
    _MOL[1].properties.charge = -0.5
    try:
        to_rdmol(_MOL)
    except Exception as ex:
        ArgumentError: type[Exception] = type(ex)
    else:
        raise TypeError("Failed to extract Boost.Python.ArgumentError") from None
    del _MOL

KT = TypeVar("KT")
VT = TypeVar("VT")

__all__ = ['generate_psf', 'generate_psf2', 'extract_ligand']


def generate_psf(
    qd: str | Molecule,
    ligand: None | str | Molecule = None,
    rtf_file: None | PathType = None,
    str_file: None | PathType = None,
) -> PSFContainer:
    """Generate a :class:`PSFContainer` instance for **qd**.

    Parameters
    ----------
    qd : :class:`str` or :class:`Molecule`
        The ligand-pacifated quantum dot.
        Should be supplied as either a Molecule or .xyz file.

    ligand : :class:`str` or :class:`Molecule`, optional
        A single ligand.
        Should be supplied as either a Molecule or .xyz file.

    rtf_file : :class:`str`, optional
        The path+filename of the ligand's .rtf file.
        Used for assigning atom types.
        Alternativelly, one can supply a .str file with the **str_file** argument.

    str_file : :class:`str`, optional
        The path+filename of the ligand's .str file.
        Used for assigning atom types.
        Alternativelly, one can supply a .rtf file with the **rtf_file** argument.

    Returns
    -------
    :class:`PSFContainer`
        A PSFContainer instance with the new .psf file.

    """
    if not isinstance(qd, Molecule):
        qd = Molecule(qd)
    if not isinstance(ligand, Molecule) and ligand is not None:
        ligand = cast(Optional[Molecule], Molecule(ligand))

    if ligand is not None:
        qd_atnum = {at.atnum for at in qd}
        lig_atnum = {at.atnum for at in ligand}
        if not qd_atnum.issuperset(lig_atnum):
            atom_symbol = ", ".join(PT.get_symbol(i) for i in sorted(lig_atnum - qd_atnum))
            raise MoleculeError(f'No atoms {atom_symbol} found within {qd.get_formula()}')

        # Find the start of the ligand
        atnum = ligand[1].atnum
        for ligand_start, at in enumerate(qd):
            if at.atnum == atnum:
                break

        # Create an array with atomic-indice pairs defining bonds
        ligand.set_atoms_id()
        bonds = np.array([(b.atom1.id, b.atom2.id) for b in ligand.bonds])
        bonds += ligand_start
        ligand.unset_atoms_id()

        # Manually add bonds to the quantum dot
        ligand_len = len(ligand)
        qd.delete_all_bonds()
        while True:
            try:
                qd[bonds[0, 0]]
            except IndexError:
                break
            else:
                for j, k in bonds:
                    at1, at2 = qd[j], qd[k]
                    qd.add_bond(at1, at2)
                bonds += ligand_len

        # Create a nested list with residue indices
        res_ar = np.arange(ligand_start, len(qd))
        res_ar.shape = -1, ligand_len
        res_list: List[Sequence[int]] = res_ar.tolist()
        res_list.insert(0, range(ligand_start))
    else:
        if rtf_file is not None:
            raise TypeError("`rtf_file` cannot be specified if `ligand=None`")
        elif str_file is not None:
            raise TypeError("`str_file` cannot be specified if `ligand=None`")
        res_list = [range(len(qd))]
    _assign_residues(qd, res_list)

    # Create the .psf file
    psf = PSFContainer()
    psf.generate_bonds(qd)
    psf.generate_angles(qd)
    psf.generate_dihedrals(qd)
    psf.generate_impropers(qd)
    psf.generate_atoms(qd)
    if rtf_file is not None:
        overlay_rtf_file(psf, rtf_file)
    if str_file is not None:
        overlay_str_file(psf, str_file)

    # Set the charge to zero and return
    psf.charge = 0.0
    return psf


def extract_ligand(
    qd: str | Molecule,
    ligand_len: int,
    ligand_atoms: str | Iterable[str],
) -> Molecule:
    """Extract a single ligand from **qd**.

    Parameters
    ----------
    qd : :class:`str` or :class:`Molecule`
        The ligand-pacifated quantum dot.
        Should be supplied as either a Molecule or .xyz file.

    ligand_len : :class:`int`
        The number of atoms within a single ligand.

    ligand_atoms : :class:`str` or :class:`Iterable<collections.abc.Iterable>` [:class:`str`]
        One or multiple strings with the atomic symbols of all atoms within a single ligand.

    Returns
    -------
    :class:`Molecule`
        A single ligand Molecule.

    """
    if not isinstance(qd, Molecule):
        qd = Molecule(qd)
    if not isinstance(ligand_atoms, set):
        ligand_atoms = set(ligand_atoms) if not isinstance(ligand_atoms, str) else {ligand_atoms}

    # Identify where the core ends and the ligands start
    for i, at in enumerate(qd):
        if at.symbol in ligand_atoms:
            break
    else:
        raise MoleculeError(f'No atoms {tuple(ligand_atoms)} found within {qd.get_formula()}')

    # Construct a ligand
    j = i + ligand_len
    ligand = Molecule()
    ligand.atoms = [Atom(atnum=at.atnum, coords=at.coords, mol=ligand) for at in qd.atoms[i:j]]
    ligand.guess_bonds()

    return ligand


@raise_if(RDKIT_EX)
def generate_psf2(
    qd: str | Molecule,
    *ligands: str | Molecule | Mol,
    rtf_file: None | PathType | Iterable[PathType] = None,
    str_file: None | PathType | Iterable[PathType] = None,
    ret_failed_lig: bool = False,
) -> PSFContainer:
    r"""Generate a :class:`PSFContainer` instance for **qd** with multiple different **ligands**.

    Note
    ----
    Requires the optional RDKit package.

    Parameters
    ----------
    qd : :class:`str` or :class:`Molecule`
        The ligand-pacifated quantum dot.
        Should be supplied as either a Molecule or .xyz file.

    \*ligands : :class:`str`, :class:`Molecule` or :class:`Chem.Mol`
        One or more PLAMS/RDkit Molecules and/or SMILES strings representing ligands.

    rtf_file : :class:`str` or :class:`Iterable<collections.abc.Iterable>` [:class:`str`], optional
        The path+filename of the ligand's .rtf files.
        Filenames should be supplied in the same order as **ligands**.
        Used for assigning atom types.
        Alternativelly, one can supply a .str file with the **str_file** argument.

    str_file : :class:`str` or :class:`Iterable<collections.abc.Iterable>` [:class:`str`], optional
        The path+filename of the ligand's .str files.
        Filenames should be supplied in the same order as **ligands**.
        Used for assigning atom types.
        Alternativelly, one can supply a .rtf file with the **rtf_file** argument.

    ret_failed_lig : :class:`bool`
        If ``True``, return a list of all failed (potential) ligands
        if the function cannot identify any ligands within a certain range.
        Usefull for debugging.
        If ``False``, raise a :exc:`MoleculeError`.

    Returns
    -------
    :class:`Molecule`
        A single ligand Molecule.

    Raises
    ------
    :exc:`MoleculeError`
        Raised if the function fails to identify any ligands within a certain range.
        If ``ret_failed_lig = True``, return a list of failed (potential) ligands instead and
        issue a warning.

    """
    if not isinstance(qd, Molecule):
        qd = Molecule(qd)

    # Create a dictionary with RDKit molecules and the number of atoms contained therein
    rdmol_dict = _get_rddict(ligands)

    # Find the starting atom
    ligand_atoms = {at.GetAtomicNum() for rdmol in rdmol_dict for at in rdmol.GetAtoms()}
    for i, at in enumerate(qd):
        if at.atnum in ligand_atoms:
            break
    else:
        raise MoleculeError(f'No atoms {tuple(PT.get_symbol(i) for i in ligand_atoms)} found '
                            f'within {qd.get_formula()}')

    # Identify all bonds and residues
    res_list = [np.arange(i)]
    res_dict = {}
    while True:
        new, j = _get_initial_lig(qd, rdmol_dict, i)
        if new is None:
            break

        ref0, _ = next(iter(rdmol_dict.items()))
        for ref, k in rdmol_dict.items():
            k = 0 if ref is ref0 else k
            new = _update_lig(new, k, copy=False)
            j += k

            if _get_matches(new, ref):
                qd.bonds += [Bond(atom1=qd[bond.atom1.id],
                                  atom2=qd[bond.atom2.id],
                                  order=bond.order, mol=qd) for bond in new.bonds]
                res_list.append(np.arange(i, i+j))
                res_dict[len(res_list)] = id(ref)
                break
            else:
                continue

        else:
            err = (f'Failed to identify any ligands {ligands} within the range '
                   f'[{i}:{i + next(iter(rdmol_dict.items()))[1]}]')
            if not ret_failed_lig:
                raise MoleculeError(err)
            else:
                warnings.warn(err, category=MoleculeWarning)
                return _return_failed_ligs(qd, rdmol_dict, i)  # type: ignore
        i += j

    # Create the .psf file
    _assign_residues(qd, res_list)
    psf = PSFContainer()
    psf.generate_bonds(qd)
    psf.generate_angles(qd)
    psf.generate_dihedrals(qd)
    psf.generate_impropers(qd)
    psf.generate_atoms(qd, res_dict)

    if not (rtf_file is str_file is None):
        _id_dict = group_by_values(res_dict.items())
        id_range = (_id_dict[id(k)] for k in rdmol_dict.keys())
        _overlay(psf, 'rtf', id_range, rtf_file) if rtf_file is not None else None
        _overlay(psf, 'str', id_range, str_file) if str_file is not None else None

    # Set the charge to zero and return
    psf.charge = 0.0
    return psf


def _get_initial_lig(
    qd: Molecule,
    rdmol_dict: Mapping[Mol, int],
    i: int,
) -> Tuple[None | Molecule, int]:
    """Construct a new ligand at the begining of the :func:`generate_psf2` ``while`` loop."""
    _, j = next(iter(rdmol_dict.items()))
    new = Molecule()
    new.atoms = [Atom(atnum=at.atnum, coords=at.coords, mol=new) for at in qd.atoms[i:i+j]]

    if not new:
        return None, j
    elif len(new) != j:  # Pad with dummy atoms
        new.atoms += [Atom(atnum=0, coords=[0, 0, 0], mol=new) for _ in range(j - len(new))]

    new.set_atoms_id(start=i+1)
    return new, j


def _update_lig(ligand: Molecule, k: int, copy: bool = False) -> Molecule:
    """Update a ligand by removing the last **k** atoms."""
    ligand = ligand.copy() if copy else ligand
    atoms_del = ligand.atoms[k:] if k != 0 else []
    for at in atoms_del:
        ligand.delete_atom(at)

    ligand.guess_bonds()
    set_integer_bonds(ligand)
    fix_bond_orders(ligand)
    return ligand


def _return_failed_ligs(qd: Molecule, rdmol_dict: Mapping[Mol, int], i: int) -> List[Molecule]:
    """Return a list of failed ligands in case :func:`generate_psf2` fails to identify ligands."""
    new, j = _get_initial_lig(qd, rdmol_dict, i)
    if new is None:
        raise MoleculeError
    ret = []

    ref0, _ = next(iter(rdmol_dict.items()))
    for ref, k in rdmol_dict.items():
        k = 0 if ref is ref0 else k
        new = _update_lig(new, k, copy=True)
        ret.append(new)
        j += k
    return ret


class MoleculeWarning(RuntimeWarning):  # Molecule related warnings
    pass


#: Map a :class:`type` object to a callable for creating :class:`rdkit.Chem.Mol` instances.
if TYPE_CHECKING or RDKIT_EX is None:
    MolType = Union[Molecule, str, Mol]
    MOL_MAPPING: MappingProxyType[Type[MolType], Callable[[Any], Mol]] = MappingProxyType({
        str: lambda mol: to_rdmol(from_smiles(mol)),
        Molecule: to_rdmol,
        Mol: lambda mol: mol
    })
else:
    MOL_MAPPING = MappingProxyType({})


def _overlay(
    psf: PSFContainer,
    mode: str,
    id_ranges: Iterable[Iterable[int]],
    files: PathType | Iterable[PathType],
) -> None:
    """Overlay one or more .str or .rtf files."""
    if not isinstance(files, abc.Iterable) or isinstance(files, (str, bytes)):
        files_iter: Iterable[PathType] = (files,)
    else:
        files_iter = files

    if mode == 'rtf':
        func = overlay_rtf_file
    elif mode == 'str':
        func = overlay_str_file
    else:
        raise ValueError(f"'mode' expected either 'rtf' or 'str'; supplied value: {repr(mode)}")

    for file, id_range in zip(files_iter, id_ranges):
        func(psf, file, id_range=id_range)


def _items_sorted(dct: Mapping[KT, VT]) -> Iterator[Tuple[KT, VT]]:
    """Return a :meth:`dict.items()` iterator whose items are sorted by the dictionary values."""
    return iter(sorted(dct.items(), key=lambda kv: kv[1], reverse=True))


@raise_if(RDKIT_EX)
def _get_matches(mol: Molecule, ref: Mol) -> bool:
    """Check if the structures of **mol** and **ref** match."""
    try:
        rdmol = to_rdmol(mol)
    except ArgumentError:
        return False
    matches = rdmol.GetSubstructMatches(ref)
    match_set = set(chain.from_iterable(matches))
    return match_set == set(range(len(mol))) and len(match_set) == len(mol)


@raise_if(RDKIT_EX)
def _get_rddict(ligands: Iterable[str | Molecule | Mol]) -> Dict[Mol, int]:
    """Create an ordered dict with rdkit molecules and delta atom counts for :func:`generate_psf`."""  # noqa
    tmp_dct = {MOL_MAPPING[type(lig)](lig): 0 for lig in ligands}
    for rdmol in tmp_dct:
        tmp_dct[rdmol] = len(rdmol.GetAtoms())

    v_old = 0
    rdmol_dict = {}
    for k, v in _items_sorted(tmp_dct):
        rdmol_dict[k] = v - v_old
        v_old = v
    return rdmol_dict


def set_integer_bonds(self) -> None:
    """Convert non-integer bond orders into integers.

    For example, bond orders of aromatic systems are no longer set to the non-integer
    value of ``1.5``, instead adopting bond orders of ``1`` and ``2``.

    The implemented function walks a set of graphs constructed from all non-integer bonds,
    converting the orders of aforementioned bonds to integers by alternating calls to
    :func:`math.ceil` and :func:`math.floor`.
    The implication herein is that both :math:`i` and :math:`i+1` are considered valid
    (integer) values for any bond order within the :math:`(i, i+1)` interval.
    Floats which can be represented exactly as an integer, *e.g.* :math:`1.0`,
    are herein treated as integers.

    Can be used for sanitizaing any Molecules passed to the
    :mod:`rdkit<scm.plams.interfaces.molecule.rdkit>` module,
    as its functions are generally unable to handle Molecules with non-integer bond orders.

    ..code:: python

        >>> from scm.plams import Molecule

        >>> benzene = Molecule(...)
        >>> print(benzene)
            Atoms:
            1         C      1.193860     -0.689276      0.000000
            2         C      1.193860      0.689276      0.000000
            3         C      0.000000      1.378551      0.000000
            4         C     -1.193860      0.689276      0.000000
            5         C     -1.193860     -0.689276      0.000000
            6         C     -0.000000     -1.378551      0.000000
            7         H      2.132911     -1.231437     -0.000000
            8         H      2.132911      1.231437     -0.000000
            9         H      0.000000      2.462874     -0.000000
            10         H     -2.132911      1.231437     -0.000000
            11         H     -2.132911     -1.231437     -0.000000
            12         H     -0.000000     -2.462874     -0.000000
            Bonds:
            (3)--1.5--(4)
            (5)--1.5--(6)
            (1)--1.5--(6)
            (2)--1.5--(3)
            (4)--1.5--(5)
            (1)--1.5--(2)
            (3)--1.0--(9)
            (6)--1.0--(12)
            (5)--1.0--(11)
            (4)--1.0--(10)
            (2)--1.0--(8)
            (1)--1.0--(7)

        >>> benzene.set_integer_bonds()
        >>> print(benzene)
            Atoms:
            1         C      1.193860     -0.689276      0.000000
            2         C      1.193860      0.689276      0.000000
            3         C      0.000000      1.378551      0.000000
            4         C     -1.193860      0.689276      0.000000
            5         C     -1.193860     -0.689276      0.000000
            6         C     -0.000000     -1.378551      0.000000
            7         H      2.132911     -1.231437     -0.000000
            8         H      2.132911      1.231437     -0.000000
            9         H      0.000000      2.462874     -0.000000
            10         H     -2.132911      1.231437     -0.000000
            11         H     -2.132911     -1.231437     -0.000000
            12         H     -0.000000     -2.462874     -0.000000
            Bonds:
            (3)--1.0--(4)
            (5)--1.0--(6)
            (1)--2.0--(6)
            (2)--2.0--(3)
            (4)--2.0--(5)
            (1)--1.0--(2)
            (3)--1.0--(9)
            (6)--1.0--(12)
            (5)--1.0--(11)
            (4)--1.0--(10)
            (2)--1.0--(8)
            (1)--1.0--(7)

    """
    ceil = math.ceil
    floor = math.floor
    func_invert: Dict[Callable[[float], float], Callable[[float], float]] = {
        ceil: floor,
        floor: ceil,
    }

    def dfs(atom: Atom, func: Callable[[float], float]) -> None:
        """Depth-first search algorithm for integer-ifying the bond orders."""
        for b2 in atom.bonds:
            if b2._visited:
                continue

            b2._visited = True
            b2.order = func(b2.order)  # func = ``math.ceil()`` or ``math.floor()``
            del bond_dict[b2]

            atom_new = b2.atom1 if b2.atom1 is not atom else b2.atom2
            dfs(atom_new, func=func_invert[func])

    # Mark all non-integer bonds; floats which can be represented exactly
    # by an integer (e.g. 1.0 and 2.0) are herein treated as integers
    bond_dict: Dict[Bond, None] = {}  # An improvised OrderedSet (as it does not exist)
    for bond in self.bonds:
        if hasattr(bond.order, 'is_integer') and not bond.order.is_integer():
            bond._visited = False
            bond_dict[bond] = None
        else:
            bond._visited = True

    while bond_dict:
        b1, _ = bond_dict.popitem()
        order = b1.order

        # Start with either ``math.ceil()`` if the ceiling is closer than the floor;
        # start with ``math.floor()`` otherwise
        delta_ceil, delta_floor = ceil(order) - order, floor(order) - order
        func = ceil if abs(delta_ceil) < abs(delta_floor) else floor

        b1.order = func(order)
        b1._visited = True
        dfs(b1.atom1, func=func_invert[func])
        dfs(b1.atom2, func=func_invert[func])

    for bond in self.bonds:
        del bond._visited
