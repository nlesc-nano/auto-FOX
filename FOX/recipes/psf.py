"""
FOX.recipes.psf
=================

A set of functions for creating .psf files.

Examples
--------
Example code for generating a .psf file.
Ligand atoms within the ligand .xyz file and the qd .xyz file should be in the *exact* same order.
For example, implicit hydrogen atoms added by the
:func:`from_smiles<scm.plams.interfaces.molecule.rdkit.from_smiles>` functions are not guaranteed
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

    >>> from scm.plams import Molecule
    >>> from FOX import PSFContainer
    >>> from FOX.recipes import generate_psf2

    >>> qd = Molecule(...)  # Accepts an .xyz, .pdb, .mol or .mol2 file
    >>> ligands = ('C[O-]', 'CC[O-]', 'CCC[O-]')
    >>> rtf_files = (..., ..., ...)

    >>> psf: PSFContainer = generate_psf2(qd, *ligands, rtf_file=rtf_files)
    >>> psf.write(...)


Index
-----
.. currentmodule:: FOX.recipes.psf
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
import math
import heapq
from os import PathLike
from sys import version_info
from types import MappingProxyType
from typing import (Union, Iterable, Optional, TypeVar, Callable, Mapping, Type, Iterator,
                    Hashable, Any, Tuple, MutableMapping, AnyStr)
from collections import abc
from itertools import chain

import numpy as np
from scm.plams import Molecule, Atom, Bond, MoleculeError, PT

from FOX import PSFContainer, assert_error
from FOX.io.read_psf import overlay_rtf_file, overlay_str_file
from FOX.functions.utils import group_by_values
from FOX.functions.molecule_utils import fix_bond_orders
from FOX.armc_functions.sanitization import _assign_residues

if version_info.minor < 7:
    from collections import OrderedDict
else:  # Dictionaries are ordered starting from python 3.7
    OrderedDict = dict

try:
    from rdkit import Chem
    from scm.plams import from_smiles, to_rdmol

    Mol: Union[str, type] = Chem.Mol
    RDKIT_ERROR = None

    # A somewhat contrived way of loading :exc:`ArgumentError<Boost.Python.ArgumentError>`
    _MOL = Molecule()
    _MOL.atoms = [Atom(symbol='H', coords=[0, 0, 0], mol=_MOL)]
    _MOL[1].properties.charge = -0.5
    try:
        to_rdmol(_MOL)
    except Exception as ex:
        ArgumentError: Optional[Type[Exception]] = type(ex)
    del _MOL

except ImportError:
    Mol: Union[str, type] = 'rdkit.Chem.rdchem.Mol'
    ArgumentError: Optional[Type[Exception]] = None
    RDKIT_ERROR = ("Use of the FOX.{} function requires the 'rdkit' package."
                   "\n'rdkit' can be installed via conda with the following command:"
                   "\n\tconda install -n FOX -c conda-forge rdkit")


__all__ = ['generate_psf', 'generate_psf2', 'extract_ligand']


def generate_psf(qd: Union[str, Molecule], ligand: Union[str, Molecule],
                 rtf_file: Optional[str] = None, str_file: Optional[str] = None) -> PSFContainer:
    """Generate a :class:`PSFContainer` instance for **qd**.

    Parameters
    ----------
    qd : :class:`str` or :class:`Molecule`
        The ligand-pacifated quantum dot.
        Should be supplied as either a Molecule or .xyz file.

    ligand : :class:`str` or :class:`Molecule`
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
    if not isinstance(ligand, Molecule):
        ligand = Molecule(ligand)

    # Find the start of the ligand
    atnum = ligand[1].atnum
    for ligand_start, at in enumerate(qd):
        if at.atnum == atnum:
            break
    else:
        raise MoleculeError(f'No atom {ligand[1].symbol} found within {qd.get_formula()}')

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
    res_list = res_ar.tolist()
    res_list.insert(0, np.arange(ligand_start))
    _assign_residues(qd, res_list)

    # Create the .psf file
    psf = PSFContainer()
    psf.generate_bonds(qd)
    psf.generate_angles(qd)
    psf.generate_dihedrals(qd)
    psf.generate_impropers(qd)
    psf.generate_atoms(qd)
    overlay_rtf_file(psf, rtf_file) if rtf_file is not None else None
    overlay_str_file(psf, str_file) if str_file is not None else None

    # Set the charge to zero and return
    psf.charge = 0.0
    return psf


def extract_ligand(qd: Union[str, Molecule], ligand_len: int,
                   ligand_atoms: Union[str, Iterable[str]]) -> Molecule:
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


@assert_error(RDKIT_ERROR)
def generate_psf2(qd: Union[str, Molecule],
                  *ligands: Union[str, Molecule, Mol],
                  rtf_file: Union[None, str, Iterable[str]] = None,
                  str_file: Union[None, str, Iterable[str]] = None) -> PSFContainer:
    r"""Generate a :class:`PSFContainer` instance for **qd** with multiple different **ligands**.

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

    Returns
    -------
    :class:`Molecule`
        A single ligand Molecule.

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
        raise MoleculeError(f'No atoms {tuple(PT[i] for i in ligand_atoms)} found '
                            f'within {qd.get_formula()}')

    # Identify all bonds and residues
    res_list = [np.arange(i)]
    res_dict = {}
    while True:
        ref_j, j = next(iter(rdmol_dict.items()))
        new = Molecule()
        new.atoms = [Atom(atnum=at.atnum, coords=at.coords, mol=new) for at in qd.atoms[i:i+j]]
        if not new:
            break
        elif len(new) != j:  # Pad with dummy atoms
            new.atoms += [Atom(atnum=0, coords=[0, 0, 0], mol=new) for _ in range(j - len(new))]
        new.set_atoms_id(start=i+1)

        for ref, k in rdmol_dict.items():
            k = 0 if ref is ref_j else k
            atoms_del = new.atoms[k:] if k != 0 else []
            for at in atoms_del:
                new.delete_atom(at)
            guess_bonds(new)
            fix_bond_orders(new)

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
            raise MoleculeError(f'Failed to identify any ligands {ligands} within the range '
                                f'[{i}:{i + next(iter(rdmol_dict.items()))[1]}]')
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


MolType = TypeVar('MolType', Molecule, str, Mol)

try:
    #: Map a :class:`type` object to a callable for creating :class:`rdkit.Chem.Mol` instances.
    MOL_MAPPING: Mapping[Type[MolType], Callable[[MolType], Mol]] = MappingProxyType({
        str: lambda mol: to_rdmol(from_smiles(mol)),
        Molecule: to_rdmol,
        Mol: lambda mol: mol
    })
except NameError:
    MOL_MAPPING = None  # rdkit is not installed


def _overlay(psf: PSFContainer, mode: str, id_ranges: Iterable[Iterable[int]],
             files: Union[AnyStr, PathLike, Iterable[AnyStr], Iterable[PathLike]]) -> None:
    """Overlay one or more .str or .rtf files."""
    if not isinstance(files, abc.Iterable) or isinstance(files, (str, bytes)):
        files_iter = (files,)
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


def _items_sorted(dct: Mapping) -> Iterator[Tuple[Hashable, Any]]:
    """Return a :meth:`dict.items()` iterator whose items are sorted by the dictionary values."""
    return iter(sorted(dct.items(), key=lambda kv: kv[1], reverse=True))


@assert_error(RDKIT_ERROR)
def _get_matches(mol: Molecule, ref: Mol) -> bool:
    """Check if the structures of **mol** and **ref** match."""
    try:
        rdmol = to_rdmol(mol)
    except ArgumentError:
        return False
    matches = rdmol.GetSubstructMatches(ref)
    match_set = set(chain.from_iterable(matches))
    return match_set == set(range(len(mol))) and len(match_set) == len(mol)


@assert_error(RDKIT_ERROR)
def _get_rddict(ligands: Iterable[Union[str, Molecule, Mol]]) -> MutableMapping[Mol, int]:
    """Create an ordered dict with rdkit molecules and delta atom counts for :func:`generate_psf`."""  # noqa
    tmp_dct = {MOL_MAPPING[type(lig)](lig): 0 for lig in ligands}
    for rdmol in tmp_dct:
        tmp_dct[rdmol] = len(rdmol.GetAtoms())

    v_old = 0
    rdmol_dict = OrderedDict()
    for k, v in _items_sorted(tmp_dct):
        rdmol_dict[k] = v - v_old
        v_old = v
    return rdmol_dict


def guess_bonds(mol: Molecule) -> None:
    """Modified version of :meth:`Molecule.guess_bonds`.

    Bond orders for "aromatic" systems are no longer set to `1.5`, thus remaining integer.

    """
    class HeapElement:
        def __init__(self, order, ratio, atom1, atom2):
            eff_ord = order
            if order == 1.5:  # effective order for aromatic bonds
                eff_ord = 1.15
            elif order == 1 and {atom1.symbol, atom2.symbol} == {'C', 'N'}:
                eff_ord = 1.11  # effective order for single C-N bond
            value = (eff_ord + 0.9) * ratio
            self.data = (value, order, ratio)
            self.atoms = (atom1, atom2)

        def unpack(self):
            val, o, r = self.data
            at1, at2 = self.atoms
            return val, o, r, at1, at2

        def __lt__(self, other): return self.data < other.data
        def __le__(self, other): return self.data <= other.data
        def __eq__(self, other): return self.data == other.data
        def __ne__(self, other): return self.data != other.data
        def __gt__(self, other): return self.data > other.data
        def __ge__(self, other): return self.data >= other.data

    mol.delete_all_bonds()

    dmax = 1.28

    atom_list = mol
    cubesize = dmax*2.1*max([at.radius for at in atom_list])

    cubes = {}
    for i, at in enumerate(atom_list, 1):
        at._id = i
        at.free = at.connectors
        at.cube = tuple(map(lambda x: int(math.floor(x/cubesize)), at.coords))
        if at.cube in cubes:
            cubes[at.cube].append(at)
        else:
            cubes[at.cube] = [at]

    neighbors = {}
    for cube in cubes:
        neighbors[cube] = []
        for i in range(cube[0]-1, cube[0]+2):
            for j in range(cube[1]-1, cube[1]+2):
                for k in range(cube[2]-1, cube[2]+2):
                    if (i, j, k) in cubes:
                        neighbors[cube] += cubes[(i, j, k)]

    heap = []
    for at1 in atom_list:
        if at1.free > 0:
            for at2 in neighbors[at1.cube]:
                if (at2.free > 0) and (at1._id < at2._id):
                    ratio = at1.distance_to(at2) / (at1.radius + at2.radius)
                    if (ratio < dmax):
                        heap.append(HeapElement(0, ratio, at1, at2))
                        if (at1.atnum == 16 and at2.atnum == 8):
                            at1.free = 6
                        elif (at2.atnum == 16 and at1.atnum == 8):
                            at2.free = 6
                        elif (at1.atnum == 7):
                            at1.free += 1
                        elif (at2.atnum == 7):
                            at2.free += 1
    heapq.heapify(heap)

    for at in atom_list:
        if at.atnum == 7:
            if at.free > 6:
                at.free = 4
            else:
                at.free = 3

    step = 1
    while heap:
        val, o, r, at1, at2 = heapq.heappop(heap).unpack()
        if at1.free >= step and at2.free >= step:
            o += step
            at1.free -= step
            at2.free -= step
            if o < 3:
                heapq.heappush(heap, HeapElement(o, r, at1, at2))
            else:
                mol.add_bond(at1, at2, o)
        elif o > 0:
            if o == 1.5:
                o = Bond.AR
            mol.add_bond(at1, at2, o)

    for at in atom_list:
        del at.cube, at.free, at._id
