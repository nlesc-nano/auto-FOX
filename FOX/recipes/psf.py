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

    >>> psf: PSFContainer = generate_psf2(qd, *ligands, rtf_files=rtf_files)
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
from os import PathLike
from sys import version_info
from types import MappingProxyType
from typing import (Union, Iterable, Optional, TypeVar, Callable, Mapping, Type, Iterator,
                    Hashable, Any, Tuple, MutableMapping, AnyStr)
from collections import abc
from itertools import chain

import numpy as np
from scm.plams import Molecule, Atom, Bond, MoleculeError, PT

from FOX import PSFContainer
from FOX.io.read_psf import overlay_rtf_file, overlay_str_file
from FOX.functions.molecule_utils import fix_bond_orders
from FOX.armc_functions.sanitization import _assign_residues

if version_info.minor < 7:
    from collections import OrderedDict
else:  # Dictionaries are ordered starting from python 3.7
    OrderedDict = dict

try:
    from rdkit import Chem
    from scm.plams import from_smiles, to_rdmol
    RDKIT: Optional[ImportError] = None
except ImportError as ex:
    RDKIT: Optional[ImportError] = ex


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


def generate_psf2(qd: Union[str, Molecule],
                  *ligands: Union[str, Molecule, Chem.Mol],
                  rtf_files: Union[None, str, Iterable[str]] = None,
                  str_files: Union[None, str, Iterable[str]] = None) -> PSFContainer:
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
    if RDKIT is not None:
        raise RDKIT
    if not isinstance(qd, Molecule):
        qd = Molecule(qd)
    if rtf_files is not None and len(rtf_files) != len(ligands):
        raise ValueError(f"The number of files in 'rtf_file' (len(rtf_files)) should be "
                         f"the same as '*ligands' (len(ligands))")
    if str_files is not None and len(str_files) != len(ligands):
        raise ValueError(f"The number of files in 'str_file' (len(str_files)) should be "
                         f"the same as '*ligands' (len(ligands))")

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
        else:
            new.set_atoms_id(start=i+1)

        for ref, k in rdmol_dict.items():
            k = 0 if ref is ref_j else k
            atoms_del = new.atoms[-k:] if k != 0 else []
            for at in atoms_del:
                new.delete_atom(at)
            new.guess_bonds()
            fix_bond_orders(new)

            j -= k
            if _get_matches(new, ref):
                bonds = new.bonds
                qd.bonds += [Bond(atom1=qd[at1.id], atom2=qd[at1.id], mol=qd) for at1, at2 in bonds]
                res_list.append(np.arange(i, i+j))
                res_dict[len(res_list)] = id(ref)
                break
            else:
                continue
        else:
            raise MoleculeError('Failed to identify any ligands within the range '
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
    _overlay(psf, rtf_files) if rtf_files is not None else None
    _overlay(psf, str_files) if str_files is not None else None

    # Set the charge to zero and return
    psf.charge = 0.0
    return psf


Mol = TypeVar('Mol', Molecule, str, Chem.Mol)

#: Map a :class:`type` object to a callable for creating :class:`rdkit.Chem.Mol` instances.
MOL_MAPPING: Mapping[Type[Mol], Callable[[Mol], Chem.Mol]] = MappingProxyType({
    str: lambda mol: to_rdmol(from_smiles(mol)),
    Molecule: to_rdmol,
    Chem.Mol: lambda mol: mol
})


def _overlay(psf: PSFContainer,
             files: Union[AnyStr, PathLike, Iterable[AnyStr], Iterable[PathLike]]) -> None:
    if not isinstance(files, abc.Iterable) or isinstance(files, (str, bytes)):
        files_iter = (files,)
    else:
        files_iter = files

    for file in files_iter:
        overlay_rtf_file(psf, file)


def _items_sorted(dct: Mapping) -> Iterator[Tuple[Hashable, Any]]:
    """Return a :meth:`dict.items()` iterator whose items are sorted by the dictionary values."""
    return iter(sorted(dct.items(), key=lambda kv: kv[1], reverse=True))


def _get_matches(mol: Molecule, ref: Chem.Mol) -> bool:
    """Check if the structures of **mol** and **ref** match."""
    rdmol = to_rdmol(mol)
    matches = rdmol.GetSubstructMatches(ref)
    match_set = set(chain.from_iterable(matches))
    return match_set == set(range(len(mol))) and len(match_set) == len(mol)


def _get_rddict(ligands: Iterable[Union[str, Molecule, Chem.Mol]]) -> MutableMapping[Chem.Mol, int]:
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
