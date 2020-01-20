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

Index
-----
.. currentmodule:: FOX.recipes.psf
.. autosummary::
    generate_psf
    extract_ligand

API
---
.. autofunction:: generate_psf
.. autofunction:: extract_ligand

"""

from typing import Union, Iterable, Optional

import numpy as np
from scm.plams import Molecule, Atom, MoleculeError

from FOX import PSFContainer
from FOX.io.read_psf import overlay_rtf_file, overlay_str_file
from FOX.armc_functions.sanitization import _assign_residues


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
    """Extract a single lignad from **qd**.

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
