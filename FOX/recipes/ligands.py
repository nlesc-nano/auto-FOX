"""
FOX.recipes.ligands
===================

A set of functions for analyzing ligands.

Examples
--------
An example for generating a ligand center of mass RDF.

.. code:: python

    >>> import numpy as np
    >>> import pandas as pd
    >>> from FOX import MultiMolecule, example_xyz
    >>> from FOX.recipes import get_lig_center

    >>> mol = MultiMolecule.from_xyz(example_xyz)
    >>> start = 123  # Start of the ligands
    >>> step = 4  # Size of the ligands

    >>> lig_centra: np.ndarray = get_lig_center(mol, start, step)
    >>> mol_new: MultiMolecule = mol.add_atoms(lig_centra, symbols='Xx')
    >>> rdf: pd.DataFrame = mol_new.init_rdf(atom_subset=['Xx'])

.. image:: ligand_rdf.png
    :scale: 20 %
    :align: center

|
Or the ADF.

.. code:: python

    >>> ...

    >>> adf: pd.DataFrame = mol_new.init_rdf(atom_subset=['Xx'], r_max=np.inf)

.. image:: ligand_adf.png
    :scale: 20 %
    :align: center

|
Focus on a specific ligand subset is possible by slicing the new ligand Cartesian coordinate array.

.. code:: python

    >>> ...

    >>> keep_lig = [0, 1, 2, 3]  # Keep these ligands; disgard the rest
    >>> lig_centra_subset = lig_centra[:, keep_lig]

    >>> mol_new2: MultiMolecule = mol.add_atoms(lig_centra_subset, symbols='Xx')
    >>> rdf: pd.DataFrame = mol_new2.init_rdf(atom_subset=['Xx'])

.. image:: ligand_rdf_subset.png
    :scale: 20 %
    :align: center


Index
-----
.. currentmodule:: FOX.recipes.ligands
.. autosummary::
    get_lig_center

API
---
.. autofunction:: get_lig_center

"""

from typing import Optional

import numpy as np

from FOX import MultiMolecule

__all__ = ['get_lig_center']


def get_lig_center(mol: MultiMolecule, start: int, step: int, stop: Optional[int] = None,
                   mass_weighted: bool = True) -> np.ndarray:
    """Return an array with the (mass-weighted) mean position of each ligands in **mol**.

    Parameters
    ----------
    mol : :class:`MultiMolecule<FOX.classes.multi_mol.MultiMolecule>`
        A MultiMolecule instance.

    start : :class:`int`
        The atomic index of the first ligand atoms.

    step : :class:`int`
        The number of atoms per ligand.

    stop : :class:`int`, optional
        Can be used for neglecting any ligands beyond a user-specified atomic index.

    mass_weighted : :class:`bool`
        If ``True``, return the mass-weighted mean ligand position rather than
        its unweighted counterpart.

    Returns
    -------
    :class:`numpy.ndarray`
        A new array with the ligand's centra of mass.
        If ``mol.shape == (m, n, 3)`` then, given ``k`` new ligands, the to-be returned array's
        shape is ``(m, k, 3)``.

    """
    # Extract the ligands
    ligands = mol[:, start:stop].copy()
    m, n, _ = ligands.shape
    ligands.shape = m, n // step, step, 3

    if not mass_weighted:
        return np.asarray(ligands.mean(axis=2))

    mass = ligands.mass[start:stop]
    mass.shape = 1, n // step, step, 1
    ligands *= mass
    return np.asarray(ligands.sum(axis=2) / mass.sum(axis=2))
