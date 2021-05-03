"""A set of functions for analyzing ligands.

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

    # Add dummy atoms to the ligand-center of mass and calculate the RDF
    >>> lig_centra: np.ndarray = get_lig_center(mol, start, step)
    >>> mol_new: MultiMolecule = mol.add_atoms(lig_centra, symbols='Xx')
    >>> rdf: pd.DataFrame = mol_new.init_rdf(atom_subset=['Xx'])

.. image:: ligand_rdf.png
    :scale: 20 %
    :align: center

Or the ADF.

.. code:: python

    >>> ...

    >>> adf: pd.DataFrame = mol_new.init_rdf(atom_subset=['Xx'], r_max=np.inf)

.. image:: ligand_adf.png
    :scale: 20 %
    :align: center

Or the potential of mean force (*i.e.* Boltzmann-inverted RDF).

.. code:: python

    >>> ...

    >>> from scipy import constants
    >>> from scm.plams import Units

    >>> RT: float = 298.15 * constants.Boltzmann
    >>> kj_to_kcal: float = Units.conversion_ratio('kj/mol', 'kcal/mol')

    >>> with np.errstate(divide='ignore'):
    >>>     rdf_invert: pd.DataFrame = -RT * np.log(rdf) * kj_to_kcal
    >>>     rdf_invert[rdf_invert == np.inf] = np.nan  # Set all infinities to not-a-number

.. image:: ligand_rdf_inv.png
    :scale: 20 %
    :align: center

Focus on a specific ligand subset is possible by slicing the new ligand Cartesian coordinate array.

.. code:: python

    >>> ...

    >>> keep_lig = [0, 1, 2, 3]  # Keep these ligands; disgard the rest
    >>> lig_centra_subset = lig_centra[:, keep_lig]

    # Add dummy atoms to the ligand-center of mass and calculate the RDF
    >>> mol_new2: MultiMolecule = mol.add_atoms(lig_centra_subset, symbols='Xx')
    >>> rdf: pd.DataFrame = mol_new2.init_rdf(atom_subset=['Xx'])

.. image:: ligand_rdf_subset.png
    :scale: 20 %
    :align: center


Examples
--------
An example for generating a ligand center of mass RDF from a quantum dot with multiple unique
ligands.
A .psf file will herein be used as starting point.

.. code:: python

    >>> import numpy as np
    >>> from FOX import PSFContainer, MultiMolecule, group_by_values
    >>> from FOX.recipes import get_multi_lig_center

    >>> mol = MultiMolecule.from_xyz(...)
    >>> psf = PSFContainer.read(...)

    # Gather the indices of each ligand
    >>> idx_dict: dict = group_by_values(enumerate(psf.residue_id, start=1))
    >>> del idx_dict[1]  # Delete the core

    # Use the .psf segment names as symbols
    >>> symbols = [psf.segment_name[i].iloc[0] for i in idx_dict.values()]

    # Add dummy atoms to the ligand-center of mass and calculate the RDF
    >>> lig_centra: np.ndarray = get_multi_lig_center(mol, idx_dict.values())
    >>> mol_new: MultiMolecule = mol.add_atoms(lig_centra, symbols=symbols)
    >>> rdf = mol_new.init_rdf(atom_subset=set(symbols))


Index
-----
.. currentmodule:: FOX.recipes
.. autosummary::
    get_lig_center
    get_multi_lig_center

API
---
.. autofunction:: get_lig_center
.. autofunction:: get_multi_lig_center

"""

from typing import Optional, Iterable, Sequence, List

import numpy as np

from FOX import MultiMolecule

__all__ = ['get_lig_center', 'get_multi_lig_center']


def get_lig_center(mol: MultiMolecule, start: int, step: int, stop: Optional[int] = None,
                   mass_weighted: bool = True) -> np.ndarray:
    """Return an array with the (mass-weighted) mean position of each ligands in **mol**.

    Parameters
    ----------
    mol : :class:`FOX.MultiMolecule`
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


def get_multi_lig_center(mol: MultiMolecule, idx_iter: Iterable[Sequence[int]],
                         mass_weighted: bool = True) -> np.ndarray:
    """Return an array with the (mass-weighted) mean position of each ligands in **mol**.

    Contrary to :func:`get_lig_center`, this function can handle molecules with multiple
    non-unique ligands.

    Parameters
    ----------
    mol : :class:`FOX.MultiMolecule`
        A MultiMolecule instance.

    idx_iter : :class:`Iterable<collections.abc.Iterable>` [:class:`Sequence<collections.abc.Sequence>` [:class:`int`]]
        An iterable consisting of integer sequences.
        Each integer sequence represents a single ligand (by its atomic indices).

    mass_weighted : :class:`bool`
        If ``True``, return the mass-weighted mean ligand position rather than
        its unweighted counterpart.

    Returns
    -------
    :class:`numpy.ndarray`
        A new array with the ligand's centra of mass.
        If ``mol.shape == (m, n, 3)`` then, given ``k`` new ligands (aka the length of **idx_iter**)
        , the to-be returned array's shape is ``(m, k, 3)``.

    """  # noqa
    mass_ = mol.mass
    ret: List[MultiMolecule] = []
    ret_append = ret.append

    for idx in idx_iter:
        ligand = mol[:, idx]
        if not mass_weighted:
            ret_append(ligand.mean(axis=1))
            continue

        mass = mass_[idx][None, ..., None]
        ligand *= mass
        ret_append(ligand.sum(axis=1) / mass.sum(axis=1))

    return np.array(ret)
