r"""
FOX.ff.lj_calculate
===================

A module for calculating non-bonded interactions using Coulomb + Lennard-Jones potentials.

See :mod:`lj_intra_calculate<FOX.ff.lj_intra_calculate>` for the calculation of non-covalent
intra-moleculair interactions.

.. math::

    V_{LJ} = 4 \varepsilon
    \left(
        \left(
            \frac{\sigma}{r}
        \right )^{12} -
        \left(
            \frac{\sigma}{r}
        \right )^6
    \right )

    V_{Coulomb} = \frac{1}{4 \pi \varepsilon_{0}} \frac{q_{i} q_{j}}{r_{ij}}

"""

import math
from typing import Mapping, Tuple, Sequence, Optional, Iterable, Union, Generator

import numpy as np
import pandas as pd

from scm.plams import Units

from .lj_dataframe import LJDataFrame
from ..functions.utils import fill_diagonal_blocks
from ..classes.multi_mol import MultiMolecule
from ..io.read_psf import PSFContainer
from ..io.read_prm import PRMContainer

__all__ = ['get_non_bonded', 'get_V']

SliceMapping = Mapping[Tuple[str, str], Tuple[Sequence[int], Sequence[int]]]
PrmMapping = Mapping[Tuple[str, str], Tuple[float, float, float]]


def get_non_bonded(mol: Union[str, MultiMolecule],
                   psf: Union[str, PSFContainer],
                   prm: Union[None, str, PRMContainer] = None,
                   rtf: Optional[str] = None,
                   cp2k_settings: Optional[Mapping] = None,
                   max_array_size: int = 10**8) -> pd.DataFrame:
    r"""Collect forcefield parameters and calculate all non-covalent interactions in **mol**.

    Forcefield parameters (*i.e.* charges and Lennard-Jones :math:`\sigma` and
    :math:`\varepsilon` values) are collected from the provided **psf**, **prm**, **rtf** and/or
    **cp2k_settings** parameters.
    Note that only that only **psf** is strictly required, though due to the lack of
    :math:`\sigma` and :math:`\varepsilon` values the resulting potential energy
    will be exclusivelly electrostatic in nature.
    Supplying a **prm** and/or **rtf** file is redundant if
    all parameters are specified in **cp2k_settings**.

    Intra-ligand interactions are ignored.

    Parameters
    ----------
    mol : :class:`str` or :class:`MultiMolecule`
        A MultiMolecule instance or the path+filename of an .xyz file.

    psf : :class:`str` or :class:`PSFContainer`
        A PSFContainer instance or the path+filename of a .psf file.
         Used for setting :math:`q` and creating atom-subsets.

    prm : :class:`str` or :class:`PRMContainer`, optional
        A PRMContainer instance or the path+filename of a .prm file.
        Used for setting :math:`\sigma` and :math:`\varepsilon`.

    rtf : :class:`str`, optional
        The path+filename of an .rtf file.
        Used for setting :math:`q`.

    cp2k_settings : :class:`Settings`, optional
        CP2K input settings.
        Used for setting :math:`q`, :math:`\sigma` and :math:`\varepsilon`.

    max_array_size : :class:`int`
        The maximum number of elements within the to-be created NumPy array.
        NumPy's vectorized operations will be (partially) substituted for for-loops if the
        array size is exceeded.

    Returns
    -------
    :class:`pandas.DataFrame`
        A DataFrame with the electrostatic and Lennard-Jones components of the
        (inter-ligand) potential energy per atom-pair.
        The potential energy is summed over atoms with matching atom types and
        averaged over all molecules within **mol**.
        Units are in atomic units.

    See Also
    --------
    :func:`get_V`
        Calculate all non-covalent interactions averaged over all molecules in **mol**.

    """
    if not isinstance(psf, PSFContainer):
        psf = PSFContainer.read(psf)

    if not isinstance(mol, MultiMolecule):
        mol = MultiMolecule.from_xyz(mol)
    else:
        mol = mol.copy(deep=False)
    mol.atoms = mol_atoms = psf.to_atom_dict()

    prm_df = LJDataFrame(index=mol_atoms.keys())
    prm_df.overlay_psf(psf)
    if prm is not None:
        prm_df.overlay_prm(prm)
    if cp2k_settings is not None:
        prm_df.overlay_cp2k_settings(cp2k_settings)

    slice_dict = {}
    for i, j in prm_df.index:
        try:
            slice_dict[i, j] = mol_atoms[i], mol_atoms[j]
        except KeyError:
            pass
    core_atoms = set(psf.atom_type[psf.residue_id == 1])
    ligand_count = psf.residue_id.max() - 1
    return get_V(mol, slice_dict, prm_df.loc, ligand_count, core_atoms=core_atoms)


def get_V(mol: MultiMolecule, slice_mapping: SliceMapping,
          prm_mapping: PrmMapping, ligand_count: int,
          core_atoms: Optional[Iterable[str]] = None,
          max_array_size: int = 10**7) -> pd.DataFrame:
    r"""Calculate all non-covalent interactions averaged over all molecules in **mol**.

    Parameters
    ----------
    mol : :class:`MultiMolecule`
        A MultiMolecule instance.

    slice_mapping : :class:`dict`
        A mapping of atoms-pairs to matching atomic indices.

    prm_mapping : :class:`dict`
        A mapping of atoms-pairs to matching (pair-wise) values for :math:`q`,
        :math:`sigma` and :math:`\varepsilon`.
        Units should be in atomic units.

    ligand_count : :class:`int`
        The number of ligands.

    core_atoms : :class:`set` [:class:`str`], optional
        A set of all atoms within the core.

    max_array_size : :class:`int`
        The maximum number of elements within the to-be created NumPy array.
        NumPy's vectorized operations will be (partially) substituted for for-loops if the
        array size is exceeded.

    Returns
    -------
    :class:`pandas.DataFrame`
        A DataFrame with the electrostatic and Lennard-Jones components of the
        (inter-ligand) potential energy per atom-pair.
        The potential energy is summed over atoms with matching atom types and
        averaged over all molecules within **mol**.
        Units are in atomic units.

    """
    core_atoms = set(core_atoms) if core_atoms is not None else set()
    mol = mol * Units.conversion_ratio('Angstrom', 'au')

    df = pd.DataFrame(
        0.0,
        index=pd.MultiIndex.from_tuples(sorted(slice_mapping.keys())),
        columns=pd.Index(['elstat', 'lj'], name='au')
    )

    for atoms, ij in slice_mapping.items():
        charge, epsilon, sigma = prm_mapping[atoms]
        contains_core = core_atoms.intersection(atoms)

        dmat_size = len(ij[0]) * len(ij[1])  # The size of a single (2D) distance matrix
        slice_iterator = _get_slice_iterator(len(mol), dmat_size, max_array_size)

        for mol_subset in slice_iterator:
            dist = _get_dist(mol, ij, ligand_count, contains_core, mol_subset=mol_subset)
            df.at[atoms, 'elstat'] += get_V_elstat(charge, dist)
            df.at[atoms, 'lj'] += get_V_lj(sigma, epsilon, dist)

        if atoms[0] == atoms[1]:  # Avoid double-counting
            df.loc[atoms] /= 2

    df /= len(mol)
    return df


def _get_dist(mol: MultiMolecule, ij: np.ndarray, ligand_count: int,
              contains_core: bool, mol_subset: Optional[slice]) -> np.ndarray:
    """Construct an array of distance matrices for :func:`get_V`."""
    dist = mol.get_dist_mat(mol_subset=mol_subset, atom_subset=ij)
    if not contains_core:
        i = len(ij[0]) // ligand_count
        j = len(ij[1]) // ligand_count
        fill_diagonal_blocks(dist, i, j)  # Set intra-ligand interactions to np.nan
    else:
        dist[dist == 0.0] = np.nan

    return dist


def _get_slice_iterator(stop: int, dmat_size: int,
                        max_array_size: int = 10**7) -> Generator[slice, None, None]:
    """Return a generator yielding :class:`slice` instances for :func:`get_V`."""
    if stop * dmat_size < max_array_size:
        step = stop
    else:
        step = max(1, math.floor(max_array_size / dmat_size))

    # Yield the slices
    start = 0
    while start < stop:
        yield slice(start, start+step)
        start += step


def get_V_elstat(q: float, dist: np.ndarray) -> float:
    r"""Calculate and sum the electrostatic potential energy given a distance matrix **dist**..

    .. math::

        V_{elstat} = \sum_{i, j} \frac{1}{4 \pi \epsilon_{0}} \frac{q_{ij}}{r_{ij}}

        q_{ij} = q_{i} * q_{j}

        4 \pi \epsilon_{0} = 1

    Parameters
    ----------
    q : :class:`float`
        The product of two charges :math:`q_{ij}`.
    dist : :class:`numpy.ndarray`
        The distance matrix :math:`r_{ij}`.
        Units should be in Bohr.

    Returns
    -------
    :class:`float`
        The elctrostatic potential energy summed over all distance pairs in **dist**.

    """
    return np.nansum(q / dist)


def get_V_lj(sigma: float, epsilon: float, dist: np.ndarray) -> float:
    r"""Calculate and sum the Lennard-Jones potential given a distance matrix **dist**.

    .. math::

        V_{LJ} = \sum_{i, j} 4 \varepsilon_{ij}
        \left(
            \left(
                \frac{\sigma_{ij}}{r_{ij}}
            \right )^{12} -
            \left(
                \frac{\sigma_{ij}}{r_{ij}}
            \right )^6
        \right )

        \sigma_{ij} = \frac{\sigma_{i} \sigma_{j}}{2}

        \varepsilon_{ij} = \sqrt{\varepsilon_{i} \varepsilon_{j}}

    Parameters
    ----------
    sigma : :class:`float`
        The arithmetic mean of two :math:`\sigma` parameters: :math:`\sigma_{ij}`.
        Units should be in Bohr.
    epsilon : :class:`float`
        The geometric mean of two :math:`\varepsilon` parameters: :math:`\varepsilon_{ij}`.
        Units should be in Hartree.
    dist : :class:`numpy.ndarray`
        The distance matrix :math:`r_{ij}`.
        Units should be in Bohr.

    Returns
    -------
    :class:`float`
        The Lennard-Jones potential energy summed over all distance pairs in **dist**.

    """
    sigma_dist = (sigma / dist)**6
    lj = sigma_dist**2 - sigma_dist
    lj *= epsilon * 4
    return np.nansum(lj)
