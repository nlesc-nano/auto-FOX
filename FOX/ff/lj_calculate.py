r"""A module for calculating non-bonded interactions using Coulomb + Lennard-Jones potentials.

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

Index
-----
.. currentmodule:: FOX.ff.lj_calculate
.. autosummary::
    get_non_bonded
    get_V
    MAX_ARRAY_SIZE

API
---
.. autofunction:: get_bonded
.. autofunction:: get_V
.. autodata:: MAX_ARRAY_SIZE

"""

import math
import functools
from typing import Mapping, Tuple, Sequence, Optional, Iterable, Union, Generator, Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from scm.plams import Units, Settings
from nanoutils import fill_diagonal_blocks

from . import LJDataFrame
from ..classes import MultiMolecule
from ..io import PSFContainer, PRMContainer

__all__ = ['get_non_bonded', 'get_V', 'MAX_ARRAY_SIZE']

SliceMapping = Mapping[Tuple[str, str], Tuple[Sequence[int], Sequence[int]]]
PrmMapping = Mapping[Tuple[str, str], Tuple[float, float, float]]

#: The maximum number of elements to-be simultaneously stored in a single ndarray
MAX_ARRAY_SIZE: int = 10**8


def get_non_bonded(mol: Union[str, MultiMolecule],
                   psf: Union[str, PSFContainer],
                   prm: Union[None, str, PRMContainer] = None,
                   rtf: Optional[str] = None,
                   cp2k_settings: Optional[Mapping] = None,
                   distance_upper_bound: float = np.inf, k: int = 20,
                   shift_cutoff: bool = True,
                   atom_pairs: Optional[Iterable[Tuple[str, str]]] = None,
                   el_scale14: Any = None, lj_scale14: Any = None
                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    distance_upper_bound : :class:`float`
        Consider only atom-pairs within this distance.
        Using ``inf`` will default to the full, untruncated, distance matrix.

    k : :class:`int`
        The (maximum) number of to-be considered atom-pairs.
        Only relevant when **distance_upper_bound** is not set ``inf``.

    shift_cutoff : :class:`bool`
        Shift all potentials by a constant such that
        it is equal to zero at **distance_upper_bound**.
        Only relavent when ``distance_upper_bound < inf``.

    atom_pairs : :class:`Iterable<collections.abc.Iterable>` [:class:`tuple`], optional
        Explicitly specify all to-be considered atom-pairs.

    el_scale14 & el_scale14 : :data:`Any<typing.Any>`
        Dummy keywords for ensuring signature compatibility.

    Returns
    -------
    :class:`pandas.DataFrame` & :class:`pandas.DataFrame`
        Two DataFrames with, respectivelly, the electrostatic and Lennard-Jones components of the
        (inter-ligand) potential energy per atom-pair.
        The potential energy is summed over atoms with matching atom types.
        Units are in atomic units.

    See Also
    --------
    :func:`get_V`
        Calculate all non-covalent interactions averaged over all molecules in **mol**.

    :class:`cKDTree<scipy.spatial.cKDTree>
        kd-tree for quick nearest-neighbor lookup.

    """
    # Parse input parameters
    if not isinstance(psf, PSFContainer):
        psf = PSFContainer.read(psf)

    if not isinstance(mol, MultiMolecule):
        mol = MultiMolecule.from_xyz(mol)
    else:
        mol = mol.copy(deep=False)

    # Create the parameter DataFrame
    prm_df = LJDataFrame(index=psf.to_atom_dict())
    if prm is not None:
        prm_df.overlay_prm(prm)
    prm_df.overlay_psf(psf)
    if cp2k_settings is not None:
        prm_df.overlay_cp2k_settings(Settings(cp2k_settings), psf)
    mol.atoms = psf.to_atom_dict()

    # Delete all rows from prm_df whose indices are not in **atom_pairs**
    if atom_pairs is not None:
        idx_set = set(prm_df.index)
        difference = idx_set.difference(tuple(sorted(i)) for i in atom_pairs)
        prm_df.drop(difference, inplace=True)

    # Define atom subsets
    slice_dict = {}
    for i, j in prm_df.index:
        try:
            slice_dict[i, j] = mol.atoms[i], mol.atoms[j]
        except KeyError:
            pass

    # Calculate and return the potential energies
    core_atoms = set(psf.atom_type[psf.residue_id == 1])
    ligand_count = psf.residue_id.max() - 1
    return get_V(mol, slice_dict, prm_df.loc, ligand_count,
                 k=k, core_atoms=core_atoms,
                 distance_upper_bound=distance_upper_bound,
                 shift_cutoff=shift_cutoff)


def get_V(mol: MultiMolecule, slice_mapping: SliceMapping,
          prm_mapping: PrmMapping, ligand_count: int,
          core_atoms: Optional[Iterable[str]] = None,
          distance_upper_bound: float = np.inf, k: Optional[int] = 20,
          shift_cutoff: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    distance_upper_bound : :class:`float`
        Consider only atom-pairs within this distance.

    k : :class:`int`
        The (maximum) number of to-be considered atom-pairs.
        Only relevant when **distance_upper_bound** is not set ``inf``.

    shift_cutoff : :class:`bool`
        Shift all potentials by a constant such that
        it is equal to zero at **distance_upper_bound**.
        Only relavent when ``distance_upper_bound < inf``.

    Returns
    -------
    :class:`pandas.DataFrame` & :class:`pandas.DataFrame`
        Two DataFrames with, respectivelly, the electrostatic and Lennard-Jones components of the
        (inter-ligand) potential energy per atom-pair.
        The potential energy is summed over atoms with matching atom types.
        Units are in atomic units.

    """
    core_atoms = set(core_atoms) if core_atoms is not None else set()
    mol = mol * Units.conversion_ratio('Angstrom', 'au')  # type: ignore[assignment]

    index = pd.RangeIndex(0, len(mol), name='MD Iteration')
    columns = pd.MultiIndex.from_tuples(sorted(slice_mapping.keys()), names=['atom1', 'atom2'])

    elstat_df = pd.DataFrame(0.0, index=index.copy(), columns=columns.copy())
    lj_df = pd.DataFrame(0.0, index=index, columns=columns)

    # Specify the function for calculating the distance matrices
    if np.isinf(distance_upper_bound):
        dist_func = _get_dist
        k = None
    else:
        dist_func = functools.partial(_get_kd_dist, k=k, distance_upper_bound=distance_upper_bound)

    if distance_upper_bound < np.inf and shift_cutoff:
        shift: Optional[float] = distance_upper_bound * Units.conversion_ratio('angstrom', 'au')
    else:
        shift = None

    for atoms, ij in slice_mapping.items():
        charge, epsilon, sigma = prm_mapping[atoms]
        contains_core = bool(core_atoms.intersection(atoms))

        # Construct a :class:`slice` iterator based on the expected array size.
        # Precaution against creating arrays too large to hold in memory
        dmat_size = len(ij[0]) * (k or len(ij[1]))
        slice_iterator = _get_slice_iterator(len(mol), dmat_size)

        # Construct the distance matrices and calculate the potential energies
        for mol_subset in slice_iterator:
            dist = dist_func(mol, ij, ligand_count, contains_core, mol_subset=mol_subset)
            elstat_df.loc[elstat_df.index[mol_subset], atoms] = get_V_elstat(charge, dist,
                                                                             shift_cutoff=shift)
            lj_df.loc[lj_df.index[mol_subset], atoms] = get_V_lj(sigma, epsilon, dist,
                                                                 shift_cutoff=shift)
            del dist

        if atoms[0] == atoms[1]:  # Avoid double-counting
            elstat_df[atoms] /= 2
            lj_df[atoms] /= 2

    return elstat_df, lj_df


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


def _get_kd_dist(mol: MultiMolecule, ij: np.ndarray, ligand_count: int,
                 contains_core: bool, mol_subset: Optional[slice],
                 k: int = 20, distance_upper_bound: float = 10.0) -> np.ndarray:
    """Construct an array of truncated distance matrices for :func:`get_V`."""
    i_ar, j_ar = ij
    mol1 = mol[mol_subset, i_ar]
    mol2 = mol[mol_subset, j_ar]

    # Fill the (truncated) distance tensor
    dist = np.empty((len(mol1), len(i_ar), k), dtype=float)
    idx = np.empty_like(dist, dtype=int)
    for n, (xyz1, xyz2) in enumerate(zip(mol1, mol2)):
        tree = cKDTree(xyz2)
        dist[n], idx[n] = tree.query(xyz1, k=k, distance_upper_bound=distance_upper_bound)

    # Set all intra-ligand interaction to np.nan
    if not contains_core:
        i = len(i_ar) // ligand_count
        j = len(j_ar) // ligand_count

        ax1, ax2, ax3 = idx.shape
        idx.shape = ax1, ax2 // i, i, ax3
        idx -= np.arange(0, j * ax2 // i, j)[None, ..., None, None]
        idx.shape = ax1, ax2, ax3

        dist[np.isin(idx, np.arange(j))] = np.nan

    # Set zero and infinity to np.nan
    dist[dist == 0.0] = np.nan
    dist[np.isinf(dist)] = np.nan
    return dist


def _get_slice_iterator(stop: int, dmat_size: int) -> Generator[slice, None, None]:
    """Return a generator yielding :class:`slice` instances for :func:`get_V`."""
    if stop * dmat_size < MAX_ARRAY_SIZE:
        step = stop
    else:
        step = max(1, math.floor(MAX_ARRAY_SIZE / dmat_size))

    # Yield the slices
    start = 0
    while start < stop:
        yield slice(start, start+step)
        start += step


def get_V_elstat(q: float, dist: np.ndarray,
                 shift_cutoff: Optional[float] = None) -> np.ndarray:
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

    shift_cutoff : :class:`float`, optional
        When not ``None``, add a constant to the returned potantials such
        that its value is zero at the specified distance.
        Units should be in Bohr.

    Returns
    -------
    :class:`float`
        The elctrostatic potential energy summed over all distance pairs in **dist** along
        axises ``>= 1``.

    """
    ret = q / np.asarray(dist)
    axis = tuple(range(1, ret.ndim))
    if shift_cutoff is not None:
        ret -= get_V_elstat(q, shift_cutoff, shift_cutoff=None)
    return np.nansum(ret, axis=axis)


def get_V_lj(sigma: float, epsilon: float, dist: np.ndarray,
             shift_cutoff: Optional[float] = None) -> np.ndarray:
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

    shift_cutoff : :class:`float`, optional
        When not ``None``, add a constant to the returned potantials such
        that its value is zero at the specified distance.
        Units should be in Bohr.

    Returns
    -------
    :class:`numpy.ndarray`
        The Lennard-Jones potential energy summed over all distance pairs in **dist** along
        axises ``>= 1``.

    """
    sigma_dist = (sigma / np.asarray(dist))**6
    lj = sigma_dist**2 - sigma_dist
    lj *= epsilon * 4

    axis = tuple(range(1, lj.ndim))
    if shift_cutoff is not None:
        lj -= get_V_lj(sigma, epsilon, shift_cutoff, shift_cutoff=None)
    return np.nansum(lj, axis=axis)
