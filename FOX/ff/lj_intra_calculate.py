r"""A module for calculating non-bonded intra-ligand interactions using Coulomb + Lennard-Jones potentials.

See :mod:`lj_calculate<FOX.ff.lj_calculate>` for the calculation of non-covalent inter-moleculair
interactions.

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
.. currentmodule:: FOX.ff.lj_intra_calculate
.. autosummary::
    get_intra_non_bonded

API
---
.. autofunction:: get_intra_non_bonded

"""  # noqa: E501

import operator
from typing import Union, Callable, Tuple, Optional

import numpy as np
import pandas as pd

from scm.plams import Units, PT

from . import LJDataFrame
from .degree_of_separation import degree_of_separation
from .lj_calculate import get_V_elstat, get_V_lj, _get_slice_iterator
from .bonded_calculate import _dist
from ..classes import MultiMolecule
from ..io import PSFContainer, PRMContainer

__all__ = ['get_intra_non_bonded']


def get_intra_non_bonded(mol: Union[str, MultiMolecule], psf: Union[str, PSFContainer],
                         prm: Union[str, PRMContainer],
                         distance_upper_bound: float = np.inf,
                         shift_cutoff: bool = True,
                         el_scale14: float = 1.0,
                         lj_scale14: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r"""Collect forcefield parameters and calculate all non-covalent intra-ligand interactions in **mol**.

    Forcefield parameters (*i.e.* charges and Lennard-Jones :math:`\sigma` and
    :math:`\varepsilon` values) are collected from the provided **psf** and **prm** files.

    Inter-ligand, core-core and intra-core interactions are ignored.

    Parameters
    ----------
    mol : :class:`str` or :class:`MultiMolecule`
        A MultiMolecule instance or the path+filename of an .xyz file.

    psf : :class:`str` or :class:`PSFContainer`
        A PSFContainer instance or the path+filename of a .psf file.
         Used for setting :math:`q` and creating atom-subsets.

    prm : :class:`str` or :class:`PRMContainer`
        A PRMContainer instance or the path+filename of a .prm file.
        Used for setting :math:`\sigma` and :math:`\varepsilon`.

    distance_upper_bound : :class:`float`
        Consider only atom-pairs within this distance.
        Using ``inf`` will default to the full, untruncated, distance matrix.

    shift_cutoff : :class:`bool`
        Shift all potentials by a constant such that
        it is equal to zero at **distance_upper_bound**.
        Only relavent when ``distance_upper_bound < inf``.
        Serves the same purpose as the cp2k ``SHIFT_CUTOFF`` keyword.

    el_scale14 : :class:`float`
        Scaling factor to apply to all 1,4-nonbonded electrostatic interactions.
        Serves the same purpose as the cp2k ``EI_SCALE14`` keyword.

    lj_scale14 : :class:`float`
        Scaling factor to apply to all 1,4-nonbonded Lennard-Jones interactions.
        Serves the same purpose as the cp2k ``VDW_SCALE14`` keyword.

    Returns
    -------
    :class:`pandas.DataFrame` & :class:`pandas.DataFrame`
        Two DataFrames with, respectivelly, the electrostatic and Lennard-Jones components of the
        (intra--ligand) potential energy per atom-pair.
        The potential energy is summed over atoms with matching atom types.
        Units are in atomic units.

    """  # noqa
    if not isinstance(psf, PSFContainer):
        psf = PSFContainer.read(psf)

    if not isinstance(mol, MultiMolecule):
        mol_ = MultiMolecule.from_xyz(mol)
    else:
        mol_ = mol.copy(deep=False)

    # Define the various non-bonded atom-pairs
    core_atoms = psf.atoms.index[psf.residue_id == 1] - 1
    lig_atoms = psf.atoms.index[psf.residue_id != 1] - 1
    mol_.bonds = psf.bonds - 1

    # Ensure that PLAMS more or less recognizes the new (custom) atomic symbols
    values = psf.atoms[['atom type', 'atom name']].values
    PT.symtonum.update({k.capitalize(): PT.get_atomic_number(v) for k, v in values})

    # Construct the parameter DataFrames
    mol_.atoms = psf.to_atom_dict()
    prm_df = _construct_df(mol_, lig_atoms, psf, prm, pairs14=False)
    prm_df14 = _construct_df(mol_, lig_atoms, psf, prm, pairs14=True)
    mol_.atoms = psf.to_atom_dict()

    # Convert Angstroem to bohr
    mol_ *= Units.conversion_ratio('angstrom', 'au')  # type: ignore[assignment]
    distance_upper_bound *= Units.conversion_ratio('angstrom', 'au')

    # The .prm format allows one to specify special non-bonded interactions between
    # atoms three bonds removed
    # If not specified, do not distinguish between atoms removed 3 and >3 bonds
    if prm_df14.isnull().values.all():
        prm_df14 = prm_df.copy()

    elif el_scale14 == lj_scale14 == 1:
        # Don't bother with a separate calculation for 1,4-nonbonded interactions
        return _get_V(prm_df, mol_, core_atoms,
                      shift_cutoff=shift_cutoff,
                      distance_upper_bound=distance_upper_bound,
                      depth_comparison=operator.__ge__)

    # Calculate the 1,4 - potential energies
    elstat14_df, lj14_df = _get_V(prm_df14, mol_, core_atoms,
                                  shift_cutoff=shift_cutoff,
                                  distance_upper_bound=distance_upper_bound,
                                  depth_comparison=operator.__eq__)
    elstat14_df *= el_scale14
    lj14_df *= lj_scale14

    # Calculate the total potential energies
    elstat_df, lj_df = _get_V(prm_df, mol_, core_atoms,
                              shift_cutoff=shift_cutoff,
                              distance_upper_bound=distance_upper_bound,
                              depth_comparison=operator.__gt__)
    elstat_df += elstat14_df
    lj_df += lj14_df

    return elstat_df, lj_df


DepthComparison = Callable[[np.ndarray, int], np.ndarray]


def _get_V(prm_df: pd.DataFrame, mol: MultiMolecule, core_atoms: np.ndarray,
           distance_upper_bound: float = np.inf,
           shift_cutoff: bool = True,
           depth_comparison: DepthComparison = operator.__ge__,
           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Construct the distance matrix; calculate the potential and update the **prm_df** with the energies."""  # noqa
    ij = _get_idx(mol, core_atoms, depth_comparison=depth_comparison).T

    # Construct
    index = pd.RangeIndex(0, len(mol), name='MD Iteration')
    elstat_df = pd.DataFrame(0.0, index=index.copy(), columns=prm_df.index.copy())
    lj_df = pd.DataFrame(0.0, index=index, columns=prm_df.index.copy())

    if not ij.any():
        return elstat_df, lj_df

    # Map each atom-index pair (ij) to a pair of atomic symbols (more specifically: their hashes)
    symbol = mol.symbol[ij]
    symbol.sort(axis=1)

    dmat_size = len(ij)  # The size of a single (2D) distance matrix
    len_mol = len(mol)
    slice_iterator = _get_slice_iterator(len_mol, dmat_size)

    if distance_upper_bound < np.inf and shift_cutoff:
        shift: Optional[float] = distance_upper_bound
    else:
        shift = None

    # Calculate the potential energies
    for mol_subset in slice_iterator:
        dist = _dist(mol[mol_subset], ij)  # Construct the distance matrix
        if distance_upper_bound != np.inf:
            dist[dist > distance_upper_bound] = np.nan

        for idx, items in prm_df[['charge', 'epsilon', 'sigma']].iterrows():
            dist_slice = dist[:, (symbol == sorted(idx)).all(axis=1)]
            charge, epsilon, sigma = items

            elstat_df.loc[elstat_df.index[mol_subset], idx] = get_V_elstat(
                charge, dist_slice, shift_cutoff=shift

            )

            lj_df.loc[lj_df.index[mol_subset], idx] = get_V_lj(
                sigma, epsilon, dist_slice, shift_cutoff=shift
            )

    return elstat_df, lj_df


def _construct_df(mol: MultiMolecule, lig_atoms: np.ndarray,
                  psf: Union[str, PSFContainer], prm: Union[str, PRMContainer],
                  pairs14: bool = False) -> LJDataFrame:
    """Construct the DataFrame for :func:`get_intra_non_bonded`."""
    prm_df = LJDataFrame(index=set(mol.symbol[lig_atoms]))
    prm_df.overlay_prm(prm, pairs14=pairs14)
    prm_df.overlay_psf(psf)
    prm_df.dropna(inplace=True)
    return prm_df


def _get_idx(mol: MultiMolecule, core_atoms: np.ndarray, inf2value: Optional[float] = 0.0,
             depth_comparison: DepthComparison = operator.__ge__) -> np.ndarray:
    """Construct the array with all atom-pairs valid for intra-moleculair non-covalent interactions."""  # noqa
    data = np.ones(len(mol.bonds), dtype=bool)
    rows, columns = mol.atom12.T

    depth_mat = degree_of_separation(mol[0], bond_mat=(data, (rows, columns)))
    if inf2value is not None:
        depth_mat[np.isposinf(depth_mat)] = inf2value

    depth_mat_triu = np.triu(depth_mat)  # Ignore all pairs below the diagonal
    return np.array(np.where(depth_comparison(depth_mat_triu, 3)))
