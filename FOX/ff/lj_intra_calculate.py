r"""
FOX.ff.lj_intra_calculate
=========================

A module for calculating non-bonded intra-ligand interactions using Coulomb + Lennard-Jones potentials.

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

"""  # noqa

import operator
from typing import Union, Callable, Tuple, Optional

import numpy as np
import pandas as pd

from scm.plams import Units, PT

from .lj_calculate import get_V_elstat, get_V_lj, _get_slice_iterator
from .lj_dataframe import LJDataFrame
from .bonded_calculate import _dist
from .degree_of_separation import degree_of_separation
from ..classes.multi_mol import MultiMolecule
from ..io.read_psf import PSFContainer
from ..io.read_prm import PRMContainer

__all__ = ['get_intra_non_bonded']


def get_intra_non_bonded(mol: Union[str, MultiMolecule], psf: Union[str, PSFContainer],
                         prm: Union[str, PRMContainer],
                         scale_elstat: float = 1.0,
                         scale_lj: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    scale_elstat : :class:`float`
        Scaling factor to apply to all 1,4-nonbonded electrostatic interactions.
        Serves the same purpose as the cp2k ``EI_SCALE14`` keyword.

    scale_lj : :class:`float`
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
        mol = MultiMolecule.from_xyz(mol)
    else:
        mol = mol.copy(deep=False)

    # Define the various non-bonded atom-pairs
    core_atoms = psf.atoms.index[psf.residue_id == 1] - 1
    lig_atoms = psf.atoms.index[psf.residue_id != 1] - 1
    mol.bonds = psf.bonds - 1

    # Ensure that PLAMS more or less recognizes the new (custom) atomic symbols
    values = psf.atoms[['atom type', 'atom name']].values
    PT.symtonum.update({k.capitalize(): PT.get_atomic_number(v) for k, v in values})

    # Construct the parameter DataFrames
    mol.atoms = psf.to_atom_dict()
    prm_df = _construct_df(mol, lig_atoms, psf, prm, pairs14=False)
    prm_df14 = _construct_df(mol, lig_atoms, psf, prm, pairs14=True)

    # The .prm format allows one to specify special non-bonded interactions between
    # atoms three bonds removed
    # If not specified, do not distinguish between atoms removed 3 and >3 bonds
    if prm_df14.isnull().values.all():
        prm_df14 = prm_df.copy()

    # Calculate the 1,4 - potential energies
    elstat14_df, lj14_df = _get_V(prm_df14, mol, core_atoms, depth_comparison=operator.__eq__)
    elstat14_df *= scale_elstat
    lj14_df *= scale_lj

    # Calculate the total potential energies
    elstat_df, lj_df = _get_V(prm_df, mol, core_atoms, depth_comparison=operator.__gt__)
    elstat_df += elstat14_df
    lj_df += lj14_df

    return elstat_df, lj_df


DepthComparison = Callable[[np.ndarray, int], np.ndarray]
ANGSTROM2AU: float = Units.conversion_ratio('angstrom', 'au')


def _get_V(prm_df: pd.DataFrame, mol: MultiMolecule, core_atoms: np.ndarray,
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

    # Calculate the potential energies
    for mol_subset in slice_iterator:
        dist = _dist(mol[mol_subset] * ANGSTROM2AU, ij)  # Construct the distance matrix

        for idx, items in prm_df[['charge', 'epsilon', 'sigma']].iterrows():
            dist_slice = dist[:, np.all(symbol == sorted(idx), axis=1)]

            charge, epsilon, sigma = items
            elstat_df.loc[elstat_df.index[mol_subset], idx] = get_V_elstat(charge, dist_slice)
            lj_df.loc[lj_df.index[mol_subset], idx] = get_V_lj(sigma, epsilon, dist_slice)
    return elstat_df, lj_df


def _construct_df(mol: MultiMolecule, lig_atoms: np.ndarray,
                  psf: Union[str, PSFContainer], prm: Union[str, PRMContainer],
                  pairs14: bool = False) -> LJDataFrame:
    """Construct the DataFrame for :func:`get_intra_non_bonded`."""
    prm_df = LJDataFrame(index=set(mol.symbol[lig_atoms]))
    prm_df.overlay_psf(psf)
    prm_df.overlay_prm(prm, pairs14=pairs14)
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
    return np.array(np.where(depth_comparison(depth_mat, 3)))
