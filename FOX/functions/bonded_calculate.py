r"""
FOX.functions.bonded_calculate
==============================

A module for calculating bonded interactions using harmonic + cosine potentials.

.. math::

    V_{bonds} = k_{r} (r - r_{0})^2

    V_{angles} = k_{\theta} (\theta - \theta_{0})^2

    V_{diehdrals} = k_{\phi} [1 + \cos(n \phi - \delta)]

    V_{impropers} = k_{\omega} (\omega - \omega_{0})^2

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

import textwrap
from types import MappingProxyType
from typing import Mapping, Tuple, Sequence, Optional, Iterable, Union, Dict, List
from itertools import combinations_with_replacement, permutations
from collections import abc

import numpy as np
import pandas as pd

from scm.plams import Units, Settings

from FOX.functions.utils import read_rtf_file, fill_diagonal_blocks, group_by_values
from FOX import MultiMolecule, get_example_xyz
from FOX.io.read_psf import PSFContainer
from FOX.io.read_prm import PRMContainer
from FOX.functions.lj_calculate import LJDataFrame, psf_to_atom_dict

__all__ = []

SliceMapping = Mapping[Tuple[str, ...], Tuple[Sequence[int], ...]]
PrmMapping = Mapping[Tuple[str, ...], Tuple[float, ...]]


def get_bonded(mol: Union[str, MultiMolecule],
                   psf: Union[str, PSFContainer],
                   prm: Union[None, str, PRMContainer] = None) -> pd.DataFrame:
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
    psf = PSFContainer.read(psf)

    if not isinstance(mol, MultiMolecule):
        mol = MultiMolecule.from_xyz(mol)
    mol.atoms = mol_atoms = psf_to_atom_dict(psf)

    prm_df = LJDataFrame(index=mol_atoms.keys())
    prm_df.overlay_psf(psf)
    prm_df.overlay_prm(prm)

    slice_dict = {(i, j): (mol_atoms[i], mol_atoms[j]) for i, j in prm_df.index}
    return get_V_bonded(mol, slice_dict, prm_df.loc)


def get_V_bonded(mol: MultiMolecule, slice_mapping: SliceMapping,
                 prm_mapping: PrmMapping) -> pd.DataFrame:
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

    Returns
    -------
    :class:`pandas.DataFrame`
        A DataFrame with the electrostatic and Lennard-Jones components of the
        (inter-ligand) potential energy per atom-pair.
        The potential energy is summed over atoms with matching atom types and
        averaged over all molecules within **mol**.
        Units are in atomic units.

    """
    mol = mol * Units.conversion_ratio('Angstrom', 'au')

    df = pd.DataFrame(
        0.0,
        index=pd.MultiIndex.from_tuples(sorted(slice_mapping.keys())),
        columns=pd.Index(['bond', 'angle', 'improper', 'proper'], name='au')
    )

    for atoms, ij in slice_mapping.items():
        df.at[atoms, 'bonds'] = get_V_harmonic(...)
        df.at[atoms, 'angles'] = get_V_harmonic(...)
        df.at[atoms, 'dihedrals'] = get_V_harmonic(...)
        df.at[atoms, 'impropers'] = get_V_cos(...)

    df /= len(mol)
    return df


def overlay_prm(prm: Union[PRMContainer, str], df: pd.DataFrame) -> None:
    if not isinstance(prm, PRMContainer):
        prm = PRMContainer.read(prm)

    if prm.bonds is not None:
        bonds = prm.bonds.set_index([0, 1])
        df.at['bon']

    if prm.angles is not None:
        angles = prm.angles.set_index([0, 1, 2])

    if prm.dihedrals is not None:
        dihedrals = prm.dihedrals.set_index([0, 1, 2, 3])

    if prm.impropers is not None:
        impropers = prm.impropers.set_index([0, 1, 2, 3])


def _dist(mol: np.ndarray, ij: np.ndarray) -> np.ndarray:
    """Return an array with :code:`len(mol), len(ij)` distances (same unit as **mol**)."""
    i, j = ij.T
    return np.linalg.norm(mol[:, i] - mol[:, j], axis=-1)


def _angle(mol: np.ndarray, ijk: np.ndarray) -> np.ndarray:
    """Return an array with :code:`len(mol), len(ijk)` angles (radian)."""
    i, j, k = ijk.T

    vec1 = (mol[:, i] - mol[:, j])
    vec2 = (mol[:, k] - mol[:, j])
    vec1 /= np.linalg.norm(vec1, axis=-1)[..., None]
    vec2 /= np.linalg.norm(vec2, axis=-1)[..., None]

    return np.arccos((vec1 * vec2).sum(axis=2))


def _dihed(mol: np.ndarray, ijkm: np.ndarray) -> np.ndarray:
    """Return an array with :code:`len(mol), len(ijkm)` dihedral angles (radian)."""
    i, j, k, m = ijkm.T
    b0 = mol[:, i] - mol[:, j]
    b1 = mol[:, k] - mol[:, j]
    b2 = mol[:, m] - mol[:, k]

    b1 /= np.linalg.norm(b1, axis=-1)[..., None]
    v = b0 - (b0 * b1).sum(axis=2)[..., None] * b1
    w = b2 - (b2 * b1).sum(axis=2)[..., None] * b1

    x = (v * w).sum(axis=2)
    y = (np.cross(b1, v) * w).sum(axis=2)
    ret = np.arctan2(y, x)
    return np.abs(ret)


def _improp(mol: np.ndarray, ijkm: np.ndarray) -> np.ndarray:
    """Return an array with :code:`len(mol), len(ijkm)` improper dihedral angles (radian)."""
    j, i, k, m = ijkm.T
    jikm = np.array([j, i, k, m]).T
    return _dihed(mol, jikm)


def get_V_harmonic(x: np.ndarray, k: float, x0: float) -> float:
    r"""Calculate the harmonic potential energy: :math:`\sum_{i} k (x_{i} - x_{0})^2`.

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        An array of geometry parameters such as distances, angles or improper dihedral angles.

    x_ref : :class:`float`
        The equilibrium value of **x**.

    k : :class:`float`
        The force constant :math:`k`.

    """
    return np.mean(k * (x - x0)**2, axis=0)


def get_V_cos(phi: np.ndarray, k: float, n: int, delta: float = 0.0) -> float:
    r"""Calculate the cosine potential energy: :math:`\sum_{i} k_{\phi} [1 + \cos(n \phi_{i} - \delta)]`.

    Parameters
    ----------
    phi : :class:`numpy.ndarray`
        An array of dihedral angles.

    k : :class:`float`
        The force constant :math:`k`.

    n : :class:`int`
        The multiplicity :math:`n`.

    delta : :class:`float`
        The phase-correction :math:`\delta`.

    """  # noqa
    V = k * np.cos(n * phi - delta)
    return V.mean(axis=0)


prm_file = '/Users/basvanbeek/Documents/GitHub/auto-FOX/FOX/examples/ligand.prm'
psf_file = '/Users/basvanbeek/Downloads/mol.psf'
xyz_file = get_example_xyz()

psf = PSFContainer.read(psf_file)
psf.bonds -= 1
psf.angles -= 1
psf.impropers -= 1

mol = MultiMolecule.from_xyz(xyz_file)
mol.atoms = psf_to_atom_dict(psf)

prm = PRMContainer.read(prm_file)
prm.bonds.set_index([0, 1], inplace=True)
prm.angles.set_index([0, 1, 2], inplace=True)
prm.impropers.set_index([0, 1, 2, 3], inplace=True)

df2 = pd.DataFrame(prm.bonds[[2, 3]].copy())
df2.columns = ['k', 'r0']
df2[:] = df2.values.astype(float)
df2['V_bonds'] = np.nan

df3 = pd.DataFrame(prm.angles[[3, 4]].copy())
df3.columns = ['k', 'theta0']
df3[:] = df3.values.astype(float)
df3['theta0'] = np.radians(df3['theta0'])
df3['V_angles'] = np.nan

df4 = pd.DataFrame(prm.impropers[[4, 6]].copy())
df4.columns = ['k', 'omega0']
df4[:] = df4.values.astype(float)
df4['V_impropers'] = np.nan

dist = np.array(_dist(mol, psf.bonds))
angle = np.array(_angle(mol, psf.angles))
improp = np.array(_improp(mol, psf.impropers))

symbol = mol.symbol
for i, item in df2.iloc[:, 0:2].iterrows():
    j = np.all(symbol[psf.bonds] == i, axis=1)
    j |= np.all(symbol[psf.bonds[:, ::-1]] == i, axis=1)
    df2.at[i, 'V_bonds'] = get_V_harmonic(dist[:, j], *item).sum()

for i, item in df3.iloc[:, 0:2].iterrows():
    j = np.all(symbol[psf.angles] == i, axis=1)
    j |= np.all(symbol[psf.angles[:, ::-1]] == i, axis=1)
    df3.at[i, 'V_angles'] = get_V_harmonic(angle[:, j], *item).sum()

for i, item in df4.iloc[:, 0:2].iterrows():
    j = np.zeros(len(psf.impropers), dtype=bool)
    for k in permutations([1, 2, 3], r=3):
        k = (0,) + k
        j |= np.all(symbol[psf.impropers[:, k]] == i, axis=1)
    df4.at[i, 'V_impropers'] = get_V_harmonic(improp[:, j], *item).sum()
