r"""
FOX.ff.bonded_calculate
=======================

A module for calculating bonded interactions using harmonic + cosine potentials.

.. math::

    V_{bonds} = k_{r} (r - r_{0})^2

    V_{angles} = k_{\theta} (\theta - \theta_{0})^2

    V_{Urey-Bradley} = k_{\hat{r}} (|hat{r} - \hat{r}_{0})^2

    V_{dihedrals} = k_{\phi} [1 + \cos(n \phi - \delta)]

    V_{impropers} = k_{\omega} (\omega - \omega_{0})^2

"""

from typing import Union, Tuple, Optional
from itertools import permutations

import numpy as np
import pandas as pd

from scm.plams import Units

from .parse_wildcards import parse_wildcards
from ..classes.multi_mol import MultiMolecule
from ..io.read_psf import PSFContainer
from ..io.read_prm import PRMContainer

__all__ = ['get_bonded']


def get_bonded(mol: Union[str, MultiMolecule], psf: Union[str, PSFContainer],
               prm: Union[str, PRMContainer]) -> Tuple[Optional[pd.DataFrame], ...]:
    r"""Collect forcefield parameters and calculate all intra-ligand interactions in **mol**.

    Forcefield parameters are collected from the provided **psf** and **prm** files.

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
    5x :class:`pandas.Series` and/or ``None``
        Four series with the potential energies of all bonds, angles, Urey-Bradley terms,
        proper and improper dihedral angles.
        A Series is replaced with ``None`` if no parameters are available for that particular
        section.
        Units are in atomic units.

    """
    # Read the .psf file and switch from 1- to 0-based atomic indices
    if not isinstance(psf, PSFContainer):
        psf = PSFContainer.read(psf)
    else:
        psf = psf.copy()
    psf.bonds -= 1
    psf.angles -= 1
    psf.dihedrals -= 1
    psf.impropers -= 1

    # Read the molecule
    if not isinstance(mol, MultiMolecule):
        mol = MultiMolecule.from_xyz(mol)
    else:
        mol = mol.copy(deep=False)
    mol.atoms = psf.to_atom_dict()
    symbols = sorted(mol.atoms.keys())

    # Extract parameters from the .prm file
    bonds, angles, urey_bradley, dihedrals, impropers = process_prm(prm)

    # Calculate the various potential energies
    if bonds is not None:
        parse_wildcards(bonds, symbols, prm_type='bonds')
        set_V_bonds(bonds, mol, psf.bonds)
        bonds = bonds['V'] * Units.conversion_ratio('kcal/mol', 'au')

    if angles is not None:
        parse_wildcards(bonds, symbols, prm_type='angles')
        set_V_angles(angles, mol, psf.angles)
        angles = angles['V'] * Units.conversion_ratio('kcal/mol', 'au')

    if urey_bradley is not None:
        parse_wildcards(bonds, symbols, prm_type='urey_bradley')
        set_V_UB(urey_bradley, mol, psf.angles)
        urey_bradley = urey_bradley['V'] * Units.conversion_ratio('kcal/mol', 'au')

    if dihedrals is not None:
        parse_wildcards(bonds, symbols, prm_type='dihedrals')
        set_V_dihedrals(dihedrals, mol, psf.dihedrals)
        dihedrals = dihedrals['V'] * Units.conversion_ratio('kcal/mol', 'au')

    if impropers is not None:
        parse_wildcards(bonds, symbols, prm_type='impropers')
        set_V_impropers(impropers, mol, psf.impropers)
        impropers = impropers['V'] * Units.conversion_ratio('kcal/mol', 'au')

    return bonds, angles, urey_bradley, dihedrals, impropers


def process_prm(prm: Union[PRMContainer, str]) -> tuple:
    """Extract all bond, angle, dihedral and improper parameters from **prm**."""
    if not isinstance(prm, PRMContainer):
        prm = PRMContainer.read(prm)
    else:
        prm = prm.copy()

    bonds = prm.bonds
    if bonds is not None:
        bonds = bonds[[2, 3]].copy()
        bonds['V'] = np.nan

    angles = prm.angles
    if angles is not None:
        urey_bradley = angles[[5, 6]].copy()
        urey_bradley.index = urey_bradley.index.droplevel(1)
        is_null = urey_bradley.isnull()
        if is_null.values.all():
            urey_bradley = None
        else:
            urey_bradley[is_null] = 0.0
            urey_bradley['V'] = np.nan

        angles = angles[[3, 4]].copy()
        angles[4] *= np.radians(1)
        angles['V'] = np.nan

    dihedrals = prm.dihedrals
    if dihedrals is not None:
        dihedrals = dihedrals[[4, 5, 6]].copy()
        dihedrals[6] *= np.radians(1)
        dihedrals['V'] = np.nan

    impropers = prm.impropers
    if impropers is not None:
        impropers = impropers[[4, 6]].copy()
        impropers[6] *= np.radians(1)
        impropers['V'] = np.nan

    return bonds, angles, urey_bradley, dihedrals, impropers


def set_V_bonds(df: pd.DataFrame, mol: MultiMolecule, bond_idx: np.ndarray) -> None:
    """Calculate and set :math:`V_{bonds}` in **df**.

    Parameters
    ----------
    df : :class:`pd.DataFrame`
        A DataFrame with atom pairs and parameters.

    mol : :class:`MultiMolecule`
        A MultiMolecule instance.

    bond_idx : :math:`i*2` :class:`numpy.ndarray`
         A 2D numpy array with all atom-pairs defining bonds.

    """
    symbol = mol.symbol
    distance = _dist(mol, bond_idx)

    iterator = df.iloc[:, 0:2].iterrows()
    for i, item in iterator:
        j = np.all(symbol[bond_idx] == i, axis=1)
        j |= np.all(symbol[bond_idx[:, ::-1]] == i, axis=1)  # Consider all valid permutations
        df.at[i, 'V'] = get_V_harmonic(distance[:, j], *item).sum()


def set_V_angles(df: pd.DataFrame, mol: MultiMolecule, angle_idx: np.ndarray) -> None:
    """Calculate and set :math:`V_{angles}` in **df**.

    Parameters
    ----------
    df : :class:`pd.DataFrame`
        A DataFrame with atom pairs and parameters.

    mol : :class:`MultiMolecule`
        A MultiMolecule instance.

    bond_idx : :math:`i*3` :class:`numpy.ndarray`
         A 2D numpy array with all atom-pairs defining bonds.

    """
    symbol = mol.symbol
    angle = _angle(mol, angle_idx)

    iterator = df.iloc[:, 0:2].iterrows()
    for i, item in iterator:
        j = np.all(symbol[angle_idx] == i, axis=1)
        j |= np.all(symbol[angle_idx[:, ::-1]] == i, axis=1)  # Consider all valid permutations
        df.at[i, 'V'] = get_V_harmonic(angle[:, j], *item).sum()


def set_V_UB(df: pd.DataFrame, mol: MultiMolecule, angle_idx: np.ndarray) -> None:
    """Calculate and set :math:`V_{Urey-Bradley}` in **df**.

    Parameters
    ----------
    df : :class:`pd.DataFrame`
        A DataFrame with atom pairs and parameters.

    mol : :class:`MultiMolecule`
        A MultiMolecule instance.

    angle_idx : :math:`(i,3)` :class:`numpy.ndarray`
         A 2D numpy array with all atom-pairs defining angles.

    """
    symbol = mol.symbol
    bond_idx = angle_idx[:, 0::2]
    distance = _dist(mol, bond_idx)

    iterator = df.iloc[:, 0:2].iterrows()
    for i, item in iterator:
        j = np.all(symbol[bond_idx] == i, axis=1)
        j |= np.all(symbol[bond_idx[:, ::-1]] == i, axis=1)  # Consider all valid permutations
        df.at[i, 'V'] = get_V_harmonic(distance[:, j], *item).sum()


def set_V_dihedrals(df: pd.DataFrame, mol: MultiMolecule, dihed_idx: np.ndarray) -> None:
    """Calculate and set :math:`V_{dihedrals}` in **df**.

    Parameters
    ----------
    df : :class:`pd.DataFrame`
        A DataFrame with atom pairs and parameters.

    mol : :class:`MultiMolecule`
        A MultiMolecule instance.

    bond_idx : :math:`i*4` :class:`numpy.ndarray`
         A numpy array with all atom-pairs defining proper dihedral angles.

    """
    symbol = mol.symbol
    dihedral = _dihed(mol, dihed_idx)

    iterator = df.iloc[:, 0:3].iterrows()
    for i, item in iterator:
        j = np.all(symbol[dihed_idx] == i, axis=1)
        j |= np.all(symbol[dihed_idx[:, ::-1]] == i, axis=1)  # Consider all valid permutations
        df.at[i, 'V'] = get_V_cos(dihedral[:, j], *item).sum()


def set_V_impropers(df: pd.DataFrame, mol: MultiMolecule, improp_idx: np.ndarray) -> None:
    """Calculate and set :math:`V_{impropers}` in **df**.

    Parameters
    ----------
    df : :class:`pd.DataFrame`
        A DataFrame with atom pairs and parameters.

    mol : :class:`MultiMolecule`
        A MultiMolecule instance.

    bond_idx : :math:`i*2` :class:`numpy.ndarray`
         A numpy array with all atom-pairs defining improper dihedral angles.

    """
    symbol = mol.symbol
    improper = _dihed(mol, improp_idx)

    iterator = df.iloc[:, 0:2].iterrows()
    for i, item in iterator:
        j = np.zeros(len(improp_idx), dtype=bool)
        for k in permutations([1, 2, 3], r=3):
            k = (0,) + k
            j |= np.all(symbol[improp_idx[:, k]] == i, axis=1)  # Consider all valid permutations
        df.at[i, 'V'] = get_V_harmonic(improper[:, j], *item).sum()


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


def get_V_harmonic(x: np.ndarray, k: float, x0: float) -> float:
    r"""Calculate the harmonic potential energy: :math:`\sum_{i} k (x_{i} - x_{0})^2`.

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        An array of geometry parameters such as distances, angles or improper dihedral angles.
        Units should be in

    x_ref : :class:`float`
        The equilibrium value of **x**; units should be in Angstroem or radian.

    k : :class:`float`
        The force constant :math:`k`; units should be in kcal/mol/Angstroem**2 or kcal/mol/rad**2.

    """
    V = k * (x - x0)**2
    return V.mean(axis=0)


def get_V_cos(phi: np.ndarray, k: float, n: int, delta: float = 0.0) -> float:
    r"""Calculate the cosine potential energy: :math:`\sum_{i} k_{\phi} [1 + \cos(n \phi_{i} - \delta)]`.

    Parameters
    ----------
    phi : :class:`numpy.ndarray`
        An array of dihedral angles; units should be in radian.

    k : :class:`float`
        The force constant :math:`k`; units should be in kcal/mol.

    n : :class:`int`
        The multiplicity :math:`n`.

    delta : :class:`float`
        The phase-correction :math:`\delta`; units should be in radian.

    """  # noqa
    V = k * (1 + np.cos(n * phi - delta))
    return V.mean(axis=0)
