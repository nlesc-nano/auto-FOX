"""
FOX.functions.lj_calculate
==========================

A module for calculating non-bonded interactions using Coulomb + Lennard-Jones potentials.

"""

from types import MappingProxyType
from itertools import combinations_with_replacement
from typing import Mapping, Tuple, Sequence, Optional, Iterable, Union, Dict, List

import numpy as np
import pandas as pd

from scm.plams import Units, Settings

from .utils import read_rtf_file, fill_diagonal_blocks, group_by_values
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
                   cp2k_settings: Optional[Mapping] = None) -> pd.DataFrame:
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

    prm_df = df = construct_prm_df(mol_atoms)
    overlay_psf(df, psf)
    if prm is not None:
        overlay_prm(df, prm)
    if cp2k_settings is not None:
        overlay_cp2k_settings(df, cp2k_settings)

    slice_dict = {(i, j): (mol_atoms[i], mol_atoms[j]) for i, j in prm_df.index}
    core_atoms = set(psf.atom_type[psf.residue_id == 1])
    ligand_count = psf.residue_id.max() - 1
    return get_V(mol, slice_dict, df.loc, ligand_count, core_atoms=core_atoms)


def get_V(mol: MultiMolecule, slice_mapping: SliceMapping,
          prm_mapping: PrmMapping, ligand_count: int,
          core_atoms: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Calculate all non-covalent interactions averaged over all molecules in **mol**.

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
        dist = mol.get_dist_mat(atom_subset=ij)
        if not core_atoms.intersection(atoms):
            i = len(ij[0]) // ligand_count
            j = len(ij[1]) // ligand_count
            fill_diagonal_blocks(dist, i, j)  # Set intra-ligand interactions to np.nan
        else:
            dist[dist == 0.0] = np.nan

        prm = prm_mapping[atoms]
        df.at[atoms, 'elstat'] = get_V_elstat(prm['charge'], dist)
        df.at[atoms, 'lj'] = get_V_lj(prm['sigma'], prm['epsilon'], dist)

        if atoms[0] == atoms[1]:  # Avoid double-counting
            df.loc[atoms] /= 2

    df /= len(mol)
    return df


def get_V_elstat(qq: float, dist: np.ndarray) -> float:
    r"""Calculate and sum the electrostatic potential energy.

    .. math::

        V_{elstat}(r) = \frac{q_{ij}}{\epsilon r_{ij}}

        q_{ij} = q_{i} * q_{j}

        \epsilon = 1

    Parameters
    ----------
    qq : :class:`float`
        The product of two charges :math:`q_{i} q_{j}`
    dist : :class:`numpy.ndarray`
        An array with all distances :math:`r_{ij}`.
        Units should be in Bohr.

    Returns
    -------
    :class:`float`
        The elctrostatic potential energy summed over all distance pairs in **dist**.

    """
    return np.nansum(qq / dist)


def get_V_lj(sigma: float, epsilon: float, dist: np.ndarray) -> float:
    r"""Calculate and sum the Lennard-Jones potential energy.

    .. math::

        V_{LJ}(r) = 4 \varepsilon_{ij}
        \left(
            \left(
                \frac{\sigma_{ij}}{r_{ij}}
            \right )^{12} -
            \left(
                \frac{\sigma_{ij}}{r_{ij}}
            \right )^6
        \right )

        \sigma_{ij} = \frac{\sigma_{i} * \sigma_{j}}{2}

        \varepsilon_{ij} = \sqrt{\varepsilon_{i} * \varepsilon_{j}}

    Parameters
    ----------
    sigma : :class:`float`
        The parameter :math:`\sigma`.
    epsilon : :class:`float`
        The parameter :math:`\sigma`.
    dist : :class:`numpy.ndarray`
        An array with all distances :math:`r_{ij}`.
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


def construct_prm_df(atoms: Iterable[str]) -> pd.DataFrame:
    """Construct an empty parameter DataFrame for all **atoms** combinations of length 2."""
    return pd.DataFrame(
        0.0,
        index=pd.MultiIndex.from_tuples(combinations_with_replacement(sorted(atoms.keys()), 2)),
        columns=pd.Index(['charge', 'epsilon', 'sigma'], name='au')
    )


def set_charge(df: pd.DataFrame, charge_mapping: Mapping[str, float]) -> None:
    """Set :math:`q_{i} * q_{j}`."""
    atom_pairs = combinations_with_replacement(sorted(charge_mapping.keys()), 2)
    for i, j in atom_pairs:
        charge = charge_mapping[i] * charge_mapping[j]
        df.at[(i, j), 'charge'] = charge


def set_epsilon(df: pd.DataFrame, epsilon_mapping: Mapping[str, float],
                unit: str = 'kj/mol') -> None:
    r"""Set :math:`\sqrt{\varepsilon_{i} * \varepsilon_{j}}`."""
    atom_pairs = combinations_with_replacement(sorted(epsilon_mapping.keys()), 2)
    for i, j in atom_pairs:
        epsilon = (epsilon_mapping[i] * epsilon_mapping[j])**0.5
        epsilon *= Units.conversion_ratio(unit, 'au')
        df.at[(i, j), 'epsilon'] = epsilon


def set_sigma(df: pd.DataFrame, sigma_mapping: Mapping[str, float],
              unit: str = 'nm') -> None:
    r"""Set :math:`\frac{ \sigma_{i} * \sigma_{j} }{2}`."""
    unit2au = Units.conversion_ratio(unit, 'au')
    atom_pairs = combinations_with_replacement(sorted(sigma_mapping.keys()), 2)
    for i, j in atom_pairs:
        sigma = (sigma_mapping[i] + sigma_mapping[j]) / 2
        sigma *= unit2au
        df.at[(i, j), 'sigma'] = sigma


def set_charge_pairs(df: pd.DataFrame, charge_mapping: Mapping[Tuple[str, str], float]) -> None:
    """Set :math:`q_{ij}`."""
    for _ij, charge in charge_mapping.items():
        ij = tuple(sorted(_ij))
        df.at[ij, 'charge'] = charge


def set_epsilon_pairs(df: pd.DataFrame, epsilon_mapping: Mapping[Tuple[str, str], float],
                      unit: str = 'kj/mol') -> None:
    r"""Set :math:`\varepsilon_{ij}`."""
    unit2au = Units.conversion_ratio(unit, 'au')
    for _ij, epsilon in epsilon_mapping.items():
        ij = tuple(sorted(_ij))
        epsilon *= unit2au
        df.at[ij, 'epsilon'] = epsilon


def set_sigma_pairs(df: pd.DataFrame, sigma_mapping: Mapping[Tuple[str, str], float],
                    unit: str = 'nm') -> None:
    r"""Set :math:`\sigma_{ij}`."""
    unit2au = Units.conversion_ratio(unit, 'au')
    for _ij, sigma in sigma_mapping.items():
        ij = tuple(sorted(_ij))
        sigma *= unit2au
        df.at[ij, 'sigma'] = sigma


#: Map CP2K units to PLAMS units (see :class:`scm.plams.Units`).
UNIT_MAPPING: Mapping[str, str] = MappingProxyType({'kcalmol': 'kcal/mol', 'kjmol': 'kj/mol'})


def overlay_cp2k_settings(df: pd.DataFrame, cp2k_settings: Mapping) -> None:
    r"""Overlay **df** with all :math:`q`, :math:`\sigma` and :math:`\varepsilon` values from **cp2k_settings**."""  # noqa
    charge = cp2k_settings['input']['force_eval']['mm']['forcefield']['charge']
    charge_dict = {block['atom']: block['charge'] for block in charge}

    with Settings.supress_missing():
        lj = cp2k_settings['input']['force_eval']['mm']['forcefield']['nonbonded']['lennard-jones']
        epsilon_s = Settings()
        sigma_s = Settings()
        for block in lj:
            atoms = tuple(block['atoms'].split())

            try:
                unit_sigma, sigma = block['sigma'].split()
            except ValueError:
                unit_sigma, sigma = '[angstrom]', block['sigma']
            unit_sigma = unit_sigma[1:-1]
            unit_sigma = UNIT_MAPPING.get(unit_sigma, unit_sigma)
            sigma = float(sigma)

            try:
                unit_eps, epsilon = block['epsilon'].split()
            except ValueError:
                unit_eps, epsilon = '[kcalmol]', block['sigma']
            unit_eps = unit_eps[1:-1]
            unit_eps = UNIT_MAPPING.get(unit_eps, unit_eps)
            epsilon = float(epsilon)

            sigma_s[unit_sigma][atoms] = sigma
            epsilon_s[unit_eps][atoms] = epsilon

    set_charge(df, charge_dict)
    for unit, dct in epsilon_s.items():
        set_epsilon_pairs(df, dct, unit=unit)
    for unit, dct in sigma_s.items():
        set_sigma_pairs(df, dct, unit=unit)


def overlay_prm(df: pd.DataFrame, prm: Union[str, PRMContainer]) -> None:
    r"""Overlay **df** with all :math:`\sigma` and :math:`\varepsilon` values from **prm**."""
    if not isinstance(prm, PRMContainer):
        prm = PRMContainer.read(prm)

    nonbonded = prm.nonbonded.set_index(0)
    epsilon = nonbonded[2].astype(float, copy=False)
    sigma = nonbonded[3].astype(float, copy=False)
    set_epsilon(df, epsilon, unit='kcal/mol')
    set_sigma(df, sigma, unit='angstrom')


def overlay_rtf(df: pd.DataFrame, rtf: str) -> None:
    r"""Overlay **df** with all :math:`q` values from **rtf**."""
    charge_dict = dict(zip(*read_rtf_file(rtf)))
    set_charge(df, charge_dict)


def overlay_psf(df: pd.DataFrame, psf: Union[str, PRMContainer]) -> None:
    r"""Overlay **df** with all :math:`q` values from **psf**."""
    if not isinstance(psf, PSFContainer):
        psf = PSFContainer.read(psf)

    charge = psf.atoms.set_index('atom type')['charge']
    charge_dict = charge.to_dict()
    set_charge(df, charge_dict)


def psf_to_atom_dict(psf: Union[str, PSFContainer]) -> Dict[str, List[int]]:
    """Create a new dictionary of atoms and their respective indices.

    Parameters
    ----------
    psf : :class:`str` or :class:`PSFContainer`
        A PSFContainer instance or a file-like object representing a .psf file.

    Returns
    -------
    :class:`dict` [:class:`str`, :class:`list` [:class:`int`]]
        A dictionary with atom types as keys and lists of matching atomic indices as values.
        The indices are 0-based.

    """
    if not isinstance(psf, PSFContainer):
        psf = PSFContainer.read(psf)

    try:
        iterator = enumerate(psf.atom_type)
    except AttributeError as ex:
        err = "The 'psf' parameter is of invalid type: '{psf.__class__.__name__}'"
        raise TypeError(err).with_traceback(ex.__traceback__)
    return group_by_values(iterator)


"""
# Create the molecule
psf = '/Users/bvanbeek/Documents/CdSe/Week_5/qd/Cd68Cl26Se55__26_C#CCCC[=O][O-]@O7.psf'
prm = '/Users/bvanbeek/Documents/CdSe/Week_5/ligand/forcefield/ff_assignment/ff_assignment.prm'
cp2k_dict = {'input': cp2kparser.read_input('/Users/bvanbeek/Documents/CdSe/Week_5/qd/qd_opt/QD_opt/QD_opt.in')}  # noqa
mol = '/Users/bvanbeek/Documents/CdSe/Week_5/qd/qd_opt/QD_opt/cp2k-pos-1.xyz'

V_df = get_nonbonded(mol, psf, prm, cp2k_dict)
V_df.loc[('Cd', 'Cd')] = V_df.loc[('Se', 'Se')] = V_df.loc[('Cd', 'Se')] = 0.0
V_df *= Units.conversion_ratio('au', 'kcal/mol')
V_df /= 26
"""
