"""
FOX.functions.lj_calculate
==========================

A module for calculating non-bonded interactions using Coulomb + Lennard-Jones potentials.

"""

from itertools import combinations_with_replacement
from typing import Mapping, Tuple, Sequence, Optional, Iterable, Union, Dict, List

import numpy as np
import pandas as pd

from scm.plams import Units
from FOX import MultiMolecule, get_example_xyz
from FOX.functions.utils import group_by_values
from FOX.io.read_psf import PSFContainer


def fill_diagonal_blocks(ar: np.ndarray, i: int, j: int, fill_value: float = np.nan) -> None:
    """Fill diagonal blocks of size :math:`i * j`."""
    i0 = j0 = 0
    len_ar = ar.shape[1]
    while len_ar > i0:
        ar[:, i0:i0+i, j0:j0+j] = fill_value
        i0 += i
        j0 += j


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


SliceMapping = Mapping[Tuple[str, str], Tuple[Sequence[int], Sequence[int]]]
PrmMapping = Mapping[Tuple[str, str], Tuple[float, float, float]]


def get_V(mol: MultiMolecule, slice_mapping: SliceMapping,
          prm_mapping: PrmMapping,
          core_atoms: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Calculate all non-covalent interactions averaged over all molecules in **mol**."""
    core_atoms = set(core_atoms)

    mol.guess_bonds(atom_subset=[at for at in mol.atoms if at not in core_atoms])
    ligand_count = len(mol.residue_argsort(concatenate=False)[1:])
    mol *= Units.conversion_ratio('Angstrom', 'au')

    df = pd.DataFrame(
        0.0,
        index=pd.MultiIndex.from_tuples(sorted(slice_mapping)),
        columns=['elstat', 'lj']
    )

    for atoms, ij in slice_mapping.items():
        dist = mol.get_dist_mat(atom_subset=ij)
        if not core_atoms.intersection(atoms):
            i = len(ij[0]) // ligand_count
            j = len(ij[1]) // ligand_count
            fill_diagonal_blocks(dist, i, j)  # Set intra-ligand interactions to np.nan
        else:
            dist[dist == 0.0] = np.nan

        charge, sigma, eps = prm_mapping[atoms]
        df.at[atoms, 'elstat'] = get_V_elstat(charge, dist)
        df.at[atoms, 'lj'] = get_V_lj(sigma, eps, dist)

        if atoms[0] == atoms[1]:  # Avoid double-counting
            df.loc[atoms] /= 2

    df /= len(mol)
    return df


def construct_prm_df(atoms: Iterable[str]) -> pd.DataFrame:
    """Construct an empty parameter DataFrame for all **atoms** combinations of length 2."""
    return pd.DataFrame(
        0.0,
        index=pd.MultiIndex.from_tuples(combinations_with_replacement(sorted(atoms), 2)),
        columns=['charge', 'sigma', 'epsilon']
    )


def set_charges(df: pd.DataFrame, charge_mapping: Mapping[str, float]) -> None:
    """Set :math:`q_{i} * q_{j}`."""
    for ij in combinations_with_replacement(charge_mapping, 2):
        i, j = sorted(ij)
        charge = charge_mapping[i] * charge_mapping[j]
        df.at[(i, j), 'charge'] = charge


def set_epsilon(df: pd.DataFrame, epsilon_mapping: Mapping[str, float],
                unit: str = 'kj/mol') -> None:
    r"""Set :math:`\sqrt{\varepsilon_{i} * \varepsilon_{j}}`."""
    atom_pairs = list(combinations_with_replacement(sorted(epsilon_mapping), 2))
    for ij in atom_pairs:
        i, j = ij
        epsilon = epsilon_mapping[i] * epsilon_mapping[j]
        df.at[(i, j), 'epsilon'] = epsilon

    df.loc[atom_pairs, 'epsilon'] **= 0.5
    df.loc[atom_pairs, 'epsilon'] *= Units.conversion_ratio(unit, 'au')


def set_sigma(df: pd.DataFrame, sigma_mapping: Mapping[str, float],
              unit: str = 'nm') -> None:
    r"""Set :math:`\frac{ \sigma_{i} * \sigma_{j} }{2}`."""
    atom_pairs = list(combinations_with_replacement(sorted(sigma_mapping), 2))
    for ij in atom_pairs:
        i, j = ij
        sigma = sigma_mapping[i] + sigma_mapping[j]
        df.at[(i, j), 'sigma'] = sigma

    df.loc[atom_pairs, 'sigma'] /= 2
    df.loc[atom_pairs, 'sigma'] *= Units.conversion_ratio(unit, 'au')


def set_epsilon_pairs(df: pd.DataFrame, epsilon_mapping: Mapping[Tuple[str, str], float],
                      unit: str = 'kj/mol') -> None:
    r"""Set :math:`\varepsilon_{ij}`."""
    for ij, epsilon in epsilon_mapping.items():
        i, j = ij
        df.at[(i, j), 'epsilon'] = epsilon

    atom_pairs = list(epsilon_mapping.keys())
    df.loc[atom_pairs, 'epsilon'] *= Units.conversion_ratio(unit, 'au')


def set_sigma_pairs(df: pd.DataFrame, sigma_mapping: Mapping[Tuple[str, str], float],
                    unit: str = 'nm') -> None:
    r"""Set :math:`\sigma_{ij}`."""
    for ij, sigma in sigma_mapping.items():
        i, j = ij
        df.at[(i, j), 'sigma'] = sigma

    atom_pairs = list(sigma_mapping.keys())
    df.loc[atom_pairs, 'sigma'] *= Units.conversion_ratio(unit, 'au')


def get_atom_dict(psf: Union[str, PSFContainer]) -> Dict[str, List[int]]:
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
        raise TypeError("Invalid type: '{psf.__class__.__name__}'").with_traceback(ex.__traceback__)
    return group_by_values(iterator)


charges = {'Cd': 0.976800, 'Se': -0.976800, 'O': -0.470400, 'H': 0.0, 'C': 0.452400}
epsilon = {'H': -0.0460, 'C': -0.0700, 'O': -0.1200}  # kcal/mol
sigma = {'H': 0.9000, 'C': 2.0000, 'O': 1.7000}  # Ånstroms

epsilon_pairs = {('Cd', 'Cd'): 0.3101,  # kj/mol
                 ('Se', 'Se'): 0.4266,
                 ('Cd', 'Se'): 1.5225,
                 ('Cd', 'O'): 1.8340,
                 ('O', 'Se'): 1.6135}

sigma_pairs = {('Cd', 'Cd'): 0.1234,  # nm
               ('Se', 'Se'): 0.4852,
               ('Cd', 'Se'): 0.2940,
               ('Cd', 'O'): 0.2471,
               ('O', 'Se'): 0.3526}

# Create and fill a DataFrame of all (pair-wise) parameters
df = construct_prm_df(charges)
set_charges(df, charges)
set_epsilon(df, epsilon, unit='kcal/mol')
set_epsilon_pairs(df, epsilon_pairs)
set_sigma(df, sigma, unit='angstrom')
set_sigma_pairs(df, sigma_pairs)

# Create the molecule
mol = MultiMolecule.from_xyz(get_example_xyz())
mol_atoms = mol.atoms
slice_dict = {(i, j): (mol_atoms[i], mol_atoms[j]) for i, j in df.index}

# Get all non-bonded interactions
core_atoms = {'Cd', 'Se'}
df_new = get_V(mol, slice_dict, df.loc, core_atoms=core_atoms)
