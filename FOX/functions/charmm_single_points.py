from itertools import combinations_with_replacement

import numpy as np
import pandas as pd

from scm.plams import Units
from FOX import MultiMolecule, get_example_xyz


def fill_diagonal_blocks(ar: np.ndarray, i: int, j: int, fill_value: float = np.nan) -> None:
    """Fill diagonal blocks of size :math:`i * j`."""
    i0 = j0 = 0
    len_ar = ar.shape[1]
    while len_ar > i0:
        ar[:, i0:i0+i, j0:j0+j] = fill_value
        i0 += i
        j0 += j


def V_elstat(qq: float, dist: np.ndarray) -> float:
    r"""Calculate the electrostatic potential energy :math:`q_{i} q_{j} / \epsilon r_{ij}`.

    Parameters
    ----------
    qq : :class:`float`
        The product of two charges :math:`q_{i} q_{j}`
    dist : :class:`numpy.ndarray`
        An array with all distances :math:`r_{ij}`.
        Units should be in Bohr.

    """
    return np.nansum(qq / dist)


def V_lj(sigma: float, epsilon: float, dist: np.ndarray) -> float:
    r"""Calculate the Lennard-Jones potential energy.

    Parameters
    ----------
    sigma : :class:`float`
        The parameter :math:`\sigma`.
    epsilon : :class:`float`
        The parameter :math:`\sigma`.
    dist : :class:`numpy.ndarray`
        An array with all distances :math:`r_{ij}`.
        Units should be in Bohr.

    """
    sigma_dist = sigma / dist
    lj = sigma_dist**12
    lj -= sigma_dist**6
    lj *= epsilon * 4
    return np.nansum(lj)


charges = {'Cd': 0.976800, 'Se': -0.976800, 'O': -0.470400, 'H': 0.0, 'C': 0.452400}
epsilon = {'H': -0.0460, 'C': -0.0700, 'O': -0.1200}  # kcal/mol
epsilon = {k: v * 4.184 for k, v in epsilon.items()}  # kj/mol
sigma = {'H': 0.9000, 'C': 2.0000, 'O': 1.7000}  # Ånstroms
sigma = {k: v / 10 for k, v in sigma.items()}  # nm

df = pd.DataFrame(
    np.zeros((15, 3), dtype=float),
    index=pd.MultiIndex.from_tuples(combinations_with_replacement(sorted(charges), 2)),
    columns=['charge', 'sigma', 'epsilon']
)

for i, j in df.index:
    df.at[(i, j), 'charge'] = charges[i] * charges[j]

for i, j in df.index:
    try:
        df.at[(i, j), 'epsilon'] = np.sqrt(epsilon[i] * epsilon[j])
        df.at[(i, j), 'sigma'] = (sigma[i] + sigma[j]) / 2
    except KeyError:
        pass


# kj/mol
df.at[('Cd', 'Cd'), 'epsilon'] = 0.3101
df.at[('Se', 'Se'), 'epsilon'] = 0.4266
df.at[('Cd', 'Se'), 'epsilon'] = 1.5225
df.at[('Cd', 'O'), 'epsilon'] = 1.8340
df.at[('O', 'Se'), 'epsilon'] = 1.6135

# nm
df.at[('Cd', 'Cd'), 'sigma'] = 0.1234
df.at[('Se', 'Se'), 'sigma'] = 0.4852
df.at[('Cd', 'Se'), 'sigma'] = 0.2940
df.at[('Cd', 'O'), 'sigma'] = 0.2471
df.at[('O', 'Se'), 'sigma'] = 0.3526

df['epsilon'] *= Units.conversion_ratio('kJ/mol', 'au')
df['sigma'] *= Units.conversion_ratio('nm', 'au')

mol = MultiMolecule.from_xyz('/Users/basvanbeek/Documents/GitHub/auto-FOX/FOX/data/Cd68Se55_26COO_MD_trajec.xyz')
mol = MultiMolecule.from_xyz('/Users/basvanbeek/Downloads/MM_MD_workdir.003/cp2kjob.002/PROJECT-pos-1.xyz')
mol.guess_bonds(atom_subset=['C', 'O', 'H'])
mol *= Units.conversion_ratio('Angstrom', 'au')
lig_count = len(mol.residue_argsort(concatenate=False)[1:])

slice_dict = {(i, j): (mol.atoms[i], mol.atoms[j]) for i, j in df.index}

df_new = pd.DataFrame(np.zeros((15, 2), dtype=float), index=df.index, columns=['charge', 'lj'])

core_at = {'Cd', 'Se'}
for atoms, ij in slice_dict.items():
    dist = mol.get_dist_mat(atom_subset=ij)
    if not core_at.intersection(atoms):
        i = len(ij[0]) // lig_count
        j = len(ij[1]) // lig_count
        if i == j == 1:
            np.fill_diagonal(dist, np.nan)
        else:
            fill_diagonal_blocks(dist, i, j)
    else:
        dist[dist == 0.0] = np.nan

    charge, sigma, eps = df.loc[atoms]
    df_new.at[atoms, 'charge'] = V_elstat(charge, dist)
    df_new.at[atoms, 'lj'] = V_lj(sigma, eps, dist)

    if atoms[0] == atoms[1]:  # Avoid double-counting
        df_new.loc[atoms] /= 2

df_new /= len(mol)
df_new *= Units.conversion_ratio('au', 'kcal/mol')
