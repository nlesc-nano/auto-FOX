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
from typing import Set, Generator, List, Union, Callable
from itertools import chain

import numpy as np

from scm.plams import Atom, Molecule, Units

from .lj_calculate import get_V_elstat, get_V_lj
from .lj_dataframe import LJDataFrame
from .bonded_calculate import _dist
from ..classes.multi_mol import MultiMolecule
from ..io.read_psf import PSFContainer
from ..io.read_prm import PRMContainer

__all__ = ['get_intra_non_bonded']


def get_intra_non_bonded(mol: Union[str, MultiMolecule], psf: Union[str, PSFContainer],
                         prm: Union[str, PRMContainer]) -> LJDataFrame:
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

    Returns
    -------
    :class:`pandas.DataFrame`
        A DataFrame with the electrostatic and Lennard-Jones components of the
        (inter-ligand) potential energy per atom-pair.
        The potential energy is summed over atoms with matching atom types and
        averaged over all molecules within **mol**.
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

    # Construct the parameter DataFrame
    mol.atoms = psf.to_atom_dict()
    prm_df = _construct_df(mol, lig_atoms, psf, prm)

    import pdb; pdb.set_trace()

    ij = _get_idx(mol, core_atoms).T
    if not ij.any():
        return prm_df[['elstat', 'lj']]

    # Map each atom-index pair (ij) to a pair of atomic symbols (more specifically: their hashes)
    mol.atoms = {hash(k): v for k, v in mol.atoms.items()}
    symbol = mol.symbol[ij].T
    symbol.sort(axis=1)

    # Construct the distance array
    dist = _dist(mol, ij.T)
    dist *= Units.conversion_ratio('angstrom', 'au')

    # Calculate the potential energies
    for idx, items in prm_df[['charge', 'epsilon', 'sigma']].iterrows():
        idx_hash = sorted(hash(i) for i in idx)
        dist_slice = dist[:, np.all(symbol == idx_hash, axis=1)]

        charge, epsilon, sigma = items
        prm_df.at[idx, 'elstat'] = get_V_elstat(items['charge'], dist_slice) / len(mol)
        prm_df.at[idx, 'lj'] = get_V_lj(sigma, epsilon, dist_slice) / len(mol)

        # Prevent double counting when an atom-pair consists of identical atoms
        if idx[0] == idx[1]:
            prm_df.loc[idx, ['elstat', 'lj']] /= 2

    return prm_df[['elstat', 'lj']]


def _construct_df(mol, lig_atoms, psf, prm) -> LJDataFrame:
    """Construct the DataFrame for :func:`get_intra_non_bonded`."""
    prm_df = LJDataFrame(index=set(mol.symbol[lig_atoms]))
    prm_df.overlay_psf(psf)
    prm_df.overlay_prm(prm)
    prm_df['elstat'] = 0.0
    prm_df['lj'] = 0.0
    prm_df.columns.name = 'au'
    prm_df.dropna(inplace=True)
    return prm_df


def _get_idx(mol: MultiMolecule, core_atoms: np.ndarray,
             depth_comparison: Callable[[int, int], bool] = operator.__ge__) -> np.ndarray:
    def dfs(at1: Atom, id_list: list, i: int, exclude: Set[Atom], depth: int = 0):
        exclude.add(at1)
        for bond in at1.bonds:
            at2 = bond.other_end(at1)
            if at2 in exclude:
                continue
            elif depth_comparison(depth, 3):
                id_list += [i, at2.id]
            dfs(at2, id_list, i, exclude, depth=1+depth)

    def gather_idx(molecule: Molecule) -> Generator[List[int], None, None]:
        for i, at in enumerate(molecule):
            id_list = []
            dfs(at, id_list, i, set())
            yield id_list

    if not core_atoms.any():
        return np.zeros((0, 2), dtype=int)

    _mol = mol.delete_atoms(core_atoms)
    _mol.bonds -= len(core_atoms)
    molecule = _mol.as_Molecule(0)[0]
    molecule.set_atoms_id(start=0)

    idx = np.fromiter(chain.from_iterable(gather_idx(molecule)), dtype=int)
    idx += len(mol._get_atom_subset(core_atoms, as_array=True))
    idx.shape = -1, 2
    return idx
