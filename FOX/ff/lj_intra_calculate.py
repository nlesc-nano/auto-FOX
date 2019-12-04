r"""
FOX.ff.lj_intra_calculate
=========================

A module for calculating non-bonded intra-ligand interactions using Coulomb + Lennard-Jones potentials.

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

from typing import Set, Generator, List, Union, Iterable
from itertools import chain

import numpy as np

from scm.plams import Atom, Molecule

from FOX import get_example_xyz
from FOX.classes.multi_mol import MultiMolecule
from FOX.io.read_psf import PSFContainer
from FOX.io.read_prm import PRMContainer
from FOX.ff.lj_calculate import psf_to_atom_dict, LJDataFrame, get_V_elstat, get_V_lj
from FOX.ff.bonded_calculate import _dist


def start_workflow(mol: Union[str, MultiMolecule], psf: Union[str, PSFContainer],
                   prm: Union[str, PRMContainer]) -> LJDataFrame:
    if not isinstance(psf, PSFContainer):
        psf = PSFContainer.read(psf)

    if not isinstance(mol, MultiMolecule):
        mol = MultiMolecule.from_xyz(mol)
    else:
        mol = mol.copy()

    core_atoms = psf.atoms.index[psf.residue_id == 1] - 1
    lig_atoms = psf.atoms.index[psf.residue_id != 1] - 1
    mol.guess_bonds(atom_subset=lig_atoms)
    ij = get_idx(mol, core_atoms).T

    psf_to_atom_dict(psf)
    prm_df = LJDataFrame(index=mol.atoms.keys())
    prm_df.overlay_psf(psf)
    prm_df.overlay_prm(prm)
    prm_df['V_elstat'] = np.nan
    prm_df['V_LJ'] = np.nan

    #
    # dist =  _dist(mol, ij)

    return prm_df, ij


def get_idx(mol: MultiMolecule, core_atoms) -> np.ndarray:
    def dfs(at1: Atom, id_list: list, i: int, exclude: Set[Atom], depth: int = 0):
        exclude.add(at1)
        for bond in at1.bonds:
            at2 = bond.other_end(at1)
            if at2 in exclude:
                continue
            elif depth <= 3:
                id_list += [i, at2.id]
            dfs(at2, id_list, i, exclude, depth=1+depth)

    def gather_idx(molecule: Molecule) -> Generator[List[int], None, None]:
        for i, at in enumerate(molecule):
            id_list = []
            dfs(at, id_list, i, set())
            yield id_list

    _mol = mol.delete_atoms(core_atoms)
    molecule = _mol.as_Molecule(0)[0]
    molecule.set_atoms_id(start=0)

    idx = np.fromiter(chain.from_iterable(gather_idx(molecule)), dtype=int)
    idx += len(mol._get_atom_subset(core_atoms, as_array=True))
    idx.shape = -1, 2
    return idx


prm = '/Users/bvanbeek/Documents/GitHub/auto-FOX/FOX/examples/ligand.prm'
psf = '/Users/bvanbeek/Downloads/mol.psf'
xyz = get_example_xyz()

df, ij = start_workflow(xyz, psf, prm)
