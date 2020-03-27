"""
FOX.armc.monte_carlo_test
=========================

Various functions/methods for the testing of :class:`.MonteCarlo` class.

"""

import os
from typing import List, Optional, Iterable, TYPE_CHECKING

from ..io.read_xyz import XYZError

if TYPE_CHECKING:
    from ..classes import MultiMolecule
else:
    from ..type_alias import MultiMolecule


def _md_preopt(self) -> List[Optional[MultiMolecule]]:
    """Peform a geometry optimization.

    Optimizations are performed on all molecules in :attr:`MonteCarlo.job`[``"molecule"``].

    Returns
    -------
    |FOX.MultiMolecule|_ and |str|_
        A list of :class:`.MultiMolecule` instance constructed from the geometry optimization.
        The :class:`.MultiMolecule` list is replaced with ``None`` if the job crashes.

    """
    results_list = next(self.md_preopt_iterator)

    # Preoptimize
    mol_list = []
    for results in results_list:
        try:  # Construct and return a MultiMolecule object
            mol = MultiMolecule.from_xyz(results.get_xyz_path())
            mol.round(3)
        except TypeError:  # The geometry optimization crashed
            return None
        mol_list.append(mol)

    self.job_cache = []
    return mol_list


def _md(self, mol_preopt: Iterable[MultiMolecule]) -> Optional[List[MultiMolecule]]:
    """Peform a molecular dynamics simulation (MD).

    Simulations are performed on all molecules in **mol_preopt**.

    Parameters
    ----------
    mol_preopt : |list|_ [|FOX.MultiMolecule|_]
        An iterable consisting of :class:`.MultiMolecule` instance(s) constructed from
        geometry pre-optimization(s) (see :meth:`._md_preopt`).

    Returns
    -------
    |list|_ [|FOX.MultiMolecule|_], optional
        A list of :class:`.MultiMolecule` instance(s) constructed from the MD simulation.
        Return ``None`` if the job crashes.

    """
    results_list = next(self.md_iterator)

    # Run MD
    mol_list = []
    for results in results_list:
        try:  # Construct and return a MultiMolecule object
            path = results.get_xyz_path()
            mol = MultiMolecule.from_xyz(path)
            mol.round(3)
        except TypeError:  # The MD simulation crashed
            return None
        except XYZError:  # The .xyz file is unreadable for some reason
            self.logger.warning(f"Failed to parse ...{os.sep}{os.path.basename(path)}")
            return None
        mol_list.append(mol)

    self.job_cache = []
    return mol_list
