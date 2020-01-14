"""
FOX.armc_functions.monte_carlo_test
===================================

Various functions/methods for the testing of :class:`.MonteCarlo` class.

"""

import os
from typing import List, Optional, Iterable, Tuple

import numpy as np

from FOX import MultiMolecule, ARMC
from FOX.io.read_xyz import XYZError


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


def do_inner(self, kappa: int, omega: int, acceptance: np.ndarray,
             key_old: Tuple[float, ...]) -> Tuple[Tuple[float, ...], ...]:
    r"""Run the inner loop of the :meth:`ARMC.__call__` method.

    Parameters
    ----------
    kappa : int
        The super-iteration, :math:`\kappa`, in :meth:`ARMC.__call__`.

    omega : int
        The sub-iteration, :math:`\omega`, in :meth:`ARMC.__call__`.

    history_dict : |dict|_ [|tuple|_ [|float|_], |np.ndarray|_ [|np.float64|_]]
        A dictionary with parameters as keys and a list of PES descriptors as values.

    key_new : tuple [float]
        A tuple with the latest set of forcefield parameters.

    Returns
    -------
    |tuple|_ [|float|_]:
        The latest set of parameters.

    """
    key_new = ARMC.do_inner(self, kappa, omega, acceptance, key_old)
    ...  # Insert some testing function here
    return key_new
