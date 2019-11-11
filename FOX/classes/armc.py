"""
FOX.classes.armc
================

A module for performing Addaptive Rate Monte Carlo (ARMC) forcefield parameter optimizations.

Index
-----
.. currentmodule:: FOX.classes.monte_carlo
.. autosummary::
    ARMC

API
---
.. autoclass:: FOX.classes.monte_carlo.ARMC
    :members:
    :private-members:
    :special-members:

"""

from __future__ import annotations

from typing import Tuple, Dict, Any, Optional, Iterable

import numpy as np

from scm.plams import init, finish, config

from .monte_carlo import MonteCarlo
from ..io.hdf5_utils import (create_hdf5, to_hdf5, create_xyz_hdf5)
from ..functions.utils import get_template
from ..armc_functions.sanitization import init_armc_sanitization

__all__ = ['ARMC']


class ARMC(MonteCarlo):
    r"""The Addaptive Rate Monte Carlo class (:class:`.ARMC`).

    A subclass of :class:`.MonteCarlo`.

    Attributes
    ----------
    armc : |plams.Settings|_
        A PLAMS Settings instance with ARMC-specific settings.
        Contains the following keys:

        * ``"gamma"`` (|float|_): The constant :math:`\gamma`.
        * ``"a_target"`` (|float|_): The target acceptance rate :math:`\alpha_{t}`.
        * ``"iter_len"`` (|int|_): The total number of ARMC iterations :math:`\kappa \omega`.
        * ``"sub_iter_len"`` (|int|_): The length of each ARMC subiteration :math:`\omega`.

    phi : |plams.Settings|_
        A PLAM Settings instance with :math:`\phi`-specific settings.
        Contains the following keys:

        * ``"phi"`` (|float|_): The variable :math:`\phi`.
        * ``"arg"`` (|list|_): A list of arguments for :attr:`.ARMC.phi` [``"func"``].
        * ``"func"`` (|type|_): The callable used for applying :math:`\phi` to the auxiliary error.
        * ``"kwarg"`` (|dict|_): A dictionary with keyword arguments
          for :attr:`.ARMC.phi` [``"func"``].

    """

    @property
    def super_iter_len(self) -> int:
        return self.armc.iter_len // self.armc.sub_iter_len

    def __init__(self, iter_len=50000, sub_iter_len=100, gamma=200, a_target=0.25,
                 phi=1.0, apply_phi=np.add, **kwarg) -> None:
        """Initialize a :class:`ARMC` instance."""
        super().__init__(**kwarg)

        # Settings specific to addaptive rate Monte Carlo (ARMC)
        self.iter_len = iter_len
        self.sub_iter_len = sub_iter_len
        self.gamma = gamma
        self.a_target = a_target

        # Settings specific to handling the phi parameter in ARMC
        self.phi = phi
        self.apply_phi = apply_phi

    def _get_first_key(self) -> Tuple[Tuple[float, ...], ...]:
        """Create a the ``history_dict`` variable and its first key.

        The to-be returned key servers as the starting argument for :meth:`.do_inner`,
        the latter method relying on both producing and requiring a key as argument.

        Returns
        -------
        |dict|_ [|tuple|_ [|float|_], |np.ndarray|_ [|np.float64|_]] and |tuple|_ [|float|_]
            Returns two items:
            * A dictionary with parameters as keys and a list of PES descriptors as values.
            * A tuple with the latest set of forcefield parameters.

        """
        key = tuple(self.param['param'].values)
        pes, _ = self.get_pes_descriptors(key)

        self[key] = self.get_aux_error(pes)
        self.param['param_old'] = self.param['param']
        return key

    def __call__(self) -> None:
        """Initialize the Addaptive Rate Monte Carlo procedure."""
        # Construct the HDF5 file
        create_hdf5(self.hdf5_file, self)

        # Initialize the first MD calculation
        key_new = self._get_first_key()

        # Start the main loop
        for kappa in range(self.super_iter_len):
            acceptance = np.zeros(self.sub_iter_len, dtype=bool)
            create_xyz_hdf5(self.hdf5_file, self.molecule, iter_len=self.sub_iter_len)

            for omega in range(self.sub_iter_len):
                key_new = self.do_inner(kappa, omega, acceptance, key_new)
            self.update_phi(acceptance)

    def do_inner(self, kappa: int, omega: int, acceptance: np.ndarray,
                 key_old: Tuple[float]) -> Tuple[Tuple[float, ...], ...]:
        r"""Run the inner loop of the :meth:`ARMC.__call__` method.

        Parameters
        ----------
        kappa : int
            The super-iteration, :math:`\kappa`, in :meth:`ARMC.__call__`.

        omega : int
            The sub-iteration, :math:`\imega`, in :meth:`ARMC.__call__`.

        history_dict : |dict|_ [|tuple|_ [|float|_], |np.ndarray|_ [|np.float64|_]]
            A dictionary with parameters as keys and a list of PES descriptors as values.

        key_new : tuple [float]
            A tuple with the latest set of forcefield parameters.

        Returns
        -------
        |tuple|_ [|float|_]:
            The latest set of parameters.

        """
        # Step 1: Perform a random move
        key_new = self.move()

        # Step 2: Check if the move has been performed already; calculate PES descriptors if not
        pes_new, mol_list = self.get_pes_descriptors(key_new)

        # Step 3: Evaluate the auxiliary error; accept if the new parameter set lowers the error
        aux_new = self.get_aux_error(pes_new)
        aux_old = self[key_old]
        accept = True if (aux_new - aux_old).sum() < 0 else False

        # Step 4: Update the auxiliary error history, apply phi & update job settings
        acceptance[omega] = accept
        if accept:
            self[key_new] = self.apply_phi(aux_new)
            self.param['param_old'] = self.param['param']
        else:
            self[key_new] = aux_new
            self[key_old] = self.apply_phi(aux_old)
            self.param['param'] = self.param['param_old']
            key_new = key_old

        # Step 5: Export the results to HDF5
        hdf5_kwarg = self._hdf5_kwarg(mol_list, accept, aux_new, pes_new)
        to_hdf5(self.hdf5_file, hdf5_kwarg, kappa, omega)
        return key_new

    def _hdf5_kwarg(self, mol_list: Iterable[Optional['FOX.MultiMolecule']],
                    accept: bool, aux_new: np.ndarray,
                    pes_new: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Construct a dictionary with the **hdf5_kwarg** argument for :func:`.to_hdf5`.

        Parameters
        ----------
        mol_list : |List|_ [|FOX.MultiMolecule|_]
            An iterable consisting of :class:`.MultiMolecule` instances (or ``None``).

        accept : bool
            Whether or not the latest set of parameters was accepted.

        aux_new : |np.ndarray|_
            The latest auxiliary error.

        pes_new : |dict|_ [|str|_, |np.ndarray|_]
            A dictionary of PES descriptors.

        Returns
        -------
        |dict|_
            A dictionary with the **hdf5_kwarg** argument for :func:`.to_hdf5`.

        """
        param_key = 'param' if accept else 'param_old'
        hdf5_kwarg = {
            'param': self.param['param'],
            'xyz': mol_list if not None else np.nan,
            'phi': self.phi,
            'acceptance': accept,
            'aux_error': aux_new,
            'aux_error_mod': np.append(self.param[param_key].values, self.phi)
        }

        for i, dct in enumerate(pes_new):
            for k, v in dct.items():
                k += f'.{i}'
                hdf5_kwarg[k] = v

        return hdf5_kwarg

    def get_aux_error(self, pes_list: Iterable[Dict[str, np.ndarray]]) -> np.ndarray:
        r"""Return the auxiliary error :math:`\Delta \varepsilon_{QM-MM}`.

        The auxiliary error is constructed using the PES descriptors in **values**
        with respect to **self.ref**.

        The default function is equivalent to:

        .. math::

            \Delta \varepsilon_{QM-MM} =
                \frac{ \sum_{i}^{N} |r_{i}^{QM} - r_{i}^{MM}|^2 }
                {r_{i}^{QM}}

        Parameters
        ----------
        pes_list : |list|_ [|dict|_ [str, |np.ndarray|_ [|np.float64|_]]]
            An iterable consisting of :math:`m` dictionaries with :math:`n` PES descriptors each.

        Returns
        -------
        :math:`m*n` |np.ndarray|_ [|np.float64|_]:
            An array with :math:`m*n` auxilary errors

        """
        def norm_mean(key: str, mm_pes: np.ndarray, i: int) -> float:
            qm_pes = self.pes[key].ref[i]
            A, B = np.asarray(qm_pes, dtype=float), np.asarray(mm_pes, dtype=float)
            ret = (A - B)**2
            return ret.sum() / A.sum()

        ret = np.array([
            norm_mean(k, v, i) for i, dct in enumerate(pes_list) for k, v in dct.items()
        ])

        ret.shape = len(pes_list), len(ret) // len(pes_list)
        return ret

    def update_phi(self, acceptance: np.ndarray) -> None:
        r"""Update the variable :math:`\phi`.

        :math:`\phi` is updated based on the target accepatance rate, :math:`\alpha_{t}`, and the
        acceptance rate, **acceptance**, of the current super-iteration.

        * The values are updated according to the provided settings in **self.armc**.

        The default function is equivalent to:

        .. math::

            \phi_{\kappa \omega} =
            \phi_{ ( \kappa - 1 ) \omega} * \gamma^{
                \text{sgn} ( \alpha_{t} - \overline{\alpha}_{ ( \kappa - 1 ) })
            }

        Parameters
        ----------
        acceptance : |np.ndarray|_ [|bool|_]
            A 1D boolean array denoting the accepted moves within a sub-iteration.

        """
        sign = np.sign(self.a_target - np.mean(acceptance))
        self.phi *= self.gamma**sign

    def restart(self, filename: str) -> None:
        r"""Restart a previously started Addaptive Rate Monte Carlo procedure.

        Restarts from the beginning of the last super-iteration :math:`\kappa`.

        Parameters
        ----------
        filename : str
            The path+name of the an ARMC hdf5 file.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError


def run_armc(armc: ARMC, path: Optional[str] = None, folder: Optional[str] = None,
             logfile: Optional[str] = None, psf: Optional['PSFContainer'] = None) -> None:
    init(path=path, folder=folder)
    if logfile is not None:
        config.default_jobmanager.logfile = logfile
        config.log.file = 3

    # Create a .psf file if specified
    if psf:
        psf.write_psf()

    armc()
    finish()
