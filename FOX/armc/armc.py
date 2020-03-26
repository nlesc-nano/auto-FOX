"""
FOX.armc.armc
=============

A module for performing Addaptive Rate Monte Carlo (ARMC) forcefield parameter optimizations.

Index
-----
.. currentmodule:: FOX.armc.armc
.. autosummary::
    ARMC

API
---
.. autoclass:: ARMC
    :members:
    :private-members:
    :special-members:

"""

from __future__ import annotations

import os
import io
from contextlib import redirect_stdout
from typing import (Tuple, TYPE_CHECKING, Any, Optional, Iterable, Mapping,
                    Union, AnyStr, TypeVar, ContextManager)

import yaml
import numpy as np

from scm.plams import init, finish

from .guess import guess_param
from .monte_carlo import MonteCarloABC
from .armc_to_yaml import to_yaml, from_yaml
from ..logger import Plams2Logger, wrap_plams_logger
from ..type_hints import ArrayLikeOrScalar, Literal
from ..io.hdf5_utils import (
    create_hdf5, to_hdf5, create_xyz_hdf5, _get_filename_xyz, hdf5_clear_status
)

try:
    Dumper = yaml.CDumper
except AttributeError:
    Dumper = yaml.Dumper  # type: ignore

if TYPE_CHECKING:
    from .multi_mol import MultiMolecule
    from .phi_updater import PhiUpdater
    from ..io import PSFContainer
else:
    MultiMolecule = f'{__package__}.multi_mol.MultiMolecule'
    PhiUpdater = f'{__package__}.phi_updater.PhiUpdater'
    PSFContainer = 'FOX.io.read_psf.PSFContainer'

__all__ = ['ARMC', 'run_armc']

KT = TypeVar('KT', bound=Tuple[float, ...])


class Init(ContextManager[None]):
    """A context manager for calling :func:`init` and :func:`finish`."""

    def __init__(self, path: Union[None, str, os.PathLike] = None,
                 folder: Union[None, str, os.PathLike] = None) -> None:
        self.path = path
        self.folder = folder

    def __enter__(self) -> None:
        init(self.path, self.folder)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        finish()


def run_armc(armc: ARMC,
             path: Union[None, str, os.PathLike] = None,
             folder: Union[None, str, os.PathLike] = None,
             logfile: Union[None, str, os.PathLike] = None,
             psf: Optional[Iterable[PSFContainer]] = None,
             restart: bool = False,
             guess: Optional[Mapping[str, Mapping]] = None) -> None:
    """A wrapper arround :class:`ARMC` for handling the JobManager."""
    # Create a .psf file if specified
    if psf is not None:
        for item in psf:
            item.write(None)

    # Guess the remaining unspecified parameters based on either UFF or the RDF
    if guess is not None:
        for k, v in guess.items():
            frozen = k if v['frozen'] else None
            guess_param(armc, mode=v['mode'], columns=k, frozen=frozen)

    # Initialize the ARMC procedure
    with Init(path=path, folder=folder):
        armc.logger = wrap_plams_logger(logfile, armc.__class__.__name__)
        writer = Plams2Logger(armc.logger,
                              lambda n: 'STARTED' in n,
                              lambda n: 'Renaming' in n,
                              lambda n: 'Trying to obtain results of crashed or failed job' in n)

        with redirect_stdout(writer):
            if not restart:  # To restart or not? That's the question
                armc()
            else:
                armc.restart()


class ARMC(MonteCarloABC):
    r"""The Addaptive Rate Monte Carlo class (:class:`.ARMC`).

    A subclass of :class:`MonteCarloABC`.

    Attributes
    ----------
    iter_len : :class:`int`
        The total number of ARMC iterations :math:`\kappa \omega`.

    super_iter_len : :class:`int`
        The length of each ARMC subiteration :math:`\kappa`.

    sub_iter_len : :class:`int`
        The length of each ARMC subiteration :math:`\omega`.

    phi : :class:`PhiUpdaterABC`
        A PhiUpdater instance.

    \**kwargs : :data:`~typing.Any`
        Keyword arguments for the :class:`MonteCarlo` superclass.

    """

    iter_len: int
    sub_iter_len: int
    phi: PhiUpdater

    @property
    def super_iter_len(self) -> int:
        """Get :attr:`ARMC.iter_len` ``//`` :attr:`ARMC.sub_iter_len`."""
        return self.iter_len // self.sub_iter_len

    def __init__(self, phi: PhiUpdater,
                 iter_len: int = 50000,
                 sub_iter_len: int = 100,
                 **kwargs: Any) -> None:
        r"""Initialize an :class:`ARMC` instance.

        Parameters
        ----------
        iter_len : :class:`int`
            The total number of ARMC iterations :math:`\kappa \omega`.

        sub_iter_len : :class:`int`
            The length of each ARMC subiteration :math:`\omega`.

        phi : :class:`PhiUpdaterABC`
            A PhiUpdater instance.

        \**kwargs : :data:`~typing.Any`
            Keyword arguments for the :class:`MonteCarlo` superclass.

        """
        super().__init__(**kwargs)

        # Settings specific to addaptive rate Monte Carlo (ARMC)
        self.phi = phi
        self.iter_len: int = iter_len
        self.sub_iter_len: int = sub_iter_len

    @classmethod
    def from_yaml(cls, filename: str) -> Tuple[ARMC, dict]:
        """Create a :class:`.ARMC` instance from a .yaml file.

        Parameters
        ----------
        filename : str
            The path+filename of a .yaml file containing all :class:`ARMC` settings.

        Returns
        -------
        |FOX.ARMC|_ and |dict|_
            A new :class:`ARMC` instance and
            a dictionary with keyword arguments for :func:`.run_armc`.

        """
        return from_yaml(cls, filename)

    def to_yaml(self, filename: Union[AnyStr, os.PathLike, io.IOBase],
                logfile: Optional[str] = None, path: Optional[str] = None,
                folder: Optional[str] = None) -> None:
        """Convert an :class:`ARMC` instance into a .yaml readable by :class:`ARMC.from_yaml`.

        Parameters
        ----------
        filename : :class:`str`, :class:`bytes`, :class:`os.pathlike` or :class:`io.IOBase`
            A filename or a file-like object.

        """
        to_yaml(self, filename, logfile, path, folder)

    def __call__(self, start: int = 0, key_new: Optional[KT] = None) -> None:
        """Initialize the Addaptive Rate Monte Carlo procedure."""
        if start == 0:
            create_hdf5(self.hdf5, self)  # Construct the HDF5 file

            key_new = self._get_first_key()  # Initialize the first MD calculation
            if np.inf in self[key_new]:
                raise RuntimeError('One or more jobs crashed in the first ARMC iteration; '
                                   'manual inspection of the cp2k output is recomended')
            self.clear_jobs()

        elif key_new is None:
            raise TypeError("'key_new' cannot be 'None' if 'start' is larger than 0")

        # Start the main loop
        for kappa in range(start, self.super_iter_len):
            acceptance = np.zeros(self.sub_iter_len, dtype=bool)
            create_xyz_hdf5(self.hdf5, self.molecule, iter_len=self.sub_iter_len)

            for omega in range(self.sub_iter_len):
                key_new = self.do_inner(kappa, omega, acceptance, key_new)
            self.apply_phi(acceptance)

    def do_inner(self, kappa: int, omega: int, acceptance: np.ndarray, key_old: KT) -> KT:
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
        # Step 1: Perform a random move
        key_new = self.move()

        # Step 2: Check if the move has been performed already; calculate PES descriptors if not
        pes_new, mol_list = self.get_pes_descriptors(key_new)

        # Step 3: Evaluate the auxiliary error; accept if the new parameter set lowers the error
        aux_new = self.get_aux_error(pes_new)
        aux_old = self[key_old]
        error_change = (aux_new - aux_old).sum()
        accept = error_change < 0

        # Step 4: Update the auxiliary error history, apply phi & update job settings
        acceptance[omega] = accept
        if accept:
            self.logger.info(f"Accepting move {(kappa, omega)}; total error change / error: "
                             f"{round(error_change, 4)} / {round(aux_new.sum(), 4)}\n")
            self[key_new] = self.phi(aux_new)
            self.param['param_old'] = self.param['param']

        else:
            self.logger.info(f"Rejecting move {(kappa, omega)}; total error change / error: "
                             f"{round(error_change, 4)} / {round(aux_new.sum(), 4)}\n")
            self[key_new] = aux_new
            self[key_old] = self.apply_phi(aux_old)
            key_new = key_old

        # Step 5: Export the results to HDF5
        self.to_hdf5(mol_list, accept, aux_new, pes_new, kappa, omega)
        if not accept:
            self.param['param'][:] = self.param['param_old']
        return key_new

    def apply_phi(self, value: ArrayLikeOrScalar) -> np.ndarray:
        """Apply :attr:`phi` to **value**."""
        return self.phi(value)

    def _get_first_key(self) -> KT:
        """Create a the ``history_dict`` variable and its first key.

        The to-be returned key servers as the starting argument for :meth:`.do_inner`,
        the latter method relying on both producing and requiring a key as argument.

        Returns
        -------
        |tuple|_ [|np.ndarray|_ [|float|_]]
            A tuple of Numpy arrays.

        """
        key: KT = tuple(self.param['param'].values)  # type: ignore
        pes, _ = self.get_pes_descriptors(key, get_first_key=True)

        self[key] = self.get_aux_error(pes)
        self.param['param_old'][:] = self.param['param']
        return key

    def to_hdf5(self, mol_list: Optional[Iterable[MultiMolecule]],
                accept: bool, aux_new: np.ndarray,
                pes_new: Mapping[str, np.ndarray],
                kappa: int, omega: int) -> None:
        r"""Construct a dictionary with the **hdf5_kwarg** and pass it to :func:`.to_hdf5`.

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

        kappa : int
            The super-iteration, :math:`\kappa`, in :meth:`ARMC.__call__`.

        omega : int
            The sub-iteration, :math:`\omega`, in :meth:`ARMC.__call__`.

        Returns
        -------
        |dict|_
            A dictionary with the **hdf5_kwarg** argument for :func:`.to_hdf5`.

        """
        phi = self.phi.phi
        param_key: Literal['param', 'param_old'] = 'param' if accept else 'param_old'

        hdf5_kwarg = {
            'param': self.param['param'],
            'xyz': mol_list if not None else np.nan,
            'phi': phi,
            'acceptance': accept,
            'aux_error': aux_new,
            'aux_error_mod': np.append(self.param[param_key].values, phi)
        }
        hdf5_kwarg.update(pes_new)

        to_hdf5(self.hdf5, hdf5_kwarg, kappa, omega)

    def get_aux_error(self, pes_dict: Mapping[str, ArrayLikeOrScalar]) -> np.ndarray:
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
        pes_dict : [|dict|_ [str, |np.ndarray|_ [|np.float64|_]]
            An dictionary with :math:`m*n` PES descriptors each.

        Returns
        -------
        :math:`m*n` |np.ndarray|_ [|np.float64|_]:
            An array with :math:`m*n` auxilary errors

        """
        def norm_mean(key: str, mm_pes: ArrayLikeOrScalar) -> float:
            qm_pes = self.pes[key].ref
            QM = np.asarray(qm_pes, dtype=float)
            MM = np.asarray(mm_pes, dtype=float)
            ret = (QM - MM)**2
            return ret.sum() / QM.sum()

        length = 1 + max(int(k.rsplit('.')[-1]) for k in pes_dict.keys())

        generator = (norm_mean(k, v) for k, v in pes_dict.items())
        ret = np.fromiter(generator, dtype=float, count=len(pes_dict))
        ret.shape = length, -1
        return ret.T

    def restart(self) -> None:
        r"""Restart a previously started Addaptive Rate Monte Carlo procedure."""
        i, j, key, acceptance = self._restart_from_hdf5()

        # Validate the xyz .hdf5 file; create a new one if required
        xyz = _get_filename_xyz(self.hdf5)
        if not os.path.isfile(xyz):
            create_xyz_hdf5(self.hdf5, self.molecule, iter_len=self.sub_iter_len)

        # Check that both .hdf5 files can be opened; clear their status if not
        closed = hdf5_clear_status(xyz)
        if not closed:
            self.logger.warning(f"Unable to open ...{os.sep}{os.path.basename(xyz)}, "
                                "file status was forcibly reset")
        closed = hdf5_clear_status(self.hdf5)
        if not closed:
            self.logger.warning(f"Unable to open ...{os.sep}{os.path.basename(self.hdf5)}, "
                                "file status was forcibly reset")

        # Finish the current set of sub-iterations
        j += 1
        for omega in range(j, self.sub_iter_len):
            key = self.do_inner(i, omega, acceptance, key)
        self.phi.update(acceptance)
        i += 1

        # And continue
        self(start=i, key_new=key)

    def _restart_from_hdf5(self) -> Tuple[int, int, KT, np.ndarray]:
        """Read and process the .hdf5 file for :meth:`ARMC.restart`."""
        import h5py

        with h5py.File(self.hdf5, 'r', libver='latest') as f:
            i, j = f.attrs['super-iteration'], f.attrs['sub-iteration']
            if i < 0:
                raise ValueError(f'i: {i.__class__.__name__} = {i}')
            self.logger.info('Restarting ARMC procedure from super-iteration '
                             f'{i} & sub-iteration {j}')

            self.phi.phi = f['phi'][i]
            self.param['param'] = self.param['param_old'] = f['param'][i, j]
            for key, err in zip(f['param'][i], f['aux_error'][i]):
                key = tuple(key)
                self[key] = err
            acceptance = f['acceptance'][i]

            # Find the last error which is not np.nan
            final_key: KT = self._find_restart_key(f, i)
        return i, j, final_key, acceptance

    @staticmethod
    def _find_restart_key(f: Mapping[str, np.ndarray], i: int) -> KT:
        """Construct a key for the parameter which is not ``nan``."""
        while i >= 0:
            aux_error = f['aux_error'][i]
            param_old = f['param'][i]
            aux_nan = np.isnan(aux_error).any(axis=(1, 2))

            try:  # Its no longer np.nan
                return tuple(param_old[~aux_nan][-1])  # type: ignore
            except IndexError:
                i -= 1
        else:
            raise RuntimeError('Not a single successful MD-calculation was found; '
                               'restarting is not possible')
