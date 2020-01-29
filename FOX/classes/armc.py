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

import os
import io
import logging
from typing import Tuple, Dict, Any, Optional, Iterable, Callable, Mapping, Union, AnyStr
from contextlib import AbstractContextManager, redirect_stdout

import yaml
import numpy as np

from scm.plams import init, finish, config, Settings

from .monte_carlo import MonteCarlo
from ..logger import Plams2Logger, get_logger
from ..io.hdf5_utils import (
    create_hdf5, to_hdf5, create_xyz_hdf5, _get_filename_xyz, hdf5_clear_status
)
from ..functions.utils import get_template
from ..io.file_container import NullContext
from ..armc_functions.guess import guess_param
from ..armc_functions.df_to_dict import df_to_dict
from ..armc_functions.sanitization import init_armc_sanitization

__all__ = ['ARMC', 'run_armc']

try:
    Dumper = yaml.CDumper
except AttributeError:
    Dumper = yaml.Dumper


class Init(AbstractContextManager):
    """A context manager for calling :func:`init` and :func:`finish`."""

    def __init__(self, path=None, folder=None) -> None:
        self.path = path
        self.folder = folder

    def __enter__(self) -> None:
        init(self.path, self.folder)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        finish()


def run_armc(armc: 'ARMC',
             path: Optional[str] = None,
             folder: Optional[str] = None,
             logfile: Optional[str] = None,
             psf: Optional[Iterable['PSFContainer']] = None,
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
            frozen = (k if v['frozen'] else None)
            guess_param(armc, mode=v['mode'], columns=k, frozen=frozen)

    # Initialize the ARMC procedure
    with Init(path=path, folder=folder):
        armc.logger = _get_armc_logger(logfile, armc.__class__.__name__)
        writer = Plams2Logger(armc.logger,
                              lambda n: 'STARTED' in n,
                              lambda n: 'Renaming' in n,
                              lambda n: 'Trying to obtain results of crashed or failed job' in n)

        with redirect_stdout(writer):
            if not restart:  # To restart or not? That's the question
                armc()
            else:
                armc.restart()


def _get_armc_logger(logfile: Optional[str], name: str, **kwargs) -> logging.Logger:
    """Substitute the PLAMS .log file for one created by a :class:`Logger<logging.Logger>`."""
    # Define filenames
    plams_logfile = config.default_jobmanager.logfile
    logfile = os.path.abspath(logfile) if logfile is not None else plams_logfile

    # Modify the plams logger
    config.log.time = False
    config.log.file = 0

    # Replace the plams logger with a proper logging.Logger instance
    os.remove(plams_logfile)
    logger = get_logger(name, filename=logfile, **kwargs)
    if plams_logfile != logfile:
        try:
            os.symlink(logfile, plams_logfile)
        except OSError:
            pass

    return logger


class ARMC(MonteCarlo):
    r"""The Addaptive Rate Monte Carlo class (:class:`.ARMC`).

    A subclass of :class:`.MonteCarlo`.

    Parameters
    ----------
    iter_len : int
        The total number of ARMC iterations :math:`\kappa \omega`.

    sub_iter_len : int
        The length of each ARMC subiteration :math:`\omega`.

    gamma : float
        The constant :math:`\gamma`.

    a_target : float
        The target acceptance rate :math:`\alpha_{t}`.

    phi : float
        The variable :math:`\phi`.

    apply_phi : |Callable|_
        The callable used for applying :math:`\phi` to the auxiliary error.
        The callable should be able to take 2 floats as argument and return a new float.

    \**kwargs : |Any|_
        Keyword arguments for the :class:`MonteCarlo` superclass.

    """

    @property
    def super_iter_len(self) -> int:
        """Get :attr:`ARMC.iter_len` ``//`` :attr:`ARMC.sub_iter_len`."""
        return self.iter_len // self.sub_iter_len

    def __init__(self, iter_len: int = 50000, sub_iter_len: int = 100, gamma: int = 200,
                 a_target: float = 0.25, phi: float = 1.0,
                 apply_phi: Callable[[float, float], float] = np.add, **kwargs) -> None:
        """Initialize a :class:`ARMC` instance."""
        super().__init__(**kwargs)

        # Settings specific to addaptive rate Monte Carlo (ARMC)
        self.iter_len: int = iter_len
        self.sub_iter_len: int = sub_iter_len
        self.gamma: int = gamma
        self.a_target: float = a_target

        # Settings specific to handling the phi parameter in ARMC
        self.phi: float = phi
        self.apply_phi: Callable[[float, float], float] = apply_phi

    @classmethod
    def from_yaml(cls, filename: str) -> Tuple['ARMC', dict]:
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
        # Load the .yaml file
        if os.path.isfile(filename):
            path, filename = os.path.split(filename)
        else:
            path = None
        yaml_dict = get_template(filename, path=path)

        # Parse and sanitize the .yaml file
        s, pes_kwarg, job_kwarg = init_armc_sanitization(yaml_dict)
        self = cls.from_dict(s)
        for name, options in pes_kwarg.items():
            self.add_pes_evaluator(name, options.func, options.args, options.kwargs)
        return self, job_kwarg

    def to_yaml(self, filename: Union[AnyStr, os.PathLike, io.IOBase],
                logfile: Optional[str] = None, path: Optional[str] = None,
                folder: Optional[str] = None) -> None:
        """Convert an :class:`ARMC` instance into a .yaml readable by :class:`ARMC.from_yaml`.

        Parameters
        ----------
        filename : :class:`str`, :class:`bytes`, :class:`os.pathlike` or :class:`io.IOBase`
            A filename or a file-like object.

        """
        try:  # is filename an actual filename or a file-like object?
            assert callable(filename.write)
        except (AttributeError, AssertionError):
            manager = open
        else:
            manager = NullContext

        # The armc block
        s = Settings()
        s.armc.iter_len = self.iter_len
        s.armc.sub_iter_len = self.sub_iter_len
        s.armc.gamma = self.gamma
        s.armc.a_target = self.a_target
        s.armc.phi = self.phi

        # The hdf5 block
        s.hdf5_file = self.hdf5_file

        # The pram block
        s.param = df_to_dict(self.param)

        # The job block
        s.job.path = os.getcwd() if path is None else str(path)
        if logfile is not None:
            s.job.logfile = logfile
        if folder is not None:
            s.job.folder = folder
        s.job.keep_files = self.keep_files
        s.job.rmsd_threshold = self.rmsd_threshold
        s.job.job_type = f'{self.job_type.func.__module__}.{self.job_type.func.__qualname__}'
        s.job.name = self.job_type.keywords['name']
        s.job.preopt_settings = self.preopt_settings[0] if self.preopt_settings else None
        s.job.md_settings = self.md_settings[0]

        s.psf = {}
        with Settings.supress_missing():
            try:
                s.psf.psf_file = [s.input.force_eval.subsys.topology.conn_file_name for
                                  s in self.md_settings]
                del s.job.md_settings.input.force_eval.subsys.topology.conn_file_name
            except KeyError:
                pass

            try:
                del s.job.preopt_settings.input.force_eval.subsys.topology.conn_file_name
            except (AttributeError, KeyError):
                pass

        # The molecule block
        s.molecule = [mol.properties.filename for mol in self.molecule]

        # The pes block
        for name, partial in self.pes.items():
            pes_dict = s.pes[name.rsplit('.', maxsplit=1)[0]]
            pes_dict.func = f'{partial.func.__module__}.{partial.func.__qualname__}'
            pes_dict.args = list(partial.args)
            if 'kwargs' not in pes_dict:
                pes_dict.kwargs = []
            pes_dict.kwargs.append(partial.keywords)

        # The move block
        s.move.range.stop = round(float(self.move_range.max() - 1), 8)
        s.move.range.step = round(float(abs(self.move_range[-1] - self.move_range[-2])), 8)
        s.move.range.start = round(float(self.move_range[len(self.move_range) // 2] - 1), 8)

        # Write the file
        with manager(filename, 'w') as f:
            f.write(yaml.dump(s.as_dict(), Dumper=Dumper))

    def __call__(self, start: int = 0, key_new: Optional[Tuple[np.ndarray, ...]] = None) -> None:
        """Initialize the Addaptive Rate Monte Carlo procedure."""
        if start == 0:
            create_hdf5(self.hdf5_file, self)  # Construct the HDF5 file

            key_new = self._get_first_key()  # Initialize the first MD calculation
            if np.inf in self[key_new]:
                ex = RuntimeError('One or more jobs crashed in the first ARMC iteration; '
                                  'manual inspection of the cp2k output is recomended')
                self.logger.critical(f'{ex.__class__.__name__}: {ex}', exc_info=True)
                raise ex
            self.clear_job_cache()

        elif key_new is None:
            ex = TypeError("'key_new' cannot be 'None' if 'start' is larger than 0")
            self.logger.critical(f'{ex.__class__.__name__}: {ex}', exc_info=True)
            raise ex

        # Start the main loop
        for kappa in range(start, self.super_iter_len):
            acceptance = np.zeros(self.sub_iter_len, dtype=bool)
            create_xyz_hdf5(self.hdf5_file, self.molecule, iter_len=self.sub_iter_len)

            for omega in range(self.sub_iter_len):
                key_new = self.do_inner(kappa, omega, acceptance, key_new)
            self.update_phi(acceptance)

    def do_inner(self, kappa: int, omega: int, acceptance: np.ndarray,
                 key_old: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, ...]:
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
            self[key_new] = self.apply_phi(aux_new, self.phi)
            self.param['param_old'] = self.param['param']
        else:
            self.logger.info(f"Rejecting move {(kappa, omega)}; total error change / error: "
                             f"{round(error_change, 4)} / {round(aux_new.sum(), 4)}\n")
            self[key_new] = aux_new
            self[key_old] = self.apply_phi(aux_old, self.phi)
            key_new = key_old

        # Step 5: Export the results to HDF5
        hdf5_kwarg = self._hdf5_kwarg(mol_list, accept, aux_new, pes_new)
        to_hdf5(self.hdf5_file, hdf5_kwarg, kappa, omega)
        if not accept:
            self.param['param'] = self.param['param_old']
        return key_new

    def _get_first_key(self) -> Tuple[np.ndarray, ...]:
        """Create a the ``history_dict`` variable and its first key.

        The to-be returned key servers as the starting argument for :meth:`.do_inner`,
        the latter method relying on both producing and requiring a key as argument.

        Returns
        -------
        |tuple|_ [|np.ndarray|_ [|float|_]]
            A tuple of Numpy arrays.

        """
        key = tuple(self.param['param'].values)
        pes, _ = self.get_pes_descriptors(key, get_first_key=True)

        self[key] = self.get_aux_error(pes)
        self.param['param_old'] = self.param['param']
        return key

    def _hdf5_kwarg(self, mol_list: Optional[Iterable['FOX.MultiMolecule']],
                    accept: bool, aux_new: np.ndarray,
                    pes_new: Mapping[str, np.ndarray]) -> Dict[str, Any]:
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
        hdf5_kwarg.update(pes_new)

        return hdf5_kwarg

    def get_aux_error(self, pes_dict: Dict[str, np.ndarray]) -> np.ndarray:
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
        def norm_mean(key: str, mm_pes: np.ndarray) -> float:
            qm_pes = self.pes[key].ref
            QM, MM = np.asarray(qm_pes, dtype=float), np.asarray(mm_pes, dtype=float)
            ret = (QM - MM)**2
            return ret.sum() / QM.sum()

        length = 1 + max(int(k.rsplit('.')[-1]) for k in pes_dict.keys())

        generator = (norm_mean(k, v) for k, v in pes_dict.items())
        ret = np.fromiter(generator, dtype=float, count=len(pes_dict))
        ret.shape = length, -1
        return ret.T

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
        phi_old = self.phi
        self.phi *= self.gamma**sign
        self.logger.info(f"Updating phi: {phi_old} -> {self.phi}")

    def restart(self) -> None:
        r"""Restart a previously started Addaptive Rate Monte Carlo procedure."""
        i, j, key, acceptance = self._restart_from_hdf5()

        # Validate the xyz .hdf5 file; create a new one if required
        xyz = _get_filename_xyz(self.hdf5_file)
        if not os.path.isfile(xyz):
            create_xyz_hdf5(self.hdf5_file, self.molecule, iter_len=self.sub_iter_len)

        # Check that both .hdf5 files can be opened; clear their status if not
        closed = hdf5_clear_status(xyz)
        if not closed:
            self.logger.warning(f"Unable to open ...{os.sep}{os.path.basename(xyz)}, "
                                "file status was forcibly reset")
        closed = hdf5_clear_status(self.hdf5_file)
        if not closed:
            self.logger.warning(f"Unable to open ...{os.sep}{os.path.basename(self.hdf5_file)}, "
                                "file status was forcibly reset")

        # Finish the current set of sub-iterations
        j += 1
        for omega in range(j, self.sub_iter_len):
            key = self.do_inner(i, omega, acceptance, key)
        self.update_phi(acceptance)
        i += 1

        # And continue
        self(start=i, key_new=key)

    def _restart_from_hdf5(self) -> Tuple[int, int, Tuple[np.ndarray, ...], np.ndarray]:
        r"""Read and process the .hdf5 file for :meth:`ARMC.restart`."""
        import h5py

        with h5py.File(self.hdf5_file, 'r', libver='latest') as f:
            i, j = f.attrs['super-iteration'], f.attrs['sub-iteration']
            if i < 0:
                raise ValueError(f'i: {i.__class__.__name__} = {i}')
            self.logger.info('Restarting ARMC procedure from super-iteration '
                             f'{i} & sub-iteration {j}')

            self.phi = f['phi'][i]
            self.param['param'] = self.param['param_old'] = f['param'][i, j]
            for key, err in zip(f['param'][i], f['aux_error'][i]):
                key = tuple(key)
                self[key] = err
            acceptance = f['acceptance'][i]

            # Find the last error which is not np.nan
            i_ = i
            while np.isnan(err).any():
                if i_ < 0:
                    raise RuntimeError('Not a single successful MD-calculation was found; '
                                       'restarting is not possible')
                aux_error = f['aux_error'][i_]
                param_old = f['param'][i_]
                aux_nan = np.isnan(aux_error).any(axis=(1, 2))
                try:
                    key = tuple(param_old[~aux_nan][-1])
                except IndexError:
                    i_ -= 1
                else:
                    err = aux_error[~aux_nan][-1]  # Its no longer np.nan

        return i, j, key, acceptance
