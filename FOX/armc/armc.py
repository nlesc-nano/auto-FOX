"""A module for performing Addaptive Rate Monte Carlo (ARMC) forcefield parameter optimizations.

Index
-----
.. currentmodule:: FOX.armc
.. autosummary::
    ARMC

API
---
.. autoclass:: ARMC
    :members:

"""

from __future__ import annotations

import os
import io
from pathlib import Path
from collections import abc
from typing import (
    Tuple, TYPE_CHECKING, Any, Optional, Iterable, Mapping, Union, AnyStr, Callable,
    overload, Dict, List
)

import numpy as np
from nanoutils import Literal

from .monte_carlo import MonteCarloABC
from .armc_to_yaml import to_yaml
from ..type_hints import ArrayLikeOrScalar, ArrayOrScalar
from ..io.hdf5_utils import (
    create_hdf5, to_hdf5, create_xyz_hdf5, _get_filename_xyz, hdf5_clear_status
)

if TYPE_CHECKING:
    import functools
    from .phi_updater import PhiUpdater
    from ..io.read_psf import PSFContainer
    from ..classes import MultiMolecule
else:
    from ..type_alias import MultiMolecule, PhiUpdater, PSFContainer

__all__ = ['ARMC']

PesDict = Dict[str, ArrayOrScalar]
PesMapping = Mapping[str, ArrayOrScalar]

MolList = List[MultiMolecule]
MolIter = Iterable[MultiMolecule]

Key = Tuple[float, ...]


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
        return self.iter_len // self.sub_iter_len

    def acceptance(self) -> np.ndarray:
        """Create an empty 1D boolean array for holding the acceptance."""
        return np.zeros(self.sub_iter_len, dtype=bool)

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
        if len(phi.phi) != len(self.param.move_range):
            raise ValueError("'phi.phi' and 'param_mapping.move_range' "
                             "should be of the same length")
        self.iter_len = iter_len
        self.sub_iter_len = sub_iter_len

    def to_yaml_dict(  # type: ignore[override]
        self, *,
        path: str = '.',
        folder: str = 'MM_MD_workdir',
        logfile: str = 'armc.log',
        psf: Optional[Iterable[PSFContainer]] = None,
    ) -> Dict[str, Any]:
        """Convert an :class:`ARMC` instance into a .yaml readable by :class:`ARMC.from_yaml`.

        Returns
        -------
        :data:`Dict[str, Any]<typing.Dict>`
            A dictionary.

        """
        cls = type(self)
        ret: Dict[str, Any] = {}  # type: ignore[typeddict-item]
        ret['param'] = self.param.to_yaml_dict()
        ret['phi'] = self.phi.to_yaml_dict()
        ret['job'] = self.package_manager.to_yaml_dict()
        ret['job']['molecule'] = [m.properties.filename for m in self.molecule]

        ret['monte_carlo'] = {
            'type': f'{cls.__module__}.{cls.__name__}',
            'iter_len': self.iter_len,
            'sub_iter_len': self.sub_iter_len,
            'keep_files': self.keep_files,
            'path': path,
            'folder': folder,
            'logfile': logfile,
            'hdf5_file': self.hdf5_file,
        }

        # Parse the `pes` block
        pes: Dict[str, Any] = {}
        ret['pes'] = pes
        i_max = 1 + max(int(k.split('.')[-1]) for k in self.pes.keys())
        for _k, v in self.pes.items():  # type: str, functools.partial # type: ignore[assignment]
            k, _i = _k.split('.')
            i = int(_i)
            try:
                pes[k]['kwargs'][i] = v.keywords
            except KeyError:
                pes[k] = {
                    'func': f'{v.func.__module__}.{v.func.__qualname__}',
                    'kwargs': i_max * [None],
                }
                pes[k]['kwargs'][i] = v.keywords

        # Parse the `psf` block
        if psf is None:
            ret['psf'] = {'psf_file': None}
        else:
            workdir = Path(path, folder)
            if not os.path.isdir(workdir):
                os.mkdir(workdir)

            ret['psf'] = {'psf_file': []}
            psf_lst = ret['psf']['psf_file']
            for i, psf_obj in enumerate(psf):
                name = str(workdir / f'mol.{i}.psf')
                psf_lst.append(name)
                psf_obj.write(name)
        return ret

    @overload  # type: ignore[override]
    def __call__(self, *, start: None = ..., key_new: None = ...) -> None: ...
    @overload
    def __call__(self, *, start: int, key_new: Key) -> None: ...
    def __call__(self, *, start=None, key_new=None):  # noqa: E301
        """Initialize the Addaptive Rate Monte Carlo procedure."""
        key_new = self._parse_call(start, key_new)
        start_ = start if start is not None else 0

        # Start the main loop
        for kappa in range(start_, self.super_iter_len):
            acceptance = self.acceptance()
            create_xyz_hdf5(self.hdf5_file, self.molecule,
                            iter_len=self.sub_iter_len,
                            phi=self.phi.phi)

            for omega in range(self.sub_iter_len):
                key_new = self.do_inner(kappa, omega, acceptance, key_new)
            self.phi.update(acceptance)

    @overload
    def _parse_call(self, start: None = ..., key_new: None = ...) -> Key: ...
    @overload
    def _parse_call(self, start: int, key_new: Key) -> Key: ...
    def _parse_call(self, start=None, key_new=None):  # noqa: E301
        """Parse the arguments of :meth:`__call__` and prepare the first key."""
        if start is None:
            create_hdf5(self.hdf5_file, self)  # Construct the HDF5 file

            key_new = self._get_first_key()  # Initialize the first MD calculation
            if np.inf in self[key_new]:
                raise RuntimeError('One or more jobs crashed in the first ARMC iteration; '
                                   'manual inspection of the cp2k output is recomended')
            elif not self.keep_files:
                self.clear_jobs()

        elif key_new is None:
            raise TypeError("'key_new' cannot be None if 'start' is None")
        return key_new

    def do_inner(self, kappa: int, omega: int, acceptance: np.ndarray, key_old: Key) -> Key:
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
        _key_new = self._do_inner1(key_old)

        # Step 2: Calculate PES descriptors
        pes_new, mol_list = self._do_inner2()

        # Step 3: Evaluate the auxiliary error; accept if the new parameter set lowers the error
        error_change, aux_new = self._do_inner3(pes_new, key_old)
        accept = error_change < 0

        # Step 4: Update the auxiliary error history, apply phi & update job settings
        acceptance[omega] = accept
        key_new = self._do_inner4(accept, error_change, aux_new,
                                  _key_new, key_old, kappa, omega)

        # Step 5: Export the results to HDF5
        self._do_inner5(mol_list, accept, aux_new, pes_new, kappa, omega)
        return key_new

    def _do_inner1(self, key_old: Key, idx: int = 0) -> Key:
        """Perform a random move."""
        key_new = self.move(idx=idx)

        if isinstance(key_new, Exception):
            self.logger.warning(f"Recalculating move; {key_new}")
            return self._do_inner1(key_old)

        elif key_new in self:
            self.logger.info("Recalculating move; move has already been visited")
            return self._do_inner1(key_old)

        return key_new

    def _do_inner2(self) -> Tuple[PesDict, Optional[MolList]]:
        """Calculate PES-descriptors."""
        return self.get_pes_descriptors()

    def _do_inner3(self, pes_new: PesMapping, key_old: Key) -> Tuple[float, np.ndarray]:
        """Evaluate the auxiliary error; accept if the new parameter set lowers the error."""
        aux_new = self.get_aux_error(pes_new)
        aux_old = self[key_old]
        error_change = (aux_new - aux_old).sum()
        return error_change, aux_new

    def _do_inner4(self, accept: bool, error_change: float, aux_new: np.ndarray,
                   key_new: Key, key_old: Key,
                   kappa: int, omega: int) -> Key:
        """Update the auxiliary error history, apply phi & update job settings."""
        err_round = round(error_change, 4)
        aux_round = round(aux_new.sum(), 4)
        epilog = f'error_change = {err_round}; error = {aux_round}\n'

        if accept:
            self.logger.info(f"Accepting move {(kappa, omega)}: {epilog}")
            self[key_new] = self.apply_phi(aux_new)
            self.param['param_old'][:] = self.param['param']
            return key_new
        else:
            self.logger.info(f"Rejecting move {(kappa, omega)}: {epilog}")
            self[key_new] = aux_new
            self[key_old] = self.apply_phi(self[key_old])
            return key_old

    def _do_inner5(self, mol_list: Optional[MolIter], accept: bool, aux_new: np.ndarray,
                   pes_new: PesMapping, kappa: int, omega: int) -> None:
        """Export the results to HDF5."""
        self.to_hdf5(mol_list, accept, aux_new, pes_new, kappa, omega)

        not_accept = ~np.array(accept, ndmin=1, dtype=bool, copy=False)
        self.param['param'].loc[:, not_accept] = self.param['param_old'].loc[:, not_accept]

    @property
    def apply_phi(self) -> Callable[..., np.ndarray]:
        """Apply :attr:`phi` to **value**."""
        return self.phi

    def _get_first_key(self, idx: int = 0) -> Key:
        """Create a the ``history_dict`` variable and its first key.

        The to-be returned key servers as the starting argument for :meth:`.do_inner`,
        the latter method relying on both producing and requiring a key as argument.

        Parameters
        ----------
        idx : :class:`int`
            The column key for :attr:`param_mapping["param"]<ARMC.param_mapping.>`.

        Returns
        -------
        |tuple|_ [|np.ndarray|_ [|float|_]]
            A tuple of Numpy arrays.

        """
        key: Key = tuple(self.param['param'][idx].values)
        pes, _ = self.get_pes_descriptors(get_first_key=True)

        self[key] = self.get_aux_error(pes)
        self.param['param_old'][idx] = self.param['param'][idx]
        return key

    def to_hdf5(self, mol_list: Optional[MolIter],
                accept: bool, aux_new: np.ndarray,
                pes_new: PesMapping,
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
        if not isinstance(accept, abc.Iterable):
            param_key: Literal['param', 'param_old'] = 'param' if accept else 'param_old'
            aux_error_mod = np.append(self.param[param_key].values, phi)
        else:
            _aux_error_mod = [self.param['param' if acc else 'param_old'][i].values for
                              i, acc in enumerate(accept)]
            aux_error_mod = np.append(_aux_error_mod, phi)
            aux_error_mod.shape = len(self.phi), -1

        hdf5_kwarg = {
            'param': self.param['param'].values.T,
            'xyz': mol_list,
            'phi': phi,
            'acceptance': accept,
            'aux_error': aux_new,
            'aux_error_mod': aux_error_mod
        }
        hdf5_kwarg.update(pes_new)

        to_hdf5(self.hdf5_file, hdf5_kwarg, kappa, omega)

    def get_aux_error(self, pes_dict: PesMapping) -> np.ndarray:
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
            qm_pes = self.pes[key].ref  # type: ignore
            QM = np.asarray(qm_pes, dtype=float)
            MM = np.asarray(mm_pes, dtype=float)
            ret: np.ndarray = (QM - MM)**2
            return ret.sum() / QM.sum()

        length = 1 + max(int(k.rsplit('.')[-1]) for k in pes_dict.keys())

        generator = (norm_mean(k, v) for k, v in pes_dict.items())
        ret = np.fromiter(generator, dtype=float, count=len(pes_dict))
        ret.shape = length, -1
        return ret

    def restart(self) -> None:
        r"""Restart a previously started Addaptive Rate Monte Carlo procedure."""
        i, j, key, acceptance = self._restart_from_hdf5()

        # Validate the xyz .hdf5 file; create a new one if required
        xyz = _get_filename_xyz(self.hdf5_file)
        if not os.path.isfile(xyz):
            create_xyz_hdf5(self.hdf5_file, self.molecule, self.sub_iter_len, self.phi)

        # Check that both .hdf5 files can be opened; clear their status if not
        closed1 = hdf5_clear_status(xyz)
        if not closed1:
            self.logger.warning(f"Unable to open {xyz!r}, file status was forcibly reset")
        closed2 = hdf5_clear_status(self.hdf5_file)
        if not closed2:
            self.logger.warning(f"Unable to open {self.hdf5_file!r}, "
                                "file status was forcibly reset")

        # Finish the current set of sub-iterations
        j += 1
        for omega in range(j, self.sub_iter_len):
            key = self.do_inner(i, omega, acceptance, key)
        self.phi.update(acceptance)
        i += 1

        # And continue
        self(start=i, key_new=key)

    def _restart_from_hdf5(self) -> Tuple[int, int, Key, np.ndarray]:
        """Read and process the .hdf5 file for :meth:`ARMC.restart`."""
        import h5py

        with h5py.File(self.hdf5_file, 'r', libver='latest') as f:
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
            final_key: Key = self._find_restart_key(f, i)
        return i, j, final_key, acceptance

    @staticmethod
    def _find_restart_key(f: Mapping[str, np.ndarray], i: int) -> Key:
        """Construct a key for the parameter which is not ``nan``."""
        while i >= 0:
            aux_error: np.ndarray = f['aux_error'][i]
            param_old: np.ndarray = f['param'][i]
            aux_nan: np.ndarray = np.isnan(aux_error).any(axis=(1, 2))

            try:  # Its no longer np.nan
                return tuple(param_old[~aux_nan][-1])
            except IndexError:
                i -= 1
        else:
            raise RuntimeError('Not a single successful MD-calculation was found; '
                               'restarting is not possible')
