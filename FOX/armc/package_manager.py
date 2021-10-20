"""A module containing the :class:`PackageManager` class.

Index
-----
.. currentmodule:: FOX.armc
.. autosummary::
    PackageManagerABC
    PackageManager

API
---
.. autoclass:: PackageManagerABC
    :members:
.. autoclass:: PackageManager
    :members:

"""

from __future__ import annotations

import os
import shutil
import textwrap
from abc import ABC, abstractmethod
from logging import Logger
from itertools import chain, zip_longest
from collections import abc
from typing import (
    Mapping, TypeVar, Hashable, Any, KeysView, ItemsView, ValuesView, Iterator, overload,
    Union, Dict, List, Optional, Tuple, Iterable, Sequence, cast, TYPE_CHECKING
)

import numpy as np
import pandas as pd
from scm.plams import config, Molecule, JobManager  # type: ignore
from qmflows import Settings as QmSettings
from qmflows.cp2k_utils import prm_to_df
from qmflows.packages.cp2k_package import CP2K, CP2K_Result
from noodles import gather, schedule, has_scheduled_methods, run_parallel
from nanoutils import set_docstring, TypedDict

from ..classes import MultiMolecule
from ..functions.cp2k_utils import get_xyz_path
from ..logger import DummyLogger
from ..io.read_xyz import XYZError

if TYPE_CHECKING:
    from scm.plams.core.basejob import Job
    from qmflows.packages import Result, Package
    from noodles.interface import PromisedObject
else:
    from ..type_alias import PromisedObject, Result, Package, Job

__all__ = ['PackageManagerABC', 'PackageManager']


class PkgDict(TypedDict):
    """A :class:`~typing.TypedDict` representing a single job recipe."""

    type: Package
    molecule: Molecule
    settings: QmSettings


T = TypeVar('T')
RT = TypeVar('RT', bound=Result)

MolLike = Iterable[Tuple[float, float, float]]
Value = Tuple[PkgDict, ...]

#: The internal dictionary contained within :class:`PackageManagerABC`.
Data = Dict[str, Value]

DataMap = Mapping[str, Iterable[PkgDict]]
DataIter = Iterable[Tuple[str, Iterable[PkgDict]]]

JobHook = Iterator[Iterable[Result]]


class PackageManagerABC(ABC, Mapping[str, Value]):
    """A class for managing qmflows-style jobs."""

    _data: Data
    _hook: Optional[JobHook]

    def __init__(self, data: Union[DataMap, DataIter],
                 hook: Optional[JobHook] = None,
                 **kwargs: Any) -> None:
        r"""Initialize an instance.

        Parameters
        ----------
        data : :class:`~collections.abc.Mapping` [:class:`str`, :class:`~collections.abc.Iterable` [:class:`~scm.plams.core.basejob.SingleJob`]]
            A mapping with user-defined job descriptor as keys and an iterable of Job
            instances as values.
        hook : :class:`~collections.abc.Iterator` [:class:`~collections.abc.Iterable` [:class:`~qmflows.packages.packages.Result`]], optional
            An iterator yielding multiple qmflows Result objects.
            Can be used as a hook for the purpose of unit-testing.
        **kwargs : :data:`~typing.Any`
            Further keyword arguments which can be customized by :class:`PackageManagerABC` subclasses.

        See Also
        --------
        func:`evaluate_rmsd`
            Evaluate the RMSD of a geometry optimization.

        """  # noqa: E501
        if kwargs:
            name = next(iter(kwargs))
            raise TypeError(f"Unexpected argument {name!r}")
        super().__init__()

        self.data = cast(Data, data)
        self.hook = hook

    # Attributes and properties

    @property
    def hook(self) -> Optional[JobHook]:
        """Get or set the :attr:`hook` attribute."""
        return self._hook

    @hook.setter
    def hook(self, value: Optional[JobHook]) -> None:
        if value is None:
            pass
        elif not isinstance(value, abc.Iterator):
            raise TypeError("'hook' excpected an iterator; "
                            f"observed type: {value.__class__.__name__!r}")
        self._hook = value

    @property
    def data(self) -> Data:
        """A property containing this instance's underlying :class:`dict`.

        The getter will simply return the attribute's value.
        The setter will validate and assign any mapping or iterable containing of key/value pairs.

        """
        return self._data

    @data.setter
    def data(self, value: Union[DataMap, DataIter]) -> None:  # noqa
        iterable = value.items() if isinstance(value, abc.Mapping) else value
        ret = {k: tuple(v) for k, v in iterable}

        value_len = {len(v) for v in ret.values()}
        if not value:
            raise ValueError("'data' expected a non-empty Mapping")
        elif len(value_len) != 1:
            raise ValueError("All values passed to 'data' must be of the same length")

        # Ensure all settings are qmflows.Settings instances
        for job_tup in ret.values():
            for job in job_tup:
                job['settings'] = QmSettings(job['settings'])
        self._data = ret

    def __eq__(self, value: Any) -> bool:
        """Implement :code:`self == value`."""
        if type(self) is not type(value):
            return False

        iterator: Iterator[Tuple[PkgDict, PkgDict]]
        iterator = chain.from_iterable(zip_longest(v, value[k]) for k, v in self.items())
        ret = True

        try:
            for job1, job2 in iterator:
                if None in (job1, job2):
                    return False
                for k, v1 in job1.items():
                    v2 = job2[k]
                    if isinstance(v1, Molecule):
                        ret &= (np.asarray(v1) == np.asarray(v2)).all()
        except KeyError:
            return False
        else:
            return ret

    def __repr__(self) -> str:
        """Implement :code:`repr(self)` and :code:`str(self)`."""
        data = ''
        for k, v in self.items():
            end = '}, ...)' if len(v) > 1 else '},)'
            data += (
                f',\n{k!r}' +
                ': ({' +
                ', '.join(f'{_k!r}: {_v.__class__.__name__}(...)' for _k, _v in v[0].items()) +
                end
            )
        return f'{self.__class__.__name__}(' + '{\n' + textwrap.indent(data[2:], 4 * ' ') + '\n})'

    # Mapping implementation

    def __getitem__(self, key: str) -> Value:
        """Implement :code:`self[key]`."""
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        """Implement :code:`iter(self)`."""
        return iter(self.data)

    def __len__(self) -> int:
        """Implement :code:`len(self)`."""
        return len(self.data)

    def __contains__(self, key: Any) -> bool:
        """Implement :code:`key in self`."""
        return key in self.data

    def keys(self) -> KeysView[str]:
        """Return a set-like object providing a view of this instance's keys."""
        return self.data.keys()

    def items(self) -> ItemsView[str, Value]:
        """Return a set-like object providing a view of this instance's key/value pairs."""
        return self.data.items()

    def values(self) -> ValuesView[Value]:
        """Return an object providing a view of this instance's values."""
        return self.data.values()

    @overload
    def get(self, key: Hashable) -> Optional[Value]: ...
    @overload
    def get(self, key: Hashable, default: T) -> Union[Value, T]: ...
    def get(self, key, default=None):  # noqa: E301
        """Return the value for **key** if it's available; return **default** otherwise."""
        return self.data.get(key, default)  # type: ignore

    # The actual job runner

    @abstractmethod
    def __call__(
        self, logger: Optional[Logger] = None, **kwargs: Any
    ) -> Union[Tuple[None, None], Tuple[List[MultiMolecule], List[Any]]]:
        r"""Run all jobs and return a sequence of user-specified results.

        Parameters
        ----------
        logger : :class:`logging.Logger`, optional
            A logger for reporting the updated value.
        \**kwargs : :data:`~typing.Any`
            Keyword arguments which can be further customized in a sub-class.

        Returns
        -------
        :class:`~collections.abc.Sequence`, optional
            Returns ``None`` if one of the jobs crashed;
            a Sequence of user-specified objects is returned otherwise.
            The nature of the to-be returned objects should be defined in a sub-class.

        """
        raise NotImplementedError('Trying to call an abstract method')

    @staticmethod
    @abstractmethod
    def assemble_job(job: PkgDict, **kwargs: Any) -> Any:
        """Assemble a :class:`PkgDict` into an actual job."""
        raise NotImplementedError('Trying to call an abstract method')

    @abstractmethod
    def clear_jobs(self, **kwargs: Any) -> None:
        """Delete all jobs located in :attr:`_job_cache`."""
        raise NotImplementedError('Trying to call an abstract method')

    @abstractmethod
    def update_settings(self, dct: Any, **kwargs: Any) -> None:
        """Update the Settings embedded in this instance using **dct**."""
        raise NotImplementedError('Trying to call an abstract method')

    def to_yaml_dict(self) -> Dict[str, Any]:
        if self.hook is not None:
            raise NotImplementedError

        cls = type(self)
        ret: Dict[str, Any] = {
            'type': f'{cls.__module__}.{cls.__name__}',
            'molecule': [],
        }

        for k, v_tup in self.items():
            lst: List[Dict[Any, Any]] = []
            for v in v_tup:
                dct = v['settings'].as_dict()
                for k_param, v_param in list(dct.items()):
                    if isinstance(v_param, pd.DataFrame):
                        del dct[k_param]
                lst.append(dct)
            ret[k] = {'type': f'qmflows.{v["type"].pkg_name}', 'settings': lst}
        return ret


@set_docstring(PackageManagerABC.__doc__)
@has_scheduled_methods
class PackageManager(PackageManagerABC):

    def __init__(self, data: Union[DataMap, DataIter], hook: Optional[JobHook] = None) -> None:
        super().__init__(data, hook)

        # Transform all forcefield parameter blocks into DataFrames
        job_iterator = (job['settings'] for job in chain.from_iterable(self.values()))
        for settings in job_iterator:  # Type: QmSettings
            prm_to_df(settings)

    def __call__(
        self, logger: Optional[Logger] = None, n_processes: int = 1
    ) -> Union[Tuple[None, None], Tuple[List[MultiMolecule], List[Any]]]:
        r"""Run all jobs and return a sequence of list of MultiMolecules.

        Parameters
        ----------
        logger : :class:`logging.Logger`, optional
            A logger for reporting job statuses.

        Returns
        -------
        :class:`list` [:class:`FOX.MultiMolecule`], optional
            Returns ``None`` if one of the jobs crashed;
            a list of MultiMolecule is returned otherwise.

        """
        # Construct the logger
        if logger is None:
            logger = cast(Logger, DummyLogger())

        # Check if a hook has been specified
        if self.hook is not None:
            results = next(self.hook)
            return self._extract_mol(results, logger)

        jobs_iter = iter(self.items())

        name, jobs = next(jobs_iter)
        promised_jobs: List[PromisedObject] = [self.assemble_job(j, name=name) for j in jobs]

        for name, jobs in jobs_iter:
            promised_jobs = [self.assemble_job(j, p_j, name=name) for
                             j, p_j in zip(jobs, promised_jobs)]

        results = run_parallel(gather(*promised_jobs), n_threads=n_processes)
        return self._extract_mol(results, logger)

    @staticmethod
    @schedule
    def assemble_job(
        job: PkgDict,
        old_results: Optional[Result] = None,
        name: Optional[str] = None,
    ) -> PromisedObject:
        """Create a :class:`PromisedObject` from a qmflow :class:`Package` instance."""
        job_name = name if name is not None else ''
        obj_type = job['type']
        settings = job['settings']

        if old_results is None:
            mol = job['molecule']
        else:
            mol = old_results.geometry

        if isinstance(obj_type, CP2K) and isinstance(old_results, CP2K_Result):
            try:
                lattice: np.ndarray[Any, np.dtype[np.float64]] = old_results.lattice
                assert lattice is not None
            except (AssertionError, FileNotFoundError):
                pass
            else:
                settings = settings.copy()
                settings.cell_parameters = lattice[-1].tolist()

        return obj_type(mol=mol, job_name=job_name, validate_output=False, settings=settings)

    @staticmethod
    def clear_jobs() -> None:
        """Delete all jobs."""
        job_manager: JobManager = config.default_jobmanager
        workdir: Union[str, os.PathLike] = job_manager.workdir

        for job in job_manager.jobs:  # type: Job
            name = os.path.join(workdir, job.name)
            try:
                shutil.rmtree(name)
            except FileNotFoundError:
                pass

        job_manager.jobs = []
        job_manager.names = {}

    def update_settings(self, dct: Sequence[Tuple[str, Mapping]], new_keys: bool = True) -> None:
        """Update all forcefield parameter blocks in this instance's CP2K settings."""
        iterator = (job['settings'] for job in chain.from_iterable(self.values()))
        for settings in iterator:
            for key_alias, sub_dict in dct:
                param = sub_dict['param']

                if key_alias not in settings:
                    settings[key_alias] = pd.DataFrame(sub_dict, index=[param])
                    continue

                # Ensure all column-keys in **sub_dict** are also in **df**
                df: pd.DataFrame = settings[key_alias]
                if new_keys:
                    keys = set(sub_dict.keys()).difference(df.columns)
                    for k in keys:
                        df[k] = np.nan

                # Ensure that the **param** index-key is in **df** and update
                df_update = pd.DataFrame(sub_dict, index=[param])
                if param not in df.index:
                    df.loc[param] = np.nan
                if 'guess' in df.columns:
                    del df['guess']
                df.update(df_update)

    @overload
    @staticmethod
    def _extract_mol(results: None, logger: Logger) -> Tuple[None, None]: ...
    @overload  # noqa: E301
    @staticmethod
    def _extract_mol(
        results: Iterable[RT], logger: Logger
    ) -> Tuple[None, None] | Tuple[List[MultiMolecule], List[RT]]: ...
    @staticmethod  # noqa: E301
    def _extract_mol(
        results: None | Iterable[RT],
        logger: Logger,
    ) -> Tuple[None, None] | Tuple[List[MultiMolecule], List[RT]]:
        """Create a list of MultiMolecule from the passed **results**."""
        # `noodles.run_parallel()` can return `None` under certain circumstances
        if results is None:
            return None, None

        mol_list = []
        results_list = list(results)
        for result in results_list:
            if result.status in {'failed', 'crashed'}:
                return None, None

            try:
                lattice: None | np.ndarray[Any, np.dtype[np.float64]] = result.lattice
                assert lattice is not None
            except (AssertionError, FileNotFoundError):
                lattice = None

            try:  # Construct and return a MultiMolecule object
                path: str = get_xyz_path(result.archive['work_dir'])  # type: ignore
                mol = MultiMolecule.from_xyz(path)
                mol.lattice = lattice
                mol.round(3, inplace=True)
                mol_list.append(mol)
            except XYZError:  # The .xyz file is unreadable for some reason
                logger.warning(f"Failed to parse {path!r}")
                return None, None
            except Exception as ex:
                logger.warning(ex)
                return None, None
        return mol_list, results_list
