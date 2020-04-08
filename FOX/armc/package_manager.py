from __future__ import annotations

import os
import shutil
import textwrap
from abc import ABC, abstractmethod
from logging import Logger
from itertools import chain, zip_longest
from collections import abc
from typing import (
    Mapping, TypeVar, Hashable, Any, KeysView, ItemsView, ValuesView, Iterator,
    Union, Dict, List, Optional, Tuple, Iterable, Sequence, cast, TYPE_CHECKING
)

import numpy as np
import pandas as pd
from scm.plams import config, Molecule, JobManager
from qmflows import Settings as QmSettings
from qmflows.cp2k_utils import prm_to_df
from noodles import gather, schedule, has_scheduled_methods, run_parallel

from ..classes.multi_mol import MultiMolecule
from ..functions.cp2k_utils import get_xyz_path
from ..logger import DummyLogger
from ..type_hints import TypedDict
from ..io.read_xyz import XYZError

if TYPE_CHECKING:
    from scm.plams.core.basejob import Job
    from qmflows.packages import Result, Package
    from noodles.interface import PromisedObject
else:
    from ..type_alias import PromisedObject, Result, Package, Job

__all__ = ['PackageManager']


class JobMapping(TypedDict):
    """A :class:`~typing.TypedDict` representing a single job recipe."""

    type: Package
    molecule: Molecule
    settings: QmSettings


T = TypeVar('T')

MolLike = Iterable[Tuple[float, float, float]]

#: The internal dictionary contained within :class:`PackageManagerABC`.
Data = Dict[str, Tuple[JobMapping, ...]]

DataMap = Mapping[str, Iterable[JobMapping]]
DataIter = Iterable[Tuple[str, Iterable[JobMapping]]]


class PackageManagerABC(ABC, Mapping[str, Tuple[JobMapping, ...]]):

    __data: Data

    def __init__(self, data: Union[DataMap, DataIter]) -> None:
        """Initialize an instance.

        Parameters
        ----------
        data : :class:`~collections.abc.Mapping` [:class:`str`, :class:`~collections.abc.Iterable` [:class:`~scm.plams.core.basejob.SingleJob`]]
            A mapping with user-defined job descriptor as keys and an iterable of Job
            instances as values.
        post_process :class:`~collections.abc.Mapping` [:class:`str`, :class:`~collections.abc.Callable`], optional
            A mapping with user-specified post processing functions;
            its keys must be a subset of those in **data**.
            Setting this value to ``None`` will disable any post-processing.

        See Also
        --------
        func:`evaluate_rmsd`
            Evaluate the RMSD of a geometry optimization.

        """  # noqa
        super().__init__()
        self._data = cast(Data, data)

    # Attributes and properties

    @property
    def _data(self) -> Data:
        """A (private) property containing this instance's underlying :class:`dict`.

        The getter will simply return the attribute's value.
        The setter will validate and assign any mapping or iterable containing of key/value pairs.

        """
        return self.__data

    @_data.setter
    def _data(self, value: Union[DataMap, DataIter]) -> None:  # noqa
        iterable = value.items() if isinstance(value, abc.Mapping) else value
        ret = {k: tuple(v) for k, v in iterable}

        value_len = {len(v) for v in ret.values()}
        if not value:
            raise ValueError(f"{self.__class__.__name__!r}() expected a non-empty Mapping")
        elif len(value_len) != 1:
            raise ValueError(f"All values passed to {self.__class__.__name__!r}() "
                             "must be of the same length")

        # Ensure all settings are qmflows.Settings instances
        for job_tup in ret.values():
            for job in job_tup:
                job['settings'] = QmSettings(job['settings'])
        self.__data = ret

    def __eq__(self, value: Any) -> bool:
        """Implement :code:`self == value`."""
        if type(self) is not type(value):
            return False

        iterator: Iterator[Tuple[JobMapping, JobMapping]]
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
                repr(k) +
                ': ({' +
                ', '.join(f'{_k!r}: {_v.__class__.__name__}(...)' for _k, _v in v[0].items()) +
                end
            )
        return f'{self.__class__.__name__}(' + '{\n' + textwrap.indent(data, 4 * ' ') + '\n})'

    # Mapping implementation

    def __getitem__(self, key: str) -> Tuple[JobMapping, ...]:
        """Implement :code:`self[key]`."""
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        """Implement :code:`iter(self)`."""
        return iter(self._data)

    def __len__(self) -> int:
        """Implement :code:`len(self)`."""
        return len(self._data)

    def __contains__(self, key: Any) -> bool:
        """Implement :code:`key in self`."""
        return key in self._data

    def keys(self) -> KeysView[str]:
        """Return a set-like object providing a view of this instance's keys."""
        return self._data.keys()

    def items(self) -> ItemsView[str, Tuple[JobMapping, ...]]:
        """Return a set-like object providing a view of this instance's key/value pairs."""
        return self._data.items()

    def values(self) -> ValuesView[Tuple[JobMapping, ...]]:
        """Return an object providing a view of this instance's values."""
        return self._data.values()

    def get(self, key: Hashable, default: T = None) -> Union[Tuple[JobMapping, ...], T]:
        """Return the value for **key** if it's available; return **default** otherwise."""
        return self._data.get(key, default)  # type: ignore

    # The actual job runner

    @abstractmethod
    def __call__(self, logger: Optional[Logger] = None, **kwargs: Any) -> Optional[Sequence]:
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
    def assemble_job(job: JobMapping, **kwargs: Any) -> Any:
        """Assemble a :class:`JobMapping` into an actual job."""
        raise NotImplementedError('Trying to call an abstract method')

    @abstractmethod
    def clear_jobs(self, **kwargs: Any) -> None:
        """Delete all jobs located in :attr:`_job_cache`."""
        raise NotImplementedError('Trying to call an abstract method')

    @abstractmethod
    def update_settings(self, dct: Any, **kwargs: Any) -> None:
        """Update the Settings embedded in this instance using **dct**."""
        raise NotImplementedError('Trying to call an abstract method')


@has_scheduled_methods
class PackageManager(PackageManagerABC):

    def __init__(self, data):
        super().__init__(data)

        # Transform all forcefield parameter blocks into DataFrames
        job_iterator = (job['settings'] for job in chain.from_iterable(self.values()))
        for settings in job_iterator:  # type: QmSettings
            prm_to_df(settings)

    def __call__(self, logger: Optional[Logger] = None,
                 n_processes: int = 1) -> Optional[List[MultiMolecule]]:
        r"""Run all jobs and return a sequence of list of MultiMolecules.

        Parameters
        ----------
        logger : :class:`logging.Logger`, optional
            A logger for reporting job statuses.

        Returns
        -------
        :class:`list` [:class:`~FOX.classes.multi_mol.MultiMolecule`], optional
            Returns ``None`` if one of the jobs crashed;
            a list of MultiMolecule is returned otherwise.

        """
        # Construct the logger
        if logger is None:
            logger = DummyLogger()

        jobs_iter = iter(self.items())
        assemble_job = self.assemble_job

        _name, _jobs = next(jobs_iter)
        promised_jobs: List[PromisedObject] = [assemble_job(j, name=_name) for j in _jobs]

        for name, jobs in jobs_iter:
            promised_jobs = [assemble_job(j, p_j, name=name) for
                             j, p_j in zip(jobs, promised_jobs)]

        results: List[Result] = run_parallel(gather(*promised_jobs), n_threads=n_processes)
        return self._extract_mol(results, logger)

    @staticmethod
    @schedule
    def assemble_job(job: JobMapping, old_results: Optional[Result] = None,
                     name: Optional[str] = None) -> PromisedObject:
        """Create a :class:`PromisedObject` from a qmflow :class:`Package` instance."""
        kwargs: JobMapping = job.copy()

        if old_results is not None:
            mol: Molecule = old_results.geometry
        else:
            mol = kwargs['molecule']

        job_name: str = name if name is not None else ''
        obj_type: Package
        obj_type = kwargs.pop('type')  # type: ignore
        return obj_type(mol=mol, job_name=job_name, **kwargs)

    @staticmethod
    def clear_jobs() -> None:
        """Delete all jobs."""
        job_manager: JobManager = config.default_jobmanager
        workdir: Union[str, os.PathLike] = job_manager.workdir

        for job in job_manager.jobs:  # type: Job
            name = os.path.join(workdir, job.name)
            try:
                shutil.rmtree(name)  # type: ignore
            except (TypeError, FileNotFoundError):
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
                df.update(df_update)

    @staticmethod
    def _extract_mol(results: Iterable[Result], logger: Logger) -> Optional[List[MultiMolecule]]:
        """Create a list of MultiMolecule from the passed **results**."""
        ret: List[MultiMolecule] = []
        for result in results:
            try:  # Construct and return a MultiMolecule object
                path = get_xyz_path(result.archive['work_dir'])
                mol = MultiMolecule.from_xyz(path)
                mol.round(3, inplace=True)
                ret.append(mol)

            except XYZError:  # The .xyz file is unreadable for some reason
                logger.warning(f"Failed to parse {path!r}")
                return None

            except Exception as ex:
                logger.warning(ex)
                return None
        return ret


PackageManager.__doc__ = PackageManagerABC.__doc__
