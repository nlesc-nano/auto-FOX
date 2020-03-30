import os
from abc import ABC, abstractmethod
from logging import Logger
from itertools import repeat
from collections import abc
from typing import (Mapping, TypeVar, Hashable, Any, KeysView, ItemsView, ValuesView, Iterator,
                    Union, Dict, List, Optional, FrozenSet, Callable, Type, Set, Tuple,
                    Iterable, overload, Sequence, Collection, TYPE_CHECKING)

from scm.plams import config, Settings, Molecule
from qmflows import Settings as QmSettings
from qmflows.packages import registry as pkg_registry
from noodles import gather, schedule
from noodles.run.threading.sqlite3 import run_parallel

from ..functions.cp2k_utils import get_xyz_path
from ..logger import DummyLogger
from ..type_hints import Literal, TypedDict
from ..io.read_xyz import XYZError

if TYPE_CHECKING:
    from ..classes.multi_mol import MultiMolecule
    from qmflows.packages import Result, Package
    from noodles.interface import PromisedObject
    from noodles.serial import Registry
else:
    from ..type_alias import MultiMolecule, PromisedObject, Registry, Result, Package

__all__ = ['PackageManager']


class JobMapping(TypedDict, total=False):

    type: Package
    name: str
    molecule: Molecule
    settings: Settings


_KT = TypeVar('_KT')
KT = TypeVar('KT', bound=str)
RT = TypeVar('RT', bound=Result)
JT = TypeVar('JT', bound=JobMapping)
T = TypeVar('T')

MolLike = Iterable[Tuple[float, float, float]]

DataMap = Mapping[KT, Iterable[JT]]
DataIter = Iterable[Tuple[KT, Iterable[JT]]]

DictLike = Union[Mapping[_KT, Iterable[T]], Iterable[Tuple[_KT, Iterable[T]]]]


class PackageManagerABC(ABC, Mapping[KT, Tuple[JT, ...]]):

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
        self._data = data

    # Attributes and properties

    @property
    def workdir(self) -> str:
        """Get the path to the current PLAMS workdir."""
        return config.default_jobmanager.workdir

    @property
    def _data(self) -> Dict[KT, Tuple[JT, ...]]:
        """A (private) property containing this instance's underlying :class:`dict`.

        The getter will simply return the attribute's value.
        The setter will validate and assign any mapping or iterable containing of key/value pairs.

        """  # noqa
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
        self.__data = ret

    # Mapping implementation

    def __getitem__(self, key: KT) -> Tuple[JT, ...]:
        """Implement :code:`self[key]`."""
        return self._data[key]

    def __iter__(self) -> Iterator[KT]:
        """Implement :code:`iter(self)`."""
        return iter(self._data)

    def __len__(self) -> int:
        """Implement :code:`len(self)`."""
        return len(self._data)

    def __contains__(self, key: Any) -> bool:
        """Implement :code:`key in self`."""
        return key in self._data

    def keys(self) -> KeysView[KT]:
        """Return a set-like object providing a view of this instance's keys."""
        return self._data.keys()

    def items(self) -> ItemsView[KT, Tuple[JT, ...]]:
        """Return a set-like object providing a view of this instance's key/value pairs."""
        return self._data.items()

    def values(self) -> ValuesView[Tuple[JT, ...]]:
        """Return an object providing a view of this instance's values."""
        return self._data.values()

    def get(self, key: Hashable, default: T = None) -> Union[Tuple[JT, ...], T]:  # type: ignore
        """Return the value for **key** if it's available; return **default** otherwise."""
        return self._data.get(key, default)  # type: ignore

    # The actual job runner

    @abstractmethod
    def __call__(self, logger: Optional[Logger] = None, **kwargs: Any) -> Optional[Sequence[T]]:
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
    def assemble_job(job: JobMapping, **kwargs: Any) -> T:
        """Assemble a :class:`JobMapping` into an actual job."""
        raise NotImplementedError('Trying to call an abstract method')


class PackageManager(PackageManagerABC):

    def __init__(self, data):
        super().__init__(data)

    def __call__(self, logger: Optional[Logger] = None,
                 n_processes: int = 1,
                 always_cache: bool = True,
                 registry: Callable[[], Registry] = pkg_registry) -> Optional[List[MultiMolecule]]:
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

        name, jobs = next(jobs_iter)
        promised_jobs = (assemble_job(j, name=name) for j in jobs)
        for name, jobs in jobs_iter:
            promised_jobs = (assemble_job(*args, name=name) for args in zip(jobs, promised_jobs))

        results = run_parallel(gather(*promised_jobs),
                               db_file=os.path.join(self.workdir, 'cache.db'),
                               n_threads=n_processes,
                               always_cache=always_cache,
                               registry=pkg_registry)
        return self._extract_mol(results, logger)

    @schedule
    @staticmethod
    def assemble_job(job: JobMapping, old_job: Optional[PromisedObject] = None,
                     name: Optional[str] = None) -> PromisedObject:
        """Create a :class:`PromisedObject` from a qmflow :class:`Package` instance."""
        kwargs = job.copy()

        mol = old_job.molecule if old_job is not None else kwargs['molecule']
        job_name = name if name is not None else ''
        settings = QmSettings(kwargs.pop('settings'))

        obj_type = kwargs.pop('type')
        return obj_type(settings=settings, mol=mol, job_name=job_name, **kwargs)

    @staticmethod
    def _extract_mol(results: List[Result], logger: Logger) -> Optional[List[MultiMolecule]]:
        """Create a list of MultiMolecule from the passed **results**."""
        ret = []
        for result in results:
            try:  # Construct and return a MultiMolecule object
                path = get_xyz_path(result)
                mol = MultiMolecule.from_xyz(path)
                mol.round(3, inplace=True)
                ret.append(mol)
            except XYZError:  # The .xyz file is unreadable for some reason
                logger.warning(f"Failed to parse ...{os.sep}{os.path.basename(path)}")
                return None
            except Exception as ex:
                logger.warn(f'{ex.__class__.__name__}: {ex}')
                return None
        return ret


PackageManager.__doc__ = PackageManagerABC.__doc__
