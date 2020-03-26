import os
import copy
from abc import ABC, abstractmethod
from logging import Logger
from inspect import signature
from functools import wraps
from itertools import repeat
from collections import abc
from typing import (Mapping, TypeVar, Hashable, Any, KeysView, ItemsView, ValuesView, Iterator,
                    Union, Dict, List, Optional, FrozenSet, Callable, Type, Set, Tuple,
                    Iterable, overload, Sequence, Collection, TYPE_CHECKING)

from scm.plams import SingleJob
from assertionlib.dataclass import AbstractDataClass

from ..functions.cp2k_utils import get_xyz_path
from ..logger import DummyLogger
from ..type_hints import Literal
from ..io.read_xyz import XYZError

if TYPE_CHECKING:
    from .multi_mol import MultiMolecule
else:
    MultiMolecule = f'{__package__}.multi_mol.MultiMolecule'

__all__ = ['WorkflowManager']

_KT = TypeVar('_KT')
KT = TypeVar('KT', bound=str)
JT = TypeVar('JT', bound=SingleJob)
T = TypeVar('T')

ExtractNames = Callable[[type, Type[JT]], FrozenSet[str]]
MolLike = Iterable[Tuple[float, float, float]]

DataMap = Mapping[KT, Iterable[JT]]
DataIter = Iterable[Tuple[KT, Iterable[JT]]]
PostProcess = Callable[[str, Iterable[JT], Iterable[MultiMolecule], Logger], bool]
PostProcessMap = Mapping[KT, PostProcess]
PostProcessIter = Iterable[Tuple[KT, PostProcess]]


def cache_return(func: ExtractNames) -> ExtractNames:
    """A caching decorator for :meth:`WorkflowManagerABC._extract_names`.

    Caches the signatures of all passed :class:`~scm.plams.core.basejob.SingleJob` types.

    """
    cache: Dict[Type[SingleJob], FrozenSet[str]] = {}

    @wraps(func)
    def wrapper(cls, job_cls, blacklist):
        try:
            return cache[job_cls]
        except KeyError:
            cache[job_cls] = ret = func(cls, job_cls, blacklist)
            return ret
    return wrapper


class WorkflowManagerABC(AbstractDataClass, ABC, Mapping[KT, Tuple[JT, ...]]):

    def __init__(self, data: Union[DataMap, DataIter],
                 post_process: Union[None, PostProcessMap, PostProcessIter]) -> None:
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
        self.post_process = post_process

    # Attributes and properties

    @property
    def post_process(self) -> Dict[KT, PostProcess]:
        """A propery containing a dictionary for post processing jobs.

        The getter will simply return the attribute's value.
        The setter will validate and assign any mapping or iterable containing of key/value pairs.

        """
        return self._post_process

    @post_process.setter
    def post_process(self, value: Union[None, PostProcessMap, PostProcessIter]) -> None:
        if value is None:
            self._post_process: Dict[KT, PostProcess] = {}
            return

        dct = self._parse_dict_like(value, to_tuple=False)

        # Check that the keys post_process are a subset of those in self
        if not set(dct).issubset(self.keys()):
            illegal_keys = set(self.keys()).difference(dct)
            keys_str = ' '.join(repr(k) for k in illegal_keys)
            raise KeyError(f"{'post_process'!r} got one or more unexpected keys: {keys_str}")
        else:
            self._post_process = dct
        return

    @property
    def _data(self) -> Dict[KT, Tuple[JT, ...]]:
        """A (private) property containing this instance's underlying :class:`dict` or :class:`~collections.OrderedDict`.

        The getter will simply return the attribute's value.
        The setter will validate and assign any mapping or iterable containing of key/value pairs.

        """  # noqa
        return self.__data

    @_data.setter
    def _data(self, value: Union[DataMap, DataIter]) -> None:  # noqa
        ret = self._parse_dict_like(value, to_tuple=True)

        value_len = {len(v) for v in ret.values()}
        if not value:
            raise ValueError(f"{self.__class__.__name__!r}() expected a non-empty Mapping")
        elif len(value_len) != 1:
            raise ValueError(f"All values passed to {self.__class__.__name__!r}() "
                             "must be of the same length")
        self.__data = ret

    @overload
    @classmethod
    def _parse_dict_like(cls, value: Union[Mapping[_KT, T], Iterable[Tuple[_KT, T]]],
                         to_tuple: Literal[False]) -> Dict[_KT, T]: ...

    @overload
    @classmethod
    def _parse_dict_like(cls, value: Union[Mapping[_KT, Iterable[T]], Iterable[Tuple[_KT, Iterable[T]]]],  # noqa
                         to_tuple: Literal[True]) -> Dict[_KT, Tuple[T, ...]]: ...

    @classmethod
    def _parse_dict_like(cls, value, to_tuple=False):
        iterable = value.items() if isinstance(value, abc.Mapping) else value
        if to_tuple:
            return {k: tuple(v) for k, v in iterable}
        else:
            return dict(iterable)

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

    @abstractmethod
    @staticmethod
    def run_job(job: JT, logger: Logger, **kwargs: Any) -> Optional[T]:
        r"""Run a single **job** and return a user-specified result.

        Parameters
        ----------
        job : :class:`~scm.plams.core.basejob.SingleJob`
            The to-be run job.
        logger : :class:`logging.Logger`
            A logger for reporting the updated value.
        \**kwargs : :data:`~typing.Any`
            Keyword arguments which can be further customized in a sub-class.

        Returns
        -------
        :class:`~collections.abc.Sequence`, optional
            Returns ``None`` if the job crashed; a user-specified object is returned otherwise.
            The nature of the to-be returned objects should be defined in a sub-class.

        """
        raise NotImplementedError('Trying to call an abstract method')

    def reset_jobs(self) -> None:
        """Replace all Jobs stored in this instance with freshly created Job instances."""
        for k, job_tup in self.items():
            self[k] = tuple(self._copy_job(job) for job in job_tup)

    def _copy_job(self, job: JT) -> JT:
        """Create shallow-ish copy of the passed job.

        Parameters
        ----------
        job : :class:`~scm.plams.core.basejob.SingleJob`
            A PLAMS Job.

        Returns
        -------
        :class:`~scm.plams.core.basejob.SingleJob`
            A new Job instance constructed from **job**.
            All arguments passed to
            :meth:`<job.__init__()>scm.plams.core.basejob.SingleJob.__init__`
            are shallow copies of their respective counterparts in **job**.

        """
        job_type = type(job)

        kwarg_names = self._extract_names(job_type)
        kwargs = {}

        for name in kwarg_names:
            try:
                kwargs[name] = copy.copy(getattr(job, name))
            except AttributeError:
                pass
        return job_type(**kwargs)

    @cache_return
    @classmethod
    def _extract_names(cls, job_type: Type[JT],
                       blacklist: Collection[str] = ('self', 'kwargs', 'depend')
                       ) -> FrozenSet[str]:
        """Return a set with all argument names to-be passed to :meth:`job_type.__init__`.

        Searches recursivelly through all subclasses of **job_type**
        (until :class:`object` is reached) to ensure the returned set is set.

        Parameters
        ----------
        job_type : :class:`type` [:class:`~scm.plams.core.basejob.SingleJob`]
            A PLAMS Job type.
        blacklist : :class:`~collections.abc.Collection` [:class:`str`]
            An collection with to-be ignored parameter names.
            These names will be removed from the returned set if necessary.

        Returns
        -------
        :class:`frozenset` [:class:`set`]
            A set with parameter names as extracted from **job_type**.

        """
        ret: Set[str] = set()

        def dfs(obj_type: type) -> None:
            if obj_type is object:
                return
            sgn = signature(obj_type)
            ret.update(sgn.parameters.keys())
            for base in obj_type.__bases__:
                dfs(base)

        dfs(job_type)
        for name in blacklist:
            ret.remove(name)
        return frozenset(ret)


class WorkflowManager(WorkflowManagerABC):

    @WorkflowManagerABC.inherit_annotations()
    def __init__(self, data, post_process=None):
        super().__init__(data, post_process)

    def __call__(self, logger: Optional[Logger] = None) -> Optional[List[MultiMolecule]]:
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
        self.reset_jobs()
        if logger is None:
            logger = DummyLogger()

        mol_list = repeat((None,))
        for name, jobs in self.items():
            mol_list = [self.run_job(job, logger, mol[-1]) for mol, job in zip(mol_list, jobs)]
            if None in mol_list:
                logger.info(f"One or more {name!r} jobs have failed; aborting further jobs")
                return None

            func = self.post_process.get(name)
            if func is None:
                continue
            condition = func(name, jobs, mol_list, logger)
            if not condition:
                return None
            logger.info(f"All {name!r} jobs were successful")

        return mol_list

    @staticmethod
    def run_job(job: JT, logger: Logger,
                molecule: Optional[MolLike] = None) -> Optional[MultiMolecule]:
        r"""Run a single **job** and return a MultiMolecule if the **job** doesn't crash.

        Parameters
        ----------
        job : :class:`~scm.plams.core.basejob.SingleJob`
            The to-be run job.
        logger : :class:`logging.Logger`
            A logger for reporting job statuses.
        molecule : :class:`~FOX.classes.multi_mol.MultiMolecule` or :class:`~scm.plams.mol.molecule.Molecule`
            A Molecule whose Cartesian coordinates will be superimposed on
            :attr:`job.molecule<scm.plams.core.basejob.SingleJob>`.

        Returns
        -------
        :class:`~FOX.classes.multi_mol.MultiMolecule`, optional
            Returns ``None`` if the job crashed; a MultiMolecule is returned otherwise.

        """  # noqa
        if molecule is not None:
            job.molecule.from_array(molecule)

        try:
            results = job.run()
        except FileNotFoundError:
            return None  # Precaution against PLAMS unpickling old Jobs that don't exist
        if job.status in {'crashed', 'failed'}:
            return None

        try:  # Construct and return a MultiMolecule object
            path = get_xyz_path(results)
            mol = MultiMolecule.from_xyz(path)
            mol.round(3, inplace=True)
            return mol
        except XYZError:  # The .xyz file is unreadable for some reason
            logger.warning(f"Failed to parse ...{os.sep}{os.path.basename(path)}")
            return None
        except FileNotFoundError as ex:
            logger.warn(str(ex))
            return None

    @staticmethod
    def evaluate_rmsd(name: str, jobs: Iterable[JT], mol_list: Iterable[MultiMolecule],
                    logger: Logger, threshold: Optional[float] = None) -> bool:
        """Evaluate the RMSD of a geometry optimization.

        If the root mean square displacement (RMSD) of the last frame is
        larger than **threshold** then ``False`` will be returned;
        ``True`` is returned otherwise.

        Parameters
        ----------
        name : :class:`str`
            The name of the job set.
        job_list : :class:`~collections.abc.Iterable` [:class:`~scm.plams.core.basejob.SingleJob`]
            An iterable of Jobs.
        mol_list : :class:`~collections.abc.Iterable` [:class:`~FOX.classes.multi_mol.MultiMolecule`]
            An iterable consisting of MultiMolecules.
            The last frame of each molecule is compared to the first one when determining the RMSD.
        logger : :class:`~logging.Logger`
            A logger for reporting results.
        threshold : :class:`float`, optional
            An RMSD threshold in Angstrom.
            Determines whether or not a given RMSD will return ``True`` or ``False``.

        Returns
        -------
        :class:`bool`
            Return ``False`` if the RMSD is larger than **threshold** and ``True`` otherwise.

        """
        if threshold is None:
            return True

        for i, mol in enumerate(mol_list):
            rmsd = mol.get_rmsd(mol_subset=-1)
            if rmsd > threshold:
                logger.warning(f'RMSD too large for {name} job {i!r}: {rmsd!r} > {threshold!r}')
                return False
        return True


WorkflowManager.__doc__ = WorkflowManagerABC.__doc__
