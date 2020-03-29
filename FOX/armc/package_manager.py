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

from qmflows.packages import registry as pkg_registry
from noodles import gather
from noodles.run.threading.sqlite3 import run_parallel

from ..functions.cp2k_utils import get_xyz_path
from ..logger import DummyLogger
from ..type_hints import Literal
from ..io.read_xyz import XYZError

if TYPE_CHECKING:
    from ..classes.multi_mol import MultiMolecule
    from qmflows.packages import Result
    from noodles.interface import PromisedObject
    from noodles.serial import Registry
else:
    from ..type_alias import MultiMolecule, PromisedObject, Registry, Result

__all__ = ['PackageManager']

_KT = TypeVar('_KT')
KT = TypeVar('KT', bound=str)
JT = TypeVar('JT', bound=PromisedObject)
T = TypeVar('T')

MolLike = Iterable[Tuple[float, float, float]]

DataMap = Mapping[KT, Iterable[JT]]
DataIter = Iterable[Tuple[KT, Iterable[JT]]]

PostProcess = Callable[[str, Iterable[JT], Iterable[MultiMolecule], Logger], bool]
PostProcessMap = Mapping[KT, PostProcess]
PostProcessIter = Iterable[Tuple[KT, PostProcess]]


class PackageManagerABC(ABC, Mapping[KT, Tuple[JT, ...]]):

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
        self.post_process = None

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
        """A (private) property containing this instance's underlying :class:`dict`.

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


class PackageManager(PackageManagerABC):

    def __init__(self, data, post_process=None):
        super().__init__(data, post_process)

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

        jobs_iter = iter(self.values())

        jobs = next(jobs_iter)
        promised_jobs = (j(molecule=None) for j in jobs)
        for jobs in jobs_iter:
            promised_jobs = (j(molecule=j_old.molecule) for j, j_old in zip(jobs, promised_jobs))

        results = run_parallel(gather(*promised_jobs),
                               db_file='cache.db',
                               n_threads=n_processes,
                               always_cache=always_cache,
                               registry=registry)
        return self._extract_mol(results, logger)

    @staticmethod
    def _extract_mol(results: List['Result'], logger: Logger) -> Optional[List[MultiMolecule]]:
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
