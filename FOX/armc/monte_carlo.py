"""A module for performing Monte Carlo-based forcefield parameter optimizations.

Index
-----
.. currentmodule:: FOX.armc
.. autosummary::
    MonteCarloABC

API
---
.. autoclass:: MonteCarloABC
    :members:

"""

from __future__ import annotations

import copy
import warnings
from os import PathLike
from abc import ABC, abstractmethod
from logging import Logger
from functools import wraps, partial
from itertools import repeat, cycle, chain
from collections import abc
from typing import (
    Tuple, List, Dict, Optional, Union, Iterable, Hashable, Iterator, Any, Mapping, Callable,
    KeysView, ValuesView, ItemsView, Sequence, TypeVar, overload, TYPE_CHECKING, cast, ClassVar
)

import numpy as np
from assertionlib.dataclass import AbstractDataClass
from nanoutils import EMPTY_MAPPING
from qmflows.packages import Result
from qmflows.warnings_qmflows import QMFlows_Warning

from ..logger import DEFAULT_LOGGER
from ..type_hints import ArrayOrScalar

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from .package_manager import PackageManager
    from .param_mapping import ParamMapping
    from ..classes import MultiMolecule
else:
    from ..type_alias import PackageManager, ParamMapping, MultiMolecule

__all__ = ['MonteCarloABC']

T = TypeVar('T')

PostProcess = Callable[[Optional[Iterable[MultiMolecule]], Optional['MonteCarloABC']], None]
GetPesDescriptor = Callable[[MultiMolecule, Result], ArrayOrScalar]
Key = Tuple[float, ...]


def _template_func1(_func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    _mol, _, *args = args
    return _func(_mol, *args, **kwargs)


def _template_func2(_func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    _, _result, *args = args
    return _func(_result, *args, **kwargs)


class MonteCarloABC(AbstractDataClass, ABC, Mapping[Key, np.ndarray]):
    r"""The base :class:`.MonteCarloABC` class."""

    param: ParamMapping
    package_manager: PackageManager
    keep_files: bool
    hdf5_file: Union[str, 'PathLike[str]']
    pes: Dict[str, GetPesDescriptor]
    pes_validation: Dict[str, GetPesDescriptor]
    swap_phi: Optional[Callable[..., Any]]
    HAS_LOOP: ClassVar[bool] = False

    @property
    def molecule(self) -> Tuple[MultiMolecule, ...]:
        """Get **value** or set **value** as a tuple of MultiMolecule instances."""
        return self._molecule

    @molecule.setter
    def molecule(self, value: Iterable[MultiMolecule]) -> None:
        self._molecule = tuple(value)

    @property
    def logger(self) -> Logger:
        """Get or set the logger."""
        return self._logger

    @logger.setter
    def logger(self, value: Optional[Logger]) -> None:
        if value is not None:
            self._logger = value
        else:
            self._logger = DEFAULT_LOGGER

    @property
    def pes_post_process(self) -> Tuple[PostProcess, ...]:
        """Get or set post-processing functions."""
        return self._pes_post_process

    @pes_post_process.setter
    def pes_post_process(self, value: Optional[Iterable[PostProcess]]) -> None:
        if value is not None:
            self._pes_post_process = tuple(value)
        else:
            self._pes_post_process = ()

    def __init__(self, molecule: Iterable[MultiMolecule],
                 package_manager: PackageManager,
                 param: ParamMapping,
                 keep_files: bool = False,
                 hdf5_file: Union[str, PathLike] = 'armc.hdf5',
                 logger: Optional[Logger] = None,
                 pes_post_process: Optional[Iterable[PostProcess]] = None,
                 **kwargs: Any) -> None:
        """Initialize a :class:`MonteCarlo` instance."""
        if kwargs:
            name = next(iter(kwargs))
            raise TypeError(f"Unexpected argument {name!r}")
        super().__init__()

        self.param = param

        # Settings for running the actual MD calculations
        self.molecule = cast(Tuple[MultiMolecule, ...], molecule)
        self.package_manager = package_manager
        self.keep_files = keep_files
        self.pes_post_process = cast(Tuple[PostProcess, ...], pes_post_process)

        # HDF5 settings
        self.hdf5_file = hdf5_file

        # Logging settings
        self.logger = cast(Logger, logger)

        # Internally set attributes
        self.pes = {}
        self.pes_validation = {}
        self._data: Dict[Key, np.ndarray] = {}

    @AbstractDataClass.inherit_annotations()
    def _str_iterator(self):
        iterator = ((k.strip('_'), v) for k, v in super()._str_iterator())
        return sorted(iterator)

    def __eq__(self, value: Any) -> bool:
        """Implement :code:`self == value`."""
        if type(self) is not type(value):
            return False
        elif self.keys() != value.keys():
            return False

        names = ('package_manager', 'param', 'phi')
        for name in names:
            if getattr(self, name) != getattr(value, name):
                return False

        if not np.allclose(self.molecule, value.molecule):
            return False

        if self.molecule[0].lattice is not None:
            lat0 = np.array([m.lattice for m in self.molecule], dtype=np.float64)
            lat1 = np.array([m.lattice for m in value.molecule], dtype=np.float64)
            if not np.allclose(lat0, lat1):
                return False
        elif value.molecule[0].lattice is not None:
            return False

        iterator1 = ((v, value.pes[k]) for k, v in self.pes.items())
        for p1, p2 in iterator1:  # type: partial, partial  # type: ignore
            if p1.func != p2.func or p1.keywords != p2.keywords:
                return False

        iterator2 = ((v, value[k]) for k, v in self.items())
        return all((v1 == v2).all() for v1, v2 in iterator2)

    # Implementation of the Mapping protocol

    def __setitem__(self, key: Key, value: np.ndarray) -> None:
        """Implement :code:`self[key] = value`."""
        self._data[key] = value

    def __getitem__(self, key: Key) -> np.ndarray:
        """Implement :code:`self[key]`."""
        return self._data[key]

    def __iter__(self) -> Iterator[Key]:
        """Implement :code:`iter(self)`."""
        return iter(self._data)

    def __len__(self) -> int:
        """Implement :code:`len(self)`."""
        return len(self._data)

    def __contains__(self, key: Any) -> bool:
        """Implement :code:`key in self`."""
        return key in self._data

    def keys(self) -> KeysView[Key]:
        """Return a set-like object providing a view of this instance's keys."""
        return self._data.keys()

    def items(self) -> ItemsView[Key, np.ndarray]:
        """Return a set-like object providing a view of this instance's key/value pairs."""
        return self._data.items()

    def values(self) -> ValuesView[np.ndarray]:
        """Return an object providing a view of this instance's values."""
        return self._data.values()

    @overload
    def get(self, key: Hashable) -> Optional[np.ndarray]: ...
    @overload
    def get(self, key: Hashable, default: T) -> Union[np.ndarray, T]: ...
    def get(self, key, default=None):  # noqa: E301
        """Return the value for **key** if it's available; return **default** otherwise."""
        return self._data.get(key, default)

    # Monte Carlo stuff

    @overload
    def add_pes_evaluator(
        self,
        name: str,
        func: GetPesDescriptor,
        err_func: Callable[[ArrayLike, ArrayLike], np.number | float],
        args: Sequence[Any],
        kwargs: Mapping[str, Any] = ...,
        validation: bool = ...,
        ref: Optional[Sequence[np.ndarray]] = None,
    ) -> None: ...
    @overload  # noqa: E301
    def add_pes_evaluator(
        self,
        name: str,
        func: GetPesDescriptor,
        err_func: Callable[[ArrayLike, ArrayLike], np.number | float],
        args: Sequence[Any],
        kwargs: Iterable[Mapping[str, Any]],
        validation: bool = ...,
        ref: Optional[Sequence[np.ndarray]] = None,
    ) -> None: ...
    def add_pes_evaluator(self, name, func, err_func, args=(), kwargs=EMPTY_MAPPING, validation=False, ref=None):  # noqa: E301, E501
        r"""Add a callable to this instance for constructing PES-descriptors.

        Examples
        --------
        .. code:: python

            >>> from FOX import MonteCarlo, MultiMolecule

            >>> mc = MonteCarlo(...)
            >>> mol = MultiMolecule.from_xyz(...)

            # Prepare arguments
            >>> name = 'rdf'
            >>> func = FOX.MultiMolecule.init_rdf
            >>> atom_subset = ['Cd', 'Se', 'O']  # Keyword argument for func

            # Add the PES-descriptor constructor
            >>> mc.add_pes_evaluator(name, func, kwargs={'atom_subset': atom_subset})

        Parameters
        ----------
        name : :class:`str`
            The name under which the PES-descriptor will be stored (*e.g.* ``"RDF"``).
        func : :class:`~Collections.abc.Callable`
            The callable for constructing the PES-descriptor.
            The callable should take an array-like object as input and
            return a new array-like object as output.
        err_func : :class:`~Collections.abc.Callable`
            The function for computing the auxilary error.
        args : :class:`~collections.abc.Sequence`
            A sequence of positional arguments.
        kwargs : :class:`dict` or :class:`Iterable[dict] <collections.abc.Iterable>`
            A dictionary or an iterable of dictionaries with keyword arguments.
            Providing an iterable allows one to use a unique set of keyword arguments for each
            molecule in :attr:`MonteCarlo.molecule`.
        validation : :class:`bool`
            Whether the PES-descriptor is used exclusively for validation or not.

        """
        mol_list = [m.copy() for _ in self.param.move_range for m in self.molecule]
        for f1 in self.pes_post_process:
            f1(mol_list, self)

        if ref is None:
            ref_iter: Iterable[Optional[np.ndarray]] = repeat(None)
        else:
            ref_iter = [ar for _ in self.param.move_range for ar in ref]

        if not isinstance(kwargs, abc.Mapping):
            _Iterator = Iterator[Tuple[MultiMolecule, Any, Mapping[str, Any]]]
            iterator: _Iterator = zip(mol_list, ref_iter, kwargs)
        else:
            iterator = zip(mol_list, ref_iter, repeat(kwargs, len(mol_list)))

        for i, (mol, ref_, kwarg) in enumerate(iterator):
            if ref_ is None:
                f2 = wraps(func)(partial(_template_func1, func, *args, **kwarg))
                f2.ref = f2(mol, None)
                f2.use_mol = True
                f2.err_func = err_func
            else:
                f2 = wraps(func)(partial(_template_func2, func, *args, **kwarg))
                f2.ref = copy.deepcopy(ref_)
                f2.use_mol = False
                f2.err_func = err_func

            # Check that a numeric value is returned
            dtype = np.asanyarray(f2.ref).dtype
            if dtype.kind not in 'buifc':
                raise TypeError(f"PES descriptor {name!r} has an invalid return dtype:  {dtype}")

            if validation:
                self.pes_validation[f'{name}.{i}'] = f2
            else:
                self.pes[f'{name}.{i}'] = f2

    @abstractmethod
    def __call__(self, **kwargs: Any) -> None:
        raise NotImplementedError('Trying to call an abstract method')

    def restart(self, **kwargs: Any) -> None:
        raise NotImplementedError('Method not implemented')

    def to_yaml_dict(self, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError('Method not implemented')

    @property
    def clear_jobs(self) -> Callable[[], None]:
        """Delete all cp2k output files."""
        return self.package_manager.clear_jobs

    def run_jobs(self) -> Union[Tuple[None, None], Tuple[List[MultiMolecule], List[Any]]]:
        """Run a geometry optimization followed by a molecular dynamics (MD) job.

        Returns a new :class:`~FOX.MultiMolecule` instance constructed from the MD trajectory
        and the path to the MD results.
        If no trajectory is available (*i.e.* the job crashed) return *None* instead.

        * The MD job is constructed according to the provided settings in **self.job**.

        Returns
        -------
        :class:`list[FOX.MultiMolecule] <list>`, optional
            A list of MultiMolecule instance(s) constructed from the MD trajectory.
            Will return :data:`None` if one of the jobs crashed

        """
        return self.package_manager(logger=self.logger)

    def move(self, idx: Optional[int] = None) -> Union[Exception, Key]:
        """Update a random parameter in **self.param** by a random value from **self.move.range**.

        Performs in inplace update of the ``'param'`` column in **self.param**.
        By default the move is applied in a multiplicative manner.
        **self.job.md_settings** and **self.job.preopt_settings** are updated to reflect the
        change in parameters.

        Examples
        --------
        .. code:: python

            >>> print(armc.param['param'])
            charge   Br      -0.731687
                     Cs       0.731687
            epsilon  Br Br    1.045000
                     Cs Br    0.437800
                     Cs Cs    0.300000
            sigma    Br Br    0.421190
                     Cs Br    0.369909
                     Cs Cs    0.592590
            Name: param, dtype: float64

            >>> for _ in range(1000):  # Perform 1000 random moves
            >>>     armc.move()

            >>> print(armc.param['param'])
            charge   Br      -0.597709
                     Cs       0.444592
            epsilon  Br Br    0.653053
                     Cs Br    1.088848
                     Cs Cs    1.025769
            sigma    Br Br    0.339293
                     Cs Br    0.136361
                     Cs Cs    0.101097
            Name: param, dtype: float64

        Parameters
        ----------
        idx : :class:`int`, optional
            The column key for :attr:`param_mapping["param"]<MonteCarloABC.param_mapping.>`.

        Returns
        -------
        :class:`tuple[float, ...] <float>`
            A tuple with the (new) values in the ``'param'`` column of **self.param**.

        """
        idx_: int = idx or 0

        # Perform the move
        ret = self.param(logger=self.logger, param_idx=idx_)
        if isinstance(ret, Exception):
            return ret
        else:
            key, prm_name, _ = ret

        prm_update = self.param.param.loc[(key, prm_name), idx_].to_frame().T
        prm_update.index = [prm_name]
        if idx is None:
            _iterator = chain.from_iterable(self.package_manager.values())
            iterator = (job['settings'] for job in _iterator)
        else:
            iterator = (job_tup[idx_]['settings'] for job_tup in self.package_manager.values())

        # Update the job settings
        for settings in iterator:
            settings[key].update(prm_update)

        return cast(Key, tuple(self.param.param[idx_].values))

    def get_pes_descriptors(
        self, get_first_key: bool = False,
    ) -> Tuple[
        Dict[str, ArrayOrScalar],
        Dict[str, ArrayOrScalar],
        Optional[List[MultiMolecule]]
    ]:
        """Check if a **key** is already present in **history_dict**.

        If :data:`True`, return the matching list of PES descriptors;
        If :data:`False`, construct and return a new list of PES descriptors.

        * The PES descriptors are constructed by the provided settings in **self.pes**.

        Parameters
        ----------
        get_first_key : :class:`bool`
            Keep both the files and the job_cache if this is the first ARMC iteration.
            Usefull for manual inspection in case cp2k hard-crashes at this point.

        Returns
        -------
        :class:`dict[str, np.ndarray[np.float64]] <dict>`, :class:`dict[str, np.ndarray[np.float64]] <dict>` and :class:`list[FOX.MultiMolecule] <list>`
            A previous value from **history_dict** or a new value from an MD calculation &
            a :class:`~FOX.MultiMolecule` instance constructed from the MD simulation.
            Values are set to ``np.inf`` if the MD job crashed.

        """  # noqa: E501
        # Generate PES descriptors
        mol_list, result_list = self.run_jobs()

        if mol_list is not None and result_list is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("error", QMFlows_Warning)

                self.logger.info("Applying PES post-processing")
                for func1 in self.pes_post_process:
                    func1(mol_list, self)  # Post-process the MultiMolecules

                ret1: Dict[str, ArrayOrScalar] = {}
                iterator1 = zip(self.pes.items(), cycle(mol_list), cycle(result_list))
                for (k, func2), mol, result in iterator1:
                    _k, i = k.rsplit(".", maxsplit=1)
                    self.logger.info(f"Calculating descriptor {_k!r} for PES {i}")
                    try:
                        ret1[k] = func2(mol, result)
                    except Exception as ex:
                        raise RuntimeError(
                            f"Failed to compute descriptor {_k!r} for PES {i}"
                        ) from ex

                ret2: Dict[str, ArrayOrScalar] = {}
                iterator2 = zip(self.pes_validation.items(), cycle(mol_list), cycle(result_list))
                for (k, func2), mol, result in iterator2:
                    _k, i = k.rsplit(".", maxsplit=1)
                    self.logger.info(f"Calculating validation descriptor {_k!r} for PES {i}")
                    try:
                        ret2[k] = func2(mol, result)
                    except Exception as ex:
                        raise RuntimeError(
                            f"Failed to compute validation descriptor {_k!r} for PES {i}"
                        ) from ex
        else:
            # The MD simulation crashed
            ret1 = {key: np.inf for key in self.pes.keys()}
            ret2 = {key: np.inf for key in self.pes_validation.keys()}

        if not (get_first_key or self.keep_files):
            self.logger.info("Clearing jobs")
            self.clear_jobs()

        return ret1, ret2, mol_list
