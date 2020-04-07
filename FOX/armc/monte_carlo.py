"""
FOX.armc.monte_carlo
====================

A module for performing Monte Carlo-based forcefield parameter optimizations.

Index
-----
.. currentmodule:: FOX.armc.monte_carlo
.. autosummary::
    MonteCarloABC

API
---
.. autoclass:: MonteCarloABC
    :members:
    :private-members:
    :special-members:

"""

from os import PathLike
from functools import wraps, partial
from logging import Logger, StreamHandler
from abc import ABC, abstractmethod
from types import MappingProxyType
from itertools import repeat, cycle, chain
from collections import abc
from typing import (
    Tuple, List, Dict, Optional, Union, Iterable, Hashable, Iterator, Any, Mapping, Callable,
    KeysView, ValuesView, ItemsView, Sequence, TypeVar, overload, TYPE_CHECKING, AnyStr, Generic
)

import numpy as np
from assertionlib.dataclass import AbstractDataClass

from ..logger import get_logger
from ..type_hints import ArrayOrScalar

if TYPE_CHECKING:
    from .package_manager import PackageManager
    from .param_mapping import ParamMapping
    from ..classes.multi_mol import MultiMolecule
else:
    from ..type_alias import PackageManager, ParamMapping, MultiMolecule

__all__ = ['MonteCarloABC']

KT = TypeVar('KT', bound=Tuple[float, ...])
VT = TypeVar('VT', bound=np.ndarray)
T = TypeVar('T')

PostProcess = Callable[[Optional[Iterable[MultiMolecule]], Optional['MonteCarloABC']], None]
GetPesDescriptor = Callable[[MultiMolecule], ArrayOrScalar]


class MonteCarloABC(AbstractDataClass, ABC, Mapping[KT, VT], Generic[KT, VT]):
    r"""The base :class:`.MonteCarlo` class."""

    param: ParamMapping
    package_manager: PackageManager
    keep_files: bool
    hdf5_file: Union[str, PathLike]
    pes: Dict[str, GetPesDescriptor]
    _molecule: Tuple[MultiMolecule, ...]
    _logger: Logger
    _pes_post_process: Tuple[PostProcess, ...]
    _data: Dict[KT, VT]

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
            self._logger = get_logger(self.__class__.__name__, handler_type=StreamHandler)

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
                 hdf5_file: Union[str, bytes, PathLike] = 'armc.hdf5',
                 logger: Optional[Logger] = None,
                 pes_post_process: Optional[Iterable[PostProcess]] = None) -> None:
        """Initialize a :class:`MonteCarlo` instance."""
        super().__init__()

        self.param = param

        # Settings for running the actual MD calculations
        self.molecule = molecule
        self.package_manager = package_manager
        self.keep_files = keep_files
        self.pes_post_process = pes_post_process

        # HDF5 settings
        self.hdf5_file = hdf5_file

        # Logging settings
        self.logger = logger

        # Internally set attributes
        self.pes = {}
        self._data = {}

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
        elif not (self.package_manager == value.package_manager and self.param == value.param):
            return False

        ret = True
        for k, v1 in self.items():
            v2 = value[k]
            ret &= (v1 == v2).all()
        return ret

    # Implementation of the Mapping protocol

    def __setitem__(self, key: KT, value: VT) -> None:
        """Implement :code:`self[key] = value`."""
        self._data[key] = value

    def __getitem__(self, key: KT) -> VT:
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

    def items(self) -> ItemsView[KT, VT]:
        """Return a set-like object providing a view of this instance's key/value pairs."""
        return self._data.items()

    def values(self) -> ValuesView[VT]:
        """Return an object providing a view of this instance's values."""
        return self._data.values()

    def get(self, key: Hashable, default: T = None) -> Union[VT, T]:
        """Return the value for **key** if it's available; return **default** otherwise."""
        return self._data.get(key, default)

    # Monte Carlo stuff

    @overload
    def add_pes_evaluator(self, name: str, func: GetPesDescriptor, args: Sequence,
                          kwargs: Mapping[str, Any]) -> None: ...

    @overload
    def add_pes_evaluator(self, name: str, func: GetPesDescriptor, args: Sequence,
                          kwargs: Iterable[Mapping[str, Any]]) -> None: ...

    def add_pes_evaluator(self, name, func, args=(), kwargs=MappingProxyType({})) -> None:
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
        name : str
            The name under which the PES-descriptor will be stored (*e.g.* ``"RDF"``).

        func : Callable
            The callable for constructing the PES-descriptor.
            The callable should take an array-like object as input and
            return a new array-like object as output.

        args : :class:`~collections.abc.Sequence`
            A sequence of positional arguments.

        kwargs : :class:`dict` or :class:`~collections.abc.Iterable` [:class:`dict`]
            A dictionary or an iterable of dictionaries with keyword arguments.
            Providing an iterable allows one to use a unique set of keyword arguments for each
            molecule in :attr:`MonteCarlo.molecule`.

        """
        mol_list = [m.copy() for m in self.molecule]
        for f in self.pes_post_process:
            f(mol_list, self)

        if not isinstance(kwargs, abc.Mapping):
            iterator = zip(mol_list, kwargs)
        else:
            iterator = zip(mol_list, repeat(kwargs, len(self.molecule)))

        for i, (mol, kwarg) in enumerate(iterator):
            func = wraps(func)(partial(func, *args, **kwarg))
            func.ref = func(mol)
            self.pes[f'{name}.{i}'] = func

    @abstractmethod
    def __call__(self, **kwargs: Any) -> None:
        raise NotImplementedError('Trying to call an abstract method')

    def restart(self, **kwargs: Any) -> None:
        raise NotImplementedError('Method not implemented')

    def to_yaml(self, filename: Union[AnyStr, PathLike], **kwargs: Any) -> None:
        raise NotImplementedError('Method not implemented')

    @property
    def clear_jobs(self) -> Callable[[], None]:
        """Delete all cp2k output files."""
        return self.package_manager.clear_jobs

    def run_jobs(self) -> Optional[List[MultiMolecule]]:
        """Run a geometry optimization followed by a molecular dynamics (MD) job.

        Returns a new :class:`.MultiMolecule` instance constructed from the MD trajectory and the
        path to the MD results.
        If no trajectory is available (*i.e.* the job crashed) return *None* instead.

        * The MD job is constructed according to the provided settings in **self.job**.

        Returns
        -------
        :class:`list` [:class:`MultiMolecule`]
            A list of MultiMolecule instance(s) constructed from the MD trajectory.
            Will return ``None`` if one of the jobs crashed

        """
        return self.package_manager(logger=self.logger)

    def move(self, idx: int = 0) -> Union[Exception, KT]:
        """Update a random parameter in **self.param** by a random value from **self.move.range**.

        Performs in inplace update of the ``'param'`` column in **self.param**.
        By default the move is applied in a multiplicative manner.
        **self.job.md_settings** and **self.job.preopt_settings** are updated to reflect the
        change in parameters.

        Examples
        --------
        .. code:: python

            >>> print(armc.mover['param'])
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
        idx : :class:`int`
            The column key for :attr:`param_mapping["param"]<MonteCarloABC.param_mapping.>`.

        Returns
        -------
        |tuple|_ [|float|_]:
            A tuple with the (new) values in the ``'param'`` column of **self.param**.

        """
        # Perform the move
        ret = self.param(logger=self.logger)
        if isinstance(ret, Exception):
            return ret
        else:
            key, prm_name, _ = ret

        prm_update = self.param['param'][idx].loc[(key, prm_name)].to_frame().T
        prm_update.index = [prm_name]
        iterator = (job['settings'] for job in chain.from_iterable(self.package_manager.values()))

        # Update the job settings
        for settings in iterator:
            settings[key].update(prm_update)

        return tuple(self.param['param'][idx].values)

    def get_pes_descriptors(self, get_first_key: bool = False,
                            ) -> Tuple[Dict[str, ArrayOrScalar], Optional[List[MultiMolecule]]]:
        """Check if a **key** is already present in **history_dict**.

        If ``True``, return the matching list of PES descriptors;
        If ``False``, construct and return a new list of PES descriptors.

        * The PES descriptors are constructed by the provided settings in **self.pes**.

        Parameters
        ----------
        get_first_key : :class:`bool`
            Keep both the files and the job_cache if this is the first ARMC iteration.
            Usefull for manual inspection in case cp2k hard-crashes at this point.

        Returns
        -------
        |dict|_ [|str|_, |np.ndarray|_ [|np.float64|_]) and |FOX.MultiMolecule|_
            A previous value from **history_dict** or a new value from an MD calculation &
            a :class:`.MultiMolecule` instance constructed from the MD simulation.
            Values are set to ``np.inf`` if the MD job crashed.

        """
        # Generate PES descriptors
        mol_list = self.run_jobs()

        if mol_list is None:  # The MD simulation crashed
            ret = {key: np.inf for key in self.pes.keys()}
        else:
            for func in self.pes_post_process:
                func(mol_list, self)  # Post-process the MultiMolecules
            iterator = zip(self.pes.items(), cycle(mol_list))
            ret = {k: func(mol) for (k, func), mol in iterator}

        if not (get_first_key and self.keep_files):
            self.clear_jobs()

        return ret, mol_list
