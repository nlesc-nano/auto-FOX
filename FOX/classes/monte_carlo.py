"""
FOX.classes.monte_carlo
=======================

A module for performing Monte Carlo-based forcefield parameter optimizations.

Index
-----
.. currentmodule:: FOX.classes.monte_carlo
.. autosummary::
    MonteCarlo

API
---
.. autoclass:: FOX.classes.monte_carlo.MonteCarlo
    :members:
    :private-members:
    :special-members:

"""

import os
import copy as pycopy
import shutil
import logging
import functools
from sys import version_info
from types import MappingProxyType
from itertools import repeat, cycle
from collections import abc
from typing import (
    Tuple, List, Dict, Optional, Union, Iterable, Hashable, Iterator, Any, Mapping, Type, Callable,
    KeysView, ValuesView, ItemsView, Sequence
)

import numpy as np
import pandas as pd

from scm.plams import Molecule, Settings, Cp2kJob, Cp2kResults, add_to_class
from scm.plams.core.basejob import Job
from assertionlib.dataclass import AbstractDataClass

from .multi_mol import MultiMolecule
from ..logger import get_logger
from ..io.read_xyz import XYZError
from ..functions.utils import _get_move_range
from ..functions.charge_utils import update_charge

if version_info.minor < 7:
    from collections import OrderedDict  # noqa
else:  # Dictionaries are ordered starting from python 3.7
    OrderedDict = dict

__all__ = []


@add_to_class(Cp2kResults)
def get_xyz_path(self):
    """Return the path + filename to an .xyz file."""
    for file in self.files:
        if '-pos' in file and '.xyz' in file:
            return self[file]
    raise FileNotFoundError('No .xyz files found in ' + self.job.path)


PostProcess = Callable[[Optional[Iterable[MultiMolecule]], Optional['MonteCarlo']], None]


class MonteCarlo(AbstractDataClass, abc.Mapping):
    r"""The base :class:`.MonteCarlo` class."""

    @property
    def molecule(self) -> Tuple[MultiMolecule, ...]:
        """Get **value** or set **value** as a tuple of MultiMolecule instances."""
        return self._molecule

    @molecule.setter
    def molecule(self, value: Union[MultiMolecule, Iterable[MultiMolecule]]) -> None:
        self._molecule = (value,) if isinstance(value, MultiMolecule) else tuple(value)
        self._plams_molecule = tuple(mol.as_Molecule(0)[0] for mol in self._molecule)

    @property
    def md_settings(self) -> Tuple[Settings, ...]:
        """Get **value** or set **value** as a |plams.Settings| instance."""
        return self._md_settings

    @md_settings.setter
    def md_settings(self, value: Union[Mapping, Iterable[Mapping]]) -> None:
        if isinstance(value, abc.Mapping):
            self._md_settings = (Settings(value),)
        else:
            self._md_settings = tuple(Settings(i) for i in value)

    @property
    def preopt_settings(self) -> Tuple[Settings, ...]:
        """Get **value** or set **value** as a |plams.Settings| instance."""
        return self._preopt_settings

    @preopt_settings.setter
    def preopt_settings(self, value: Union[None, Mapping, Iterable[Mapping]]) -> None:
        if value is None:
            self._preopt_settings = None
        elif isinstance(value, abc.Mapping):
            self._preopt_settings = (Settings(value),)
        else:
            self._preopt_settings = tuple(Settings(i) for i in value)

    @property
    def move_range(self) -> np.ndarray:
        """Get **value** or set **value** as a |np.ndarray|_."""
        return self._move_range

    @move_range.setter
    def move_range(self, value: Optional[Iterable[float]]) -> np.ndarray:
        if value is None:
            self._move_range = _get_move_range()
        else:
            try:
                self._move_range = np.array(value, dtype=float, ndmin=1, copy=False)
            except TypeError:
                self._move_range = np.fromiter(value, dtype=float)

    @property
    def job_name(self) -> str:
        """Get the (lowered) name of :attr:`MonteCarlo.job_type`."""
        try:
            return self.job_type.__name__.lower()
        except AttributeError:  # A functools.partial object
            return self.job_type.func.__name__.lower()

    @property
    def logger(self) -> logging.Logger:
        """Get or set the logger."""
        return self._logger

    @logger.setter
    def logger(self, value: Optional[logging.Logger]) -> None:
        if value is not None:
            self._logger = value
        else:
            self._logger = get_logger(self.__class__.__name__, handler_type=logging.StreamHandler)

    @property
    def pes_post_process(self) -> Tuple[Callable, ...]:
        return self._pes_post_process

    @pes_post_process.setter
    def pes_post_process(self, value: Union[PostProcess, Iterable[PostProcess]]) -> None:
        if value is None:
            self._pes_post_process = ()
        elif isinstance(value, abc.Iterable):
            self._pes_post_process = tuple(value)
        else:
            self._pes_post_process = (value,)

    _PRIVATE_ATTR = frozenset({'_plams_molecule', 'job_cache'})

    def __init__(self, molecule: Union[MultiMolecule, Iterable[MultiMolecule]],
                 param: pd.DataFrame,
                 md_settings: Union[Mapping, Iterable[Mapping]],
                 preopt_settings: Union[None, Mapping, Iterable[Mapping]] = None,
                 rmsd_threshold: float = 5.0,
                 job_type: Type[Job] = Cp2kJob,
                 hdf5_file: str = 'ARMC.hdf5',
                 apply_move: Callable[[float, float], float] = np.multiply,
                 move_range: Optional[np.ndarray] = None,
                 keep_files: bool = False,
                 logger: Optional[logging.Logger] = None,
                 pes_post_process: Union[PostProcess, Iterable[PostProcess]] = None) -> None:
        """Initialize a :class:`MonteCarlo` instance."""
        super().__init__()

        # Set the inital forcefield parameters
        self.param: pd.DataFrame = param

        # Settings for running the actual MD calculations
        self._plams_molecule: Tuple[Molecule, ...] = None  # set by self.molecule
        self.molecule: Tuple[MultiMolecule, ...] = molecule
        self.job_type: Type[Job] = job_type
        self.md_settings: Settings = md_settings
        self.preopt_settings: Optional[Settings] = preopt_settings
        self.rmsd_threshold: float = rmsd_threshold
        self.keep_files: bool = keep_files
        self.pes_post_process: Tuple[PostProcess, ...] = pes_post_process

        # HDF5 settings
        self.hdf5_file: str = hdf5_file

        # Settings for generating Monte Carlo moves
        self.apply_move: Callable[[float, float], float] = apply_move
        self.move_range = move_range

        # Logging settings
        self.logger = logger

        # Internally set attributes
        self.history_dict = {}
        self.pes: Dict[str, Callable[[np.ndarray], np.ndarray]] = OrderedDict()
        self.job_cache: List[Job] = []

    @AbstractDataClass.inherit_annotations()
    def _str_iterator(self):
        iterator = ((k.strip('_'), v) for k, v in super()._str_iterator())
        return sorted(iterator)

    def __eq__(self, value: Any) -> bool: return object.__eq__(self, value)

    # Ensure compatibility with collections.abc.Mapping

    def __setitem__(self, key: Hashable, value: Any) -> None:
        """Set :attr:`MonteCarlo.history_dict` ``[key]`` to ``value``."""
        self.history_dict[key] = value

    def __getitem__(self, key: Hashable) -> Any:
        """Return :attr:`MonteCarlo.history_dict` ``[key]``."""
        return self.history_dict[key]

    def __iter__(self) -> Iterator[Hashable]:
        """Iterate over :attr:`MonteCarlo.history_dict`."""
        return iter(self.history_dict)

    def __len__(self) -> int:
        """Return the number of items in :attr:`MonteCarlo.history_dict`."""
        return len(self.history_dict)

    def __contains__(self, key: Hashable) -> bool:
        """Return whether or not :attr:`MonteCarlo.history_dict` contains the specified key."""
        return key in self.history_dict

    def keys(self) -> KeysView:
        """Return a view of :attr:`MonteCarlo.history_dict`'s keys."""
        return self.history_dict.keys()

    def items(self) -> ItemsView:
        """Return a view of :attr:`MonteCarlo.history_dict`'s items."""
        return self.history_dict.items()

    def values(self) -> ValuesView:
        """Return a view of :attr:`MonteCarlo.history_dict`'s values."""
        return self.history_dict.values()

    def add_pes_evaluator(self, name: str, func: Callable, args: Sequence[Any] = (),
                          kwargs: Mapping[str, Any] = MappingProxyType({})) -> None:
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

        args : :class:`Sequence<collections.abc.Sequence>`
            A sequence of positional arguments.

        kwargs : :class:`dict` or :class:`Iterable<collections.abc.Iterable>` [:class:`dict`]
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
            partial = functools.partial(func, *args, **kwarg)
            partial.__doc__ = func.__doc__
            partial.__name__ = func.__name__
            partial.ref = partial(mol)
            self.pes[f'{name}.{i}'] = partial

    def move(self) -> Tuple[float]:
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

        Returns
        -------
        |tuple|_ [|float|_]:
            A tuple with the (new) values in the ``'param'`` column of **self.param**.

        """
        def _update_settings(k: Tuple[str, ...], v: float, fstring: str) -> None:
            for s in self.md_settings:
                s.set_nested(k, fstring.format(v))
            if self.preopt_settings is not None:
                for s in self.preopt_settings:
                    s.set_nested(k, fstring.format(v))

        # Unpack arguments
        param = self.param

        # Prepare arguments a move
        idx, x1 = next(param.loc[:, 'param'].sample().items())
        x2 = np.random.choice(self.move_range, 1)[0]
        _value = self.apply_move(x1, x2)

        # Ensure that the moved value does not exceed the user-specified minimum and maximum
        value = self.clip_move(idx, _value)
        self.logger.info(f"Moving {idx[0]} ({idx[1]}): {x1:.4f} -> {value:.4f}")

        # Enforce all user-specified constraints
        param_type, atom = idx
        charge = param_type == 'charge'
        constraint_dict = param.at[idx, 'constraints']
        with pd.option_context('mode.chained_assignment', None):
            update_charge(atom, value, param.loc[param_type], constraint_dict, charge=charge)

        # Update the CP2K Settings
        for k, v, fstring in param.loc[param_type, ('keys', 'param', 'unit')].values:
            _update_settings(k, v, fstring)

        return tuple(self.param['param'].values)

    def clip_move(self, idx: Hashable, value: float) -> float:
        """Ensure that **value** falls within a user-specified range."""
        prm_min = self.param.at[idx, 'min']
        prm_max = self.param.at[idx, 'max']

        if value < prm_min:
            return value + (prm_min - value)
        elif value > prm_max:
            return value + (prm_max - value)
        else:
            return value

    def run_md(self) -> Optional[List[MultiMolecule]]:
        """Run a geometry optimization followed by a molecular dynamics (MD) job.

        Returns a new :class:`.MultiMolecule` instance constructed from the MD trajectory and the
        path to the MD results.
        If no trajectory is available (*i.e.* the job crashed) return *None* instead.

        * The MD job is constructed according to the provided settings in **self.job**.

        Returns
        -------
        |FOX.MultiMolecule|_ and |tuple|_ [|str|_]:
            A list of :class:`.MultiMolecule` instance(s) constructed from the MD trajectory &
            a list of paths to the PLAMS results directories.
            The :class:`.MultiMolecule` list is replaced with ``None`` if the job crashes.

        """
        # Prepare preoptimization settings
        if self.preopt_settings is not None:
            preopt_mol_list = self._md_preopt()
            preopt_accept = self._evaluate_rmsd(preopt_mol_list, self.rmsd_threshold)

            # Run an MD calculation
            if preopt_accept and preopt_mol_list is not None:
                return self._md(mol.as_Molecule(-1)[0] for mol in preopt_mol_list)
            return None

        else:  # preoptimization is disabled
            return self._md(self._plams_molecule)

    def _md_preopt(self) -> Optional[List[MultiMolecule]]:
        """Peform a geometry optimization.

        Optimizations are performed on all molecules in :attr:`MonteCarlo.job`[``"molecule"``].

        Returns
        -------
        |FOX.MultiMolecule|_ and |str|_
            A list of :class:`.MultiMolecule` instance constructed from the geometry optimization.
            The :class:`.MultiMolecule` list is replaced with ``None`` if the job crashes.

        """
        name = f'{self.job_name}.preopt'
        job_type = self.job_type
        job_list = [job_type(name=name, molecule=mol, settings=s) for
                    mol, s in zip(self._plams_molecule, self.preopt_settings)]

        # Preoptimize
        mol_list = []
        for job in job_list:
            job.name += '.opt'
            self.job_cache.append(job)
            results = job.run()
            if job.status in {'crashed', 'failed'}:
                return None
            try:  # Construct and return a MultiMolecule object
                path = results.get_xyz_path()
                mol = MultiMolecule.from_xyz(path)
                mol.round(3)
            except TypeError:  # The geometry optimization crashed
                return None
            except XYZError:  # The .xyz file is unreadable for some reason
                self.logger.warning(f"Failed to parse ...{os.sep}{os.path.basename(path)}")
                return None

            mol_list.append(mol)
        return mol_list

    def _md(self, mol_preopt: Iterable[Molecule]) -> Optional[List[MultiMolecule]]:
        """Peform a molecular dynamics simulation (MD).

        Simulations are performed on all molecules in **mol_preopt**.

        Parameters
        ----------
        mol_preopt : |list|_ [|Molecule|_]
            An iterable consisting of PLAMS Molecules.

        Returns
        -------
        |list|_ [|FOX.MultiMolecule|_], optional
            A list of :class:`.MultiMolecule` instance(s) constructed from the MD simulation.
            Return ``None`` if the job crashes.

        """
        name = self.job_name
        job_type = self.job_type
        jobs = [job_type(name=name, molecule=mol, settings=s) for
                mol, s in zip(mol_preopt, self.md_settings)]

        # Run MD
        mol_list = []
        for job in jobs:
            job.name += '.MD'
            self.job_cache.append(job)
            results = job.run()
            if job.status in {'crashed', 'failed'}:
                return None
            try:  # Construct and return a MultiMolecule object
                path = results.get_xyz_path()
                mol = MultiMolecule.from_xyz(path)
                mol.round(3)
            except TypeError:  # The MD simulation crashed
                return None
            except XYZError:  # The .xyz file is unreadable for some reason
                self.logger.warning(f"Failed to parse ...{os.sep}{os.path.basename(path)}")
                return None

            mol_list.append(mol)
        return mol_list

    @staticmethod
    def _evaluate_rmsd(mol_preopt: Optional[Iterable[MultiMolecule]],
                       threshold: Optional[float] = None) -> bool:
        """Evaluate the RMSD of the geometry optimization (see :meth:`_md_preopt`).

        If the root mean square displacement (RMSD) of the last frame is
        larger than **threshold**, return ``False``.
        Otherwise, return ``True`` .

        Parameters
        ----------
        mol_preopt : |FOX.MultiMolecule|_
            A :class:`.MultiMolecule` instance constructed from a geometry pre-optimization
            (see :meth:`_md_preopt`).

        threshold : float
            Optional: An RMSD threshold in Angstrom.
            Determines whether or not a given RMSD will return ``True`` or ``False``.

        Returns
        -------
        |bool|_:
            ``False`` if th RMSD is larger than **threshold**, ``True`` if it is not.

        """
        threshold_ = threshold or np.inf
        if mol_preopt is None:
            return False

        for mol in mol_preopt:
            rmsd = mol.get_rmsd(mol_subset=-1)
            if rmsd > threshold_:
                return False

        return True

    def clear_job_cache(self) -> None:
        """Clear :attr:`MonteCarlo.job_cache` and, optionally, delete all cp2k output files."""
        if not self.keep_files:
            for job in self.job_cache:
                shutil.rmtree(job.path)
        self.job_cache = []

    def get_pes_descriptors(self, key: Tuple[float], get_first_key: bool = False,
                            ) -> Tuple[Dict[str, np.ndarray], Optional[List[MultiMolecule]]]:
        """Check if a **key** is already present in **history_dict**.

        If ``True``, return the matching list of PES descriptors;
        If ``False``, construct and return a new list of PES descriptors.

        * The PES descriptors are constructed by the provided settings in **self.pes**.

        Parameters
        ----------
        key : tuple [float]
            A key in **history_dict**.

        get_first_key : :class:`bool`
            Keep the both the files and the job_cache if this is the first ARMC iteration.
            Usefull for manual inspection in case cp2k hard-crashes at this point.

        Returns
        -------
        |dict|_ [|str|_, |np.ndarray|_ [|np.float64|_]) and |FOX.MultiMolecule|_
            A previous value from **history_dict** or a new value from an MD calculation &
            a :class:`.MultiMolecule` instance constructed from the MD simulation.
            Values are set to ``np.inf`` if the MD job crashed.

        """
        # Generate PES descriptors
        mol_list = self.run_md()
        if mol_list is None:  # The MD simulation crashed
            ret = {key: np.inf for key in self.pes.keys()}
        else:
            for func in self.pes_post_process:
                func(mol_list, self)  # Post-process the MultiMolecules
            iterator = zip(self.pes.items(), cycle(mol_list))
            ret = {k: func(mol) for (k, func), mol in iterator}

        if not get_first_key:
            self.clear_job_cache()

        return ret, mol_list


# Fix for TypeErrors raised during deep copying prior to python 3.7
if version_info.minor < 7:
    @add_to_class(MonteCarlo)
    def copy(self, deep: bool = False) -> MonteCarlo:
        ret = AbstractDataClass.copy(self, deep=False)
        if not deep:
            return ret

        dct = vars(ret)
        for k, v in dct.items():
            try:
                dct[k] = pycopy.deepcopy(v)
            except TypeError:
                dct[k] = pycopy.copy(v)

        return ret
