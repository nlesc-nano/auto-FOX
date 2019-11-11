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

import shutil
import functools
from collections import abc
from typing import (
    Tuple, List, Dict, Optional, Union, Iterable, Hashable, Iterator, Any, Mapping, Type, Callable
)

import numpy as np
import pandas as pd

from scm.plams import Molecule, Settings, Cp2kJob, Cp2kResults, add_to_class
from scm.plams.core.basejob import Job
from assertionlib.dataclass import AbstractDataClass

from .multi_mol import MultiMolecule
from ..functions.utils import _get_move_range
from ..functions.charge_utils import update_charge

__all__: List[str] = []


@add_to_class(Cp2kResults)
def get_xyz_path(self):
    """Return the path + filename to an .xyz file."""
    for file in self.files:
        if '-pos' in file and '.xyz' in file:
            return self[file]
    raise FileNotFoundError('No .xyz files found in ' + self.job.path)


class MonteCarlo(AbstractDataClass, abc.Mapping):
    r"""The base :class:`.MonteCarlo` class.

    .. _plams.init: https://www.scm.com/doc/plams/components/functions.html#scm.plams.core.functions.init  # noqa

    Attributes
    ----------
    hdf5_file : |str|_
        The path+filename of the .hdf5 file containing all Monte Carlo results.

    param : |pd.DataFrame|_
        A DataFrame containing all to-be optimized forcefield paramaters.
        Paramater names are stored in the DataFrame index.
        Contains the following columns:

        * ``"param"`` (|np.float64|_): The current set of paramaters.
        * ``"param_old"`` (|np.float64|_): The last set of accepted paramaters.
        * ``"unit"`` (|object|_): To-be formatted strings containing all units.
        * ``"max"`` (|np.float64|_): The maximum allowed value of each paramater.
        * ``"min"`` (|np.float64|_): The minimum allowed value of each paramater.
        * ``"keys"`` (|object|_): Tuples of keys pointing the parameters.
        * ``"count"`` (|np.int64|_): The number of atoms or atom-pairs relevant for each parameter.

    job : |plams.Settings|_
        A PLAMS Settings instance with all molecular dynamics-related settings.
        Contains the following keys:

        * ``"psf"`` (|FOX.PSF|_): A :class:`.PSF` instance construced
          from :attr:`MonteCarlo.job` [``"molecule"``].
        * ``"func"`` (|type|_): A callable object constructed from a plams.Job_ subclass.
        * ``"path"`` (|str|_): The path to the :attr:`.MonteCarlo.job` [``"folder"``] directory.
          See the ``path`` argument in plams.init_ for more info.
        * ``"name"`` (|str|_): The name of each PLAMS job.
          See the ``name`` argument in plams.Job_ for more info.
        * ``"folder"`` (|str|_): The name of the directory used for storing all PLAMS jobs.
          See the ``folder`` argument in plams.init_ for more info.
        * ``"logfile"`` (|str|_): The path+filename of the PLAMS logfile.
        * ``"molecule"`` (|tuple|_ [|plams.Molecule|_]): An iterable consisting of
          PLAMS molecule(s).
        * ``"keep_files"`` (|bool|_): Whether or not results of the PLAMS jobs should be saved or
          deleted after each respective job is finished.
        * ``"md_settings"`` (|plams.Settings|_): A settings instance with all molecular dynamics
          settings.
        * ``"rmsd_threshold"`` (|float|_): An RMSD threshold for the geometry (pre-)optimization.
          Parameters are discarded if this threshold is exceeded.
        * ``"preopt_settings"`` (|plams.Settings|_): A settings instance with all geometry
          (pre-)optimization settings.

    move : |plams.Settings|_
        A PLAM Settings instance with settings related to moving paramaters.
        Contains the following keys:

        * ``"arg"`` (|list|_): A list of arguments for :attr:`.MonteCarlo.move` [``"func"``].
        * ``"func"`` (|type|_): The callable for performing paramaters moves.
        * ``"range"`` (|np.ndarray|_): An array of allowed moves.
        * ``"kwarg"`` (|dict|_): A dictionary with keyword arguments
          for :attr:`.MonteCarlo.move` [``"func"``].
        * ``"charge_constraints"`` (|dict|_): A dictionary with optional constraints for updating
          atomic charges.

    pes : |plams.Settings|_
        A PLAM Settings instance with settings related to constructing PES descriptors.
        Using the RDF as example, this section has the following general structure:

        ::

            key:
                func: FOX.MultiMolecule.init_rdf
                arg: []
                kwarg:
                    atom_subset: [Cd, Se, O]

        Contains the following keys:

        * ``key`` (|str|_): One or more user-specified keys. The exact value of each key is
          irrelevant and can be altered to ones desire.
        * ``"ref"`` (|tuple|_ [|np.ndarray|_]): An iterable consisting of array-like objects
          holding (*ab-initio*) reference PES descriptor(s).
        * ``"arg"`` (|list|_): A list of arguments
          for :attr:`.MonteCarlo.pes` [``key``][``"func"``].
        * ``"func"`` (|type|_): A callable for constructing a specific PES descriptor.
        * ``"kwarg"`` (|dict|_): A dictionary with keyword arguments
          for :attr:`.MonteCarlo.pes` [``key``][``"func"``].

    """

    @property
    def molecule(self) -> Tuple[MultiMolecule, ...]:
        """Get **value** or set **value** as a tuple of MultiMolecule instances."""
        return self._molecule

    @molecule.setter
    def molecule(self, value: Union[MultiMolecule, Iterable[MultiMolecule]]) -> None:
        self._molecule = (value,) if isinstance(value, Molecule) else tuple(value)

    @property
    def md_settings(self) -> Settings:
        """Get **value** or set **value** as a |plams.Settings| instance."""
        return self._md_settings

    @md_settings.setter
    def md_settings(self, value: Optional[Mapping]) -> None:
        self._md_settings = Settings(value)

    @property
    def preopt_settings(self) -> Settings:
        """Get **value** or set **value** as a |plams.Settings| instance."""
        return self._preopt_settings

    @preopt_settings.setter
    def preopt_settings(self, value: Optional[Mapping]) -> None:
        self._preopt_settings = Settings() if value is None else Settings(value)

    @property
    def job_name(self) -> str:
        """Get the (lowered) name of :attr:`MonteCarlo.job_type`."""
        return self.job_type.__name__.lower()

    def __init__(self, molecule: Union[Molecule, Iterable[Molecule]],
                 md_settings: Mapping,
                 preopt_settings: Optional[Mapping] = None,
                 param: Mapping = None,
                 job_type: Type[Job] = Cp2kJob,
                 hdf5_file: str = 'ARMC.hdf5',
                 apply_move: Callable[[float, float], float] = np.multiply,
                 charge_constraints: Optional[Mapping] = None,
                 **kwargs: Callable[[np.ndarray], np.ndarray]) -> None:
        """Initialize a :class:`MonteCarlo` instance."""
        super().__init__()

        # Set the inital forcefield parameters
        self.param: pd.DataFrame = param

        # Settings for running the actual MD calculations
        self.molecule: Tuple[MultiMolecule, ...] = molecule
        self.job_type: Type[Job] = job_type
        self.md_settings: Settings = md_settings
        self.preopt_settings: Settings = preopt_settings

        # HDF5 settings
        self.hdf5_file: str = hdf5_file

        # Settings for generating Monte Carlo moves
        self.apply_move: Callable[[float, float], float] = apply_move
        self.charge_constraints: dict = charge_constraints

        self.move_range = self.get_move_range()
        self.history_dict = {}
        self.pes: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}

        # Assign functions for creating PES-descriptors
        for k, v in kwargs.items():
            if not callable(v):
                raise TypeError(f"The parameter '{k}' is not a callable object; "
                                f"observed type: '{v.__class__.__name__}'")
            self.add_pes_evaluator(k, v)

    def __setitem__(self, key: Hashable, value: Any) -> None: self.history_dict[key] = value

    def __getitem__(self, key: Hashable) -> Any: return self.history_dict[key]

    def __iter__(self) -> Iterator[Hashable]: return iter(self.history_dict)

    def __len__(self) -> int: return len(self.history_dict)

    def __contains__(self, key: Hashable) -> bool: return key in self.history_dict

    def keys(self): return self.history_dict.keys()

    def items(self): return self.history_dict.items()

    def values(self): return self.history_dict.values()

    def add_pes_evaluator(self, name: str, func: Callable, *args: Any, **kwargs: Any) -> None:
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
            >>> mc.add_pes_evaluator(name, func, atom_subset=atom_subset)

        Parameters
        ----------
        name : str
            The name under which the PES-descriptor will be stored (*e.g.* ``"RDF"``).

        func : Callable
            The callable for constructing the PES-descriptor.
            The callable should take an array-like object as input and
            return a new array-like object as output.

        \*args/\**kwargs : :data:`Any<typing.Any>`
            Further positional and/or keyword arguments for **func**.

        """
        partial = functools.partial(func, *args, **kwargs)
        partial.__doc__ = func.__doc__
        partial.reference = tuple(partial(mol) for mol in self.molecule)
        self.pes[name] = partial

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
        # Unpack arguments
        param = self.param

        # Prepare arguments a move
        idx, x1 = next(param.loc[:, 'param'].sample().items())
        x2 = np.random.choice(self.move_range, 1)[0]
        value = self.apply_move(x1, x2)

        # Ensure that the moved value does not exceed the user-specified minimum and maximum
        value = self.clip_move(idx, value)

        # Constrain the atomic charges
        if 'charge' in idx:
            at = idx[1]
            with pd.option_context('mode.chained_assignment', None):
                update_charge(at, value, param.loc['charge'], self.charge_constraints)
            for k, v, fstring in param.loc['charge', ['keys', 'param', 'unit']].values:
                self.job.md_settings.set_nested(k, fstring.format(v))
                self.job.preopt_settings.set_nested(k, fstring.format(v))
        else:
            param.at[idx, 'param'] = value
            k, v, fstring = param.loc[idx, ['keys', 'param', 'unit']]
            self.job.md_settings.set_nested(k, fstring.format(v))
            self.job.preopt_settings.set_nested(k, fstring.format(v))

        return tuple(self.param['param'].values)

    def clip_move(self, idx: Hashable, value: float) -> float:
        """Ensure that **value** falls within a user-specified range."""
        prm_min = self.param.at[idx, 'min']
        if value < prm_min:
            return value + (prm_min - value)

        prm_max = self.param.at[idx, 'max']
        if value > prm_max:
            return value + (prm_max - value)

    def run_md(self) -> Tuple[Optional[List[MultiMolecule]], List[str]]:
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
        if self.job.preopt_settings is not None:
            preopt_mol_list = self._md_preopt()
            preopt_accept = self._evaluate_rmsd(preopt_mol_list, self.job.rmsd_threshold)

            # Run an MD calculation
            if preopt_accept and preopt_mol_list is not None:
                return self._md(preopt_mol_list)
            return None

        else:  # preoptimization is disabled
            return self._md()

    def _md_preopt(self) -> List[Optional[MultiMolecule]]:
        """Peform a geometry optimization.

        Optimizations are performed on all molecules in :attr:`MonteCarlo.job`[``"molecule"``].

        Returns
        -------
        |FOX.MultiMolecule|_ and |str|_
            A list of :class:`.MultiMolecule` instance constructed from the geometry optimization.
            The :class:`.MultiMolecule` list is replaced with ``None`` if the job crashes.

        """
        s = self.job.preopt_settings
        name = self.job_name
        job_type = self.job_type
        job_list = [job_type(name=name, molecule=mol, settings=s) for mol in self.molecule]

        # Preoptimize
        mol_list = []
        for job in job_list:
            results = job.run()
            try:  # Construct and return a MultiMolecule object
                mol = MultiMolecule.from_xyz(results.get_xyz_path())
                mol.round(3)
            except TypeError:  # The geometry optimization crashed
                return None

            mol_list.append(mol)
        return mol_list

    def _md(self, mol_preopt: Optional[Iterable[MultiMolecule]] = None
            ) -> Optional[List[MultiMolecule]]:
        """Peform a molecular dynamics simulation (MD).

        Simulations are performed on all molecules in **mol_preopt**.

        Parameters
        ----------
        mol_preopt : |list|_ [|FOX.MultiMolecule|_]
            An iterable consisting of :class:`.MultiMolecule` instance(s) constructed from
            geometry pre-optimization(s) (see :meth:`._md_preopt`).

        Returns
        -------
        |list|_ [|FOX.MultiMolecule|_]
            A list of :class:`.MultiMolecule` instance(s) constructed from the MD simulation.
            The :class:`.MultiMolecule` list is replaced with ``None`` if the job crashes.

        """
        s = self.job.md_settings
        name = self.job_name
        job_type = self.job_type
        mol_generator = (mol.as_Molecule(-1)[0] for mol in mol_preopt)
        jobs = [job_type(name=name, molecule=mol, settings=s) for mol in mol_generator]

        # Run MD
        mol_list = []
        for job in jobs:
            results = job.run()
            try:  # Construct and return a MultiMolecule object
                mol = MultiMolecule.from_xyz(results.get_xyz_path())
                mol.round(3)
            except TypeError:  # The MD simulation crashed
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
            mol_subset = slice(0, None, len(mol) - 1)
            rmsd = mol.get_rmsd(mol_subset)
            if rmsd[1] > threshold_:
                return False

        return True

    def get_pes_descriptors(self, key: Tuple[float]
                            ) -> Tuple[List[Dict[str, np.ndarray]], Optional[List[MultiMolecule]]]:
        """Check if a **key** is already present in **history_dict**.

        If ``True``, return the matching list of PES descriptors;
        If ``False``, construct and return a new list of PES descriptors.

        * The PES descriptors are constructed by the provided settings in **self.pes**.

        Parameters
        ----------
        key : tuple [float]
            A key in **history_dict**.

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
            mol_count = len(next(iter(self.pes)).ref)
            ret = [{key: np.inf for key, value in self.pes}] * mol_count
        else:
            ret = [{key: func(mol) for key, func in self.pes.items()} for mol in mol_list]

        # Delete the output directory and return
        path_list = None
        if not self.job.keep_files:
            for i in path_list:
                shutil.rmtree(i)

        return ret, mol_list

    @staticmethod
    def get_move_range(start: float = 0.005, stop: float = 0.1, step: float = 0.005) -> np.ndarray:
        """Generate an with array of all allowed moves.

        The move range spans a range of 1.0 +- **stop** and moves are thus intended to
        applied in a multiplicative manner (see :meth:`MonteCarlo.move`).

        Examples
        --------
        .. code:: python

            >>> move_range = ARMC.get_move_range(start=0.005, stop=0.1, step=0.005)
            >>> print(move_range)
            [0.9   0.905 0.91  0.915 0.92  0.925 0.93  0.935 0.94  0.945
             0.95  0.955 0.96  0.965 0.97  0.975 0.98  0.985 0.99  0.995
             1.005 1.01  1.015 1.02  1.025 1.03  1.035 1.04  1.045 1.05
             1.055 1.06  1.065 1.07  1.075 1.08  1.085 1.09  1.095 1.1  ]

        Parameters
        ----------
        start : float
            Start of the interval. The interval includes this value.

        stop : float
            End of the interval. The interval includes this value.

        step : float
            Spacing between values.

        Returns
        -------
        |np.ndarray|_ [|np.int64|_]:
            An array with allowed moves.

        """
        return _get_move_range(start, stop, step)
