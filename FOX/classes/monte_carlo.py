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
import shutil
from os.path import join

from typing import (Tuple, List, Dict, Optional, Union, Callable, Iterable)
import numpy as np
import pandas as pd

from scm.plams import Settings, MultiJob, add_to_class
from scm.plams.interfaces.thirdparty.cp2k import (Cp2kJob, Cp2kResults)

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


class MonteCarlo():
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

    def __init__(self, **kwarg: dict) -> None:
        """Initialize a :class:`MonteCarlo` instance."""
        # Set the inital forcefield parameters
        self.param = None

        # Settings for generating PES descriptors and assigns of reference PES descriptors
        self.pes = Settings()
        self.pes.rdf.ref = (None,)
        self.pes.rdf.func = MultiMolecule.init_rdf
        self.pes.rdf.kwarg = {'atom_subset': None}

        # Settings for running the actual MD calculations
        self.job = Settings()
        self.job.molecule = (None,)
        self.job.psf = None
        self.job.func = Cp2kJob
        self.job.md_settings = {}
        self.job.preopt_settings = {}
        self.job.name = self.job.func.__name__.lower()
        self.job.path = os.getcwd()
        self.job.folder = 'MM_MD_workdir'
        self.job.keep_files = False

        # HDF5 settings
        self.hdf5_file = join(self.job.path, 'ARMC.hdf5')

        # Settings for generating Monte Carlo moves
        self.move = Settings()
        self.move.func = np.multiply
        self.move.kwarg = {}
        self.move.charge_constraints = {}
        self.move.range = self.get_move_range()

    def __repr__(self) -> str:
        """Return a string containing a printable representation of this instance."""
        return repr(Settings(vars(self)))

    def as_dict(self, as_Settings: bool = False) -> Union[dict, Settings]:
        """Create a dictionary out of a :class:`.MonteCarlo` instance.

        Parameters
        ----------
        as_Settings : bool
            Return as a Settings instance rather than a dictionary.

        Returns
        -------
        |dict|_ or |plams.Settings|_:
            A (nested) dictionary constructed from **self**.

        """
        if as_Settings:
            return Settings(dir(self))
        else:
            return {k: (v.as_dict() if isinstance(v, Settings) else v) for k, v in dir(self)}

    def move_param(self) -> Tuple[float]:
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
            >>>     armc.move_param()

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
        x2 = np.random.choice(self.move.range, 1)[0]
        value = self.move.func(x1, x2, *self.move.arg, **self.move.kwarg)

        # Ensure that the moved value does not exceed the user-specified minimum and maximum
        if value < self.param.at[idx, 'min']:
            v_min = self.param.at[idx, 'min']
            value += v_min - value
        elif value > self.param.at[idx, 'max']:
            v_max = self.param.at[idx, 'max']
            value += v_max - value

        # Constrain the atomic charges
        if 'charge' in idx:
            at = idx[1]
            with pd.option_context('mode.chained_assignment', None):
                update_charge(at, value, param.loc['charge'], self.move.charge_constraints)
            for k, v, fstring in param.loc['charge', ['keys', 'param', 'unit']].values:
                self.job.md_settings.set_nested(k, fstring.format(v))
                self.job.preopt_settings.set_nested(k, fstring.format(v))
        else:
            param.at[idx, 'param'] = value
            k, v, fstring = param.loc[idx, ['keys', 'param', 'unit']]
            self.job.md_settings.set_nested(k, fstring.format(v))
            self.job.preopt_settings.set_nested(k, fstring.format(v))

        return tuple(self.param['param'].values)

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
        job_type = self.job.func

        # Prepare preoptimization settings
        if self.job.preopt_settings is not None:
            preopt_mol_list, path_list = self._md_preopt(job_type)
            preopt_accept = self._evaluate_rmsd(preopt_mol_list, self.job.rmsd_threshold)
            if not preopt_accept:
                # The RMSD is too large; don't bother with the actual MD simulation
                preopt_mol_list = mol_list = None

            # Run an MD calculation
            if preopt_mol_list is not None:
                mol_list, _path_list = self._md(job_type, preopt_mol_list)
                path_list += _path_list

        else:  # preoptimization is disabled
            mol_list, path_list = self._md(job_type, preopt_mol_list)

        return mol_list, path_list

    def _md_preopt(self, job_type: Callable) -> Tuple[List[Optional[MultiMolecule]], List[str]]:
        """Peform a geometry optimization.

        Optimizations are performed on all molecules in :attr:`MonteCarlo.job`[``"molecule"``].

        Parameters
        ----------
        job_type : |type|_
            The job type of the geometry optimization.
            Expects a subclass of plams.Job.

        Returns
        -------
        |FOX.MultiMolecule|_ and |str|_
            A list of :class:`.MultiMolecule` instance constructed from the geometry optimization &
            a list of paths to the PLAMS results directory.
            The :class:`.MultiMolecule` list is replaced with ``None`` if the job crashes.

        """
        s = self.job.preopt_settings
        mol_list, path_list = [], []
        jobs = [job_type(name=self.job.name + '_pre_opt', molecule=mol, settings=s)
                for mol in self.job.molecule]

        # Preoptimize
        for job in jobs:
            results = job.run()
            try:  # Construct and return a MultiMolecule object
                mol = MultiMolecule.from_xyz(results.get_xyz_path())
                mol.round(3)
            except TypeError:  # The geometry optimization crashed
                return None, path_list

            mol_list.append(mol)
            path_list.append(job.path)

        return mol_list, path_list

    def _md(self, job_type: Callable,
            mol_preopt: Iterable[MultiMolecule]) -> Tuple[Optional[List[MultiMolecule]], List[str]]:
        """Peform a molecular dynamics simulation (MD).

        Simulations are performed on all molecules in **mol_preopt**.

        Parameters
        ----------
        job_type : |type|_
            The job type of the MD simulation.
            Expects a subclass of plams.Job.

        mol_preopt : |list|_ [|FOX.MultiMolecule|_]
            An iterable consisting of :class:`.MultiMolecule` instance(s) constructed from
            geometry pre-optimization(s) (see :meth:`._md_preopt`).

        Returns
        -------
        |list|_ [|FOX.MultiMolecule|_] and |list|_ [|str|_]
            A list of :class:`.MultiMolecule` instance(s) constructed from the MD simulation &
            a list of paths to the PLAMS results directory.
            The :class:`.MultiMolecule` list is replaced with ``None`` if the job crashes.

        """
        input_mol_list = [mol.as_Molecule(-1)[0] for mol in mol_preopt]
        mol_list, path_list = [], []
        jobs = [job_type(name=self.job.name, molecule=mol, settings=self.job.md_settings)
                for mol in input_mol_list]

        # Run MD
        for job in jobs:
            results = job.run()
            try:  # Construct and return a MultiMolecule object
                mol = MultiMolecule.from_xyz(results.get_xyz_path())
                mol.round(3)
            except TypeError:  # The MD simulation crashed
                return None, path_list

            mol_list.append(mol)
            path_list.append(job.path)

        return mol_list, path_list

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

    def get_pes_descriptors(self, history_dict: Dict[Tuple[float], np.ndarray],
                            key: Tuple[float]
                            ) -> Tuple[List[Dict[str, np.ndarray]], Optional[List[MultiMolecule]]]:
        """Check if a **key** is already present in **history_dict**.

        If ``True``, return the matching list of PES descriptors;
        If ``False``, construct and return a new list of PES descriptors.

        * The PES descriptors are constructed by the provided settings in **self.pes**.

        Parameters
        ----------
        history_dict : |dict|_ [|tuple|_ [|float|_], |np.ndarray|_ [|np.float64|_]]
            A dictionary with results from previous iteractions.

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
        mol_list, path_list = self.run_md()
        if mol_list is None:  # The MD simulation crashed
            mol_count = len(next(iter(self.pes)).ref)
            ret = [{key: np.inf for key, value in self.pes}] * mol_count
        else:
            ret = [{key: value.func(mol, *value.arg, **value.kwarg) for
                    key, value in self.pes.items()} for mol in mol_list]

        # Delete the output directory and return
        if not self.job.keep_files:
            for i in path_list:
                shutil.rmtree(i)

        return ret, mol_list

    @staticmethod
    def get_move_range(start: float = 0.005,
                       stop: float = 0.1,
                       step: float = 0.005) -> np.ndarray:
        """Generate an with array of all allowed moves.

        The move range spans a range of 1.0 +- **stop** and moves are thus intended to
        applied in a multiplicative manner (see :meth:`MonteCarlo.move_param`).

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
