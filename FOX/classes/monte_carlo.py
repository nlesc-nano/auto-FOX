"""A module for performing Monte Carlo-based forcefield parameter optimizations."""

import os
import shutil
from os.path import join

from typing import (Tuple, List, Dict, Optional, Union)
import numpy as np
import pandas as pd

from scm.plams import Settings, Molecule
from scm.plams.core.functions import add_to_class
from scm.plams.interfaces.thirdparty.cp2k import (Cp2kJob, Cp2kResults)

from .multi_mol import MultiMolecule
from ..functions.utils import (get_template, _get_move_range, set_nested_value)
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
    """The base :class:`.MonteCarlo` class.

    :parameter ref: A list with :math:`n` PES descriptors as derived from , *e.g.*,
        an *Ab Initio* MD simulation.
    :type ref: :math:`n` |list|_ [|float|_ or |np.ndarray|_ [|np.float64|_]]
    :parameter param: An array with the initial to be optimized forcefield parameters.
    :type param: |np.ndarray|_ [|np.float64|_]
    :parameter molecule: A molecule.
    :type molecule: |plams.Molecule|_ or |FOX.MultiMolecule|_

    :Atributes:     * **param** (|pd.DataFrame|_ – See the **param** parameter.

                    * **pes** (|plams.Settings|_) – See :meth:`MonteCarlo.reconfigure_pes_atr`.

                    * **job** (|plams.Settings|_) – See :meth:`MonteCarlo.reconfigure_job_atr`.

                    * **hdf5_file** (|str|_) – The hdf5 path+filename.

                    * **move** (|plams.Settings|_) – See :meth:`MonteCarlo.reconfigure_move_atr`.
    """

    def __init__(self,
                 molecule: Molecule,
                 param: pd.DataFrame,
                 **kwarg: dict) -> None:
        # Set the inital forcefield parameters
        self.param = param

        # Settings for generating PES descriptors and assigns of reference PES descriptors
        self.pes = Settings()
        self.pes.rdf.ref = None
        self.pes.rdf.func = MultiMolecule.init_rdf
        self.pes.rdf.kwarg = {'atom_subset': None}

        # Settings for running the actual MD calculations
        self.job = Settings()
        self.job.molecule = molecule
        self.job.psf = {}
        self.job.func = Cp2kJob
        self.job.settings = get_template('md_cp2k_template.yaml')
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
        return repr(Settings(vars(self)))

    def as_dict(self, as_Settings: bool = False) -> Union[dict, Settings]:
        """Create a dictionary out of a :class:`.MonteCarlo` instance.

        Parameters:
            bool as_Settings:
                Return as a Settings instance rather than a dictionary.

        Returns:
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
        **self.job.settings** is updated to reflect the change in parameters.

        Returns:
            |tuple|_ [|float|_]:
                A tuple with the (new) values in the ``'param'`` column of **self.param**.
        """
        # Unpack arguments
        param = self.param

        # Perform a move
        idx, x1 = next(param.loc[:, 'param'].sample().items())
        x2 = np.random.choice(self.move.range, 1)
        param.at[idx, 'param'] = self.move.func(x1, x2, **self.move.kwarg)

        # Constrain the atomic charges
        if 'charge' in idx:
            at, charge = idx[1], param.at[idx, 'param']
            with pd.option_context('mode.chained_assignment', None):
                update_charge(at, charge, param.loc['charge'], self.move.charge_constraints)
            for key, value, fstring in param.loc['charge', ['key', 'param', 'unit']].values:
                set_nested_value(self.job.settings, key, fstring.format(value))
        else:
            key, value, fstring = next(iter(param.loc['charge', ['key', 'param', 'unit']].values))
            set_nested_value(self.job.settings, key, fstring.format(value))

        return tuple(self.param['param'].values)

    def run_md(self) -> Tuple[Optional[MultiMolecule], Tuple[str]]:
        """Run a geometry optimization followed by a molecular dynamics (MD) job.

        Returns a new :class:`.MultiMolecule` instance constructed from the MD trajectory and the
        path to the MD results.
        If no trajectory is available (*i.e.* the job crashed) return *None* instead.

        * The MD job is constructed according to the provided settings in **self.job**.

        Returns:
            |FOX.MultiMolecule|_ and |tuple|_ [|str|_]:
                A :class:`.MultiMolecule` instance constructed from the MD trajectory &
                a tuple with the paths to the PLAMS results directories.
        """
        job_type = self.job.func

        # Prepare preoptimization settings
        s_cp = self.job.settings.copy()
        s_cp.input['global'].run_type = 'geometry_optimization'
        s_cp.input.motion.geo_opt.max_iter = s_cp.input.motion.md.steps // 100
        s_cp.input.motion.geo_opt.optimizer = 'LBFGS'
        del s_cp.input.motion.md

        # Preoptimize
        job1 = job_type(name=self.job.name + '_pre_opt', molecule=self.job.molecule, settings=s_cp)
        results1 = job1.run()
        mol_path = results1.get_xyz_path()
        try:
            mol_preopt = MultiMolecule.from_xyz(mol_path).as_Molecule(-1)[0]
        except TypeError:  # The geometry optimization crashed
            return None, (job1.path, )

        # Run an MD calculation
        job2 = job_type(name=self.job.name, molecule=mol_preopt, settings=self.job.settings)
        results2 = job2.run()

        try:  # Construct and return a MultiMolecule object
            mol = MultiMolecule.from_xyz(results2.get_xyz_path())
        except TypeError:  # The MD simulation crashed
            mol = None
        return mol, (job1.path, job2.path)

    def get_pes_descriptors(self,
                            history_dict: Dict[Tuple[float], np.ndarray],
                            key: Tuple[float]
                            ) -> Tuple[Dict[str, np.ndarray], Optional[MultiMolecule]]:
        """Check if a **key** is already present in **history_dict**.

        If *True*, return the matching list of PES descriptors;
        If *False*, construct and return a new list of PES descriptors.

        * The PES descriptors are constructed by the provided settings in **self.pes**.

        Parameters:
            dict history_dict:
                A dictionary with results from previous iteractions.

            tuple key:
                A key in **history_dict**.

        Returns:
            |dict|_ (keys: |str|_, values: array-like)
                A previous value from **history_dict** or a new value from an MD calculation.
                Values are set to np.inf if the MD job crashed.
        """
        # Generate PES descriptors
        mol, path = self.run_md()
        if mol is None:  # The MD simulation crashed
            ret = {key: np.inf for key, value in self.pes.items()}
        else:
            ret = {key: value.func(mol, **value.kwarg) for key, value in self.pes.items()}

        # Delete the output directory and return
        if not self.job.keep_files:
            for i in path:
                shutil.rmtree(i)
        return ret, mol

    @staticmethod
    def get_move_range(start: float = 0.005,
                       stop: float = 0.1,
                       step: float = 0.005) -> np.ndarray:
        """Generate an with array of all allowed moves.

        The move range spans a range of 1.0 +- **stop** and moves are thus intended to
        applied in a multiplicative manner (see :meth:`MonteCarlo.move_param`).

        Example:

        .. code: python

            >>> move_range = ARMC.get_move_range(start=0.005, stop=0.1, step=0.005)
            >>> print(move_range)
            [0.9   0.905 0.91  0.915 0.92  0.925 0.93  0.935 0.94  0.945
             0.95  0.955 0.96  0.965 0.97  0.975 0.98  0.985 0.99  0.995
             1.005 1.01  1.015 1.02  1.025 1.03  1.035 1.04  1.045 1.05
             1.055 1.06  1.065 1.07  1.075 1.08  1.085 1.09  1.095 1.1  ]

        Parameters:
            float start:
                Start of the interval. The interval includes this value.

            float stop:
                End of the interval. The interval includes this value.

            float step:
                Spacing between values.

        Returns:
            |np.ndarray|_ [|np.int64|_]:
                An array with allowed moves.
        """
        return _get_move_range(start, stop, step)
