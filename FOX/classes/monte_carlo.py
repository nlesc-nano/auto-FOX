""" A module fo running MD simulations. """

__all__ = ['MonteCarlo', 'ARMC']

import os
import shutil
from os.path import isdir

import numpy as np

from scm.plams import Molecule
from scm.plams.core.results import Results
from scm.plams.core.settings import Settings
from scm.plams.core.functions import (init, finish, add_to_class)
from scm.plams.interfaces.thirdparty.cp2k import Cp2kJob

from .multi_mol import MultiMolecule
from ..functions.utils import (get_template, update_charge)
from ..functions.cp2k_utils import (update_cp2k_settings)
from ..functions.hdf5_utils import (create_hdf5, index_to_hdf5, to_hdf5)


@add_to_class(Results)
def get_xyz_path(self):
    """ Return the path + filename to an .xyz file. """
    for file in self.files:
        if 'pos.xyz' in file:
            return self[file]
    raise FileNotFoundError('No .xyz files found in ' + self.job.path)


class MonteCarlo():
    """ The base MonteCarlo class.

    :parameter ref: A list with *n* PES descriptors as derived from , *e.g.*,
        an *Ab Initio* MD simulation.
    :type ref: *n* |list|_ [|float|_ or |np.ndarray|_ [|np.float64|_]]
    :parameter param: An array with the initial to be optimized forcefield parameters.
    :type param: |np.ndarray|_ [|np.float64|_]
    :parameter molecule: A molecule.
    :type molecule: |plams.Molecule|_ or |FOX.MultiMolecule|_

    :Atributes:     * **ref** (|list|_) – See the **ref** parameter.

                    * **param** (|np.ndarray|_ [|np.float64|_]) – See the **param** parameter.

                    * **pes** (|plams.Settings|_) – See :meth:`MonteCarlo.reconfigure_pes_atr`

                    * **move** (|plams.Settings|_) – See :meth:`MonteCarlo.reconfigure_move_atr`

                    * **job** (|plams.Settings|_) – See :meth:`MonteCarlo.reconfigure_job_atr`
    """
    def __init__(self, param, molecule, **kwarg):
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
        self.job.charge_series = None
        self.job.func = Cp2kJob
        self.job.settings = get_template('md_cp2k.yaml')
        self.job.name = self._get_name()
        self.job.path = os.getcwd()
        self.job.keep_files = False

        # Settings for generating Monte Carlo moves
        self.move = Settings()
        self.move.range = self.reconfigure_move_range()
        self.move.func = np.add
        self.move.kwarg = {}
        self.move.charge_constraints = {}

        # Set user-specified keywords
        for key in kwarg:
            if key in self.param:
                self.param[key] = kwarg[key]
            elif key in self.move:
                self.move[key] = kwarg[key]
            elif key in self.job:
                self.job[key] = kwarg[key]

        self._sanitize()

    def __str__(self):
        return str(Settings(vars(self)))

    def _get_name(self):
        """ Return a jobname derived from **self.job.func**. """
        return str(self.job.func).rsplit("'", 1)[0].rsplit('.', 1)[-1]

    def _sanitize(self):
        """ Sanitize and validate all arguments of __init__(). """
        assert isinstance(self.job.molecule, (Molecule, MultiMolecule))
        if isinstance(self.job.molecule, MultiMolecule):
            self.job.molecule = self.job.molecule.as_Molecule(-1)[0]

        assert isinstance(self.job.name, str)
        assert isdir(self.job.path)

    def move(self):
        """ Update a random parameter in **self.param** by a random value from **self.move.range**.
        Performs in inplace update of the *param* column in **self.param**.

        :return: A tuple with the (new) values in the *param* column of **self.param**.
        :rtype: |tuple|_ [|float|_]
        """
        i = self.param.loc[:, 'param'].sample()
        j = np.random.choice(self.move.range, 1)
        i[:] = self.move.func(i, j, **self.move.kwarg)

        # Constrain the atomic charges
        if 'charge' in i:
            for (_, at), charge in i.iteritems():
                pass
            update_charge(at, charge, self.job.charge_series, self.move.charge_constraints)

        # Return a tuple with the new parameters
        return tuple(self.param['param'].values)

    def run_md(self):
        """ Run an MD job, updating the cartesian coordinates of **self.job.mol** and returning
        a new MultiMolecule object.

        * The MD job is constructed according to the provided settings in **self.job**.

        :return: A MultiMolecule object constructed from the MD trajectory & the path to the PLAMS
            results directory.
        :rtype: |FOX.MultiMolecule|_ and |str|_
        """
        # Run an MD calculation
        job_type = self.job.func
        job = job_type(name=self.job.name, molecule=self.job.molecule, settings=self.job.settings)
        results = job.run()
        results.wait()

        # Construct and return a MultiMolecule object
        mol =  MultiMolecule(filename=results.get_xyz_path())
        self.job.mol.from_array(mol[-1])
        return mol, job.path

    def run_first_md(self):
        """ Run the first MD job (before starting the actual main for-loop) and construct a
        matching list of PES descriptors.

        * The MD job is constructed according to the provided settings in **self.job**.

        * The PES descriptors are constructed by the provided settings in **self.pes**.

        :return: A dictionary, containing a list with one or more PES descriptors, and
            the first (and only) key in aforementioned dictionary.
        :rtype: |dict|_ (keys: |tuple|_, values: |list|_) and |tuple|_
        """
        # Run MD
        key = tuple(self.param)
        mol, path = self.run_md()

        # Create PES descriptors
        values = {i: self.pes[i].func(mol, **self.pes[i].kwarg) for i in self.pes}
        history_dict = {key: self.apply_phi(values)}

        # Delete the output directory and return
        if self.job.keep_files:
            shutil.rmtree(path)
        return history_dict, key

    def get_pes_descriptors(self, history_dict, key):
        """ Check if a **key** is already present in **history_dict**.
        If *True*, return the matching list of PES descriptors;
        If *False*, construct and return a new list of PES descriptors.

        * The PES descriptors are constructed by the provided settings in **self.pes**.

        :parameter history_dict: A dictionary with results from previous iteractions.
        :type history_dict: |dict|_ (keys: |tuple|_, values: |list|_)
        :parameter key: A key in **history_dict**.
        :type key: |tuple|_
        :return: A previous value from **history_dict** or a new value from an MD calculation.
        :rtype: |dict|_ (keys: |str|_, values: |np.ndarray|_ [|np.float64|_])
        """
        if key in history_dict:
            return history_dict[key]
        else:
            mol, path = self.run_md()
            ret = {i: self.pes[i].func(mol, **self.pes[i].kwarg) for i in self.pes}

            # Delete the output directory and return
            if not self.job.keep_files:
                shutil.rmtree(path)
            return ret

    def reconfigure_move_range(self, start=0.005, stop=0.1, step=0.005):
        """ Generate an with array of all allowed moves, the moves spanning both the positive and
        negative range.
        Performs an inplace update of **self.move.range**.

        :parameter float start: Start of the interval. The interval includes this value.
        :parameter float stop: End of the interval. The interval includes this value.
        :parameter float step: Spacing between values.
        """
        rng_range = np.arange(start, start + step, step)
        self.move_range = np.concatenate((-rng_range, rng_range))

    def reconfigure_move_atr(self, move_range=None, func=np.add, kwarg={}):
        """ Reconfigure the attributes in **self.move**., the latter containg all settings related
        to generating Monte Carlo moves.
        See :meth:`MonteCarlo.set_move_range` for more extensive **set_move_range** options.

        :parameter move_range: An array of allowed moves.
        :type move_range: |None|_ or |np.ndarray|_ [|np.float64|_]
        :parameter type func: A type object of a function used for performing moves.
            The function in **func** is applied to floats.
        :parameter dict kwarg: Keyword arguments used in **func**.
        """
        self.move.range = move_range or self.set_move_range()
        self.move.func = func
        self.move.kwarg = kwarg

    def reconfigure_job_atr(self, molecule=None, func=Cp2kJob, settings=None,
                            name=None, path=None, keep_files=False):
        """ Reconfigure the attributes in **self.job**, the latter containing all settings related
        to the PLAMS Job class and its subclasses.

        :parameter molecule: A PLAMS molecule.
        :type molecule: |None|_ or |plams.Molecule|_
        :parameter type func: A type object of PLAMS Job subclass.
            Used for running the MD calculations.
        :parameter settings: The job settings used by **func**.
        :type settings: |plams.Settings|_
        :parameter str name: The name of the directories holding the MD results produced by
            **func**.
        :parameter str path: The path where **name** will be stored.
        :parameter bool keep_files: Whether or not all files produced should during the MD
            calculations should be kept or deleted. WARNING: The size of a single MD trajectory can
            easily be in the dozens or even hundreds of megabytes; the MC parameter optimization
            will require thousands of such trajectories.
        """
        if molecule is not None:
            self.job.molecule = molecule
        self.job.func = func
        self.job.settings = settings or get_template('md_cp2k.yaml')
        self.job.name = name or self._get_name()
        self.job.path = path or os.getcwd()
        self.job.keep_files = keep_files


class ARMC(MonteCarlo):
    """ The addaptive rate Monte Carlo (ARMC) class, a subclass of the FOX.MonteCarlo_.

    :Atributes:     * **armc** (|plams.Settings|_) – See :meth:`ARMC.reconfigure_armc_atr`

                    * **phi** (|plams.Settings|_) – See :meth:`ARMC.reconfigure_phi_atr`
    """
    def __init__(self, param, molecule, **kwarg):
        MonteCarlo.__init__(self, param, molecule, **kwarg)

        # Settings specific to addaptive rate Monte Carlo (ARMC)
        self.armc = Settings()
        self.armc.iter_len = 50000
        self.armc.sub_iter_len = 100
        self.armc.gamma = 2.0
        self.armc.a_target = 0.25

        # Settings specific to handling the phi parameter in ARMC
        self.phi = Settings()
        self.phi.phi = 1.0
        self.phi.func = np.add
        self.phi.kwarg = {}

        # Set user-specified keywords
        for key in kwarg:
            if key in self.armc:
                self.armc[key] = kwarg[key]

    def init_armc(self):
        """ Initialize the Addaptive Rate Monte Carlo procedure.

        :return: A new set of parameters.
        :rtype: |pd.DataFrame|_ (index: |pd.MultiIndex|_, values: |np.float64|_)
        """
        # Unpack attributes
        acceptance = np.zeros(self.armc.sub_iter_len, dtype=bool)
        super_iter = self.armc.iter_len // self.armc.sub_iter_len

        # Construct the HDF5 file
        create_hdf5(self)
        hdf5_kwarg = {'param': self.param, 'acceptance': None}

        # Initialize the first MD simulation
        init(path=self.job.path, folder='MM_MD_workdir')
        key_new, history_dict = self.run_first_md()
        hdf5_kwarg.update(history_dict[key_new])
        index_to_hdf5(hdf5_kwarg, self.job.path)

        # Start the ARMC optimization
        for i in range(super_iter):
            for j in range(self.armc.sub_iter_len):
                # Step 1: Perform a random move
                key_old, key_new = key_new, self.move()
                hdf5_kwarg['param'] = self.param

                # Step 2: Check if the move has been performed already
                pes_new = self.get_pes_descriptors(history_dict, key_new)
                hdf5_kwarg.update(pes_new)

                # Step 3: Evaluate the auxilary error
                pes_old = history_dict[key_old]
                accept = bool(sum(self.get_aux_error(pes_old) - self.get_aux_error(pes_new)))
                hdf5_kwarg['acceptance'] = accept

                # Step 4: Update the PES descriptor history
                if accept:
                    acceptance[j] = True
                    history_dict[key_new] = self.apply_phi(pes_new)
                    update_cp2k_settings(self.job.settings, self.param)
                else:
                    history_dict[key_old] = self.apply_phi(pes_old)

                # Step 5: Export the results to HDF5
                to_hdf5(hdf5_kwarg, i, j, self.job.path)

            self.update_phi(acceptance)
            acceptance[:] = False
        finish()
        return self.param

    def get_aux_error(self, pes_dict):
        """ Return the auxiliary error of the PES descriptors in **values** with respect to
        **self.ref**.

        :parameter pes_dict: A dictionary of *n* PES descriptors.
        :type pes_dict: *n* |dict|_ (keys: |str|_, values: |np.ndarray|_ [|np.float64|_])
        :return: An array with *n* auxilary errors
        :rtype: *n* |np.ndarray|_ [|np.float64|_]
        """
        def get_norm(a, b):
            return np.linalg.norm(a - b, axis=0).sum()
        return np.array([get_norm(pes_dict[j], self.pes[j].ref) for i, j in enumerate(pes_dict)])

    def apply_phi(self, values):
        """ Apply **self.phi.phi** to all PES descriptors in **values**.

        * The values are updated according to the provided settings in **self.armc**.

        :parameter values: A list of *n* PES descriptors.
        :type values: *n* |list|_ [|float|_ or |np.ndarray|_ [|np.float64|_]]
        """
        phi = self.phi.phi
        func = self.phi.func
        kwarg = self.phi.kwargf
        for i in values:
            func(i, phi, **kwarg, out=i)

    def update_phi(self, acceptance):
        """ Update **self.phi** based on **self.armc.a_target** and **acceptance**.

        * The values are updated according to the provided settings in **self.armc**.

        :parameter acceptance: An array denoting the accepted moves within a sub-iteration.
        :type acceptance: |np.ndarray|_ [|bool|_]
        """
        sign = np.sign(self.armc.a_target - np.mean(acceptance))
        self.phi.phi *= self.armc.gamma**sign

    def reconfigure_armc_atr(self, iter_len=50000, sub_iter_len=100, gamma=2.0, a_target=0.25):
        """ Reconfigure the attributes in **self.armc**, the latter containing all settings
        specific to addaptive rate Monte Carlo (except phi).

        :parameter int iter_len: The total number of iterations (including sub-iterations).
        :parameter int sub_iter_len: The length of each sub-iteration.
        :parameter float gamma: The parameter gamma.
        :parameter float a_target: The target acceptance rate.
        """
        self.armc.iter_len = iter_len
        self.armc.sub_iter_len = sub_iter_len
        self.armc.gamma = gamma
        self.armc.a_target = a_target

    def reconfigure_phi_atr(self, phi=1.0, func=np.add, kwarg={}):
        """ Reconfigure the attributes in **self.phi**, the latter containing all settings specific
        to the phi parameter in addaptive rate Monte Carlo.

        :parameter float phi: The parameter phi.
        :parameter type func: A type object of a function used for performing moves.
            The function in **func** is applied to scalars, arrays and/or dataframes.
        :parameter dict kwarg: Keyword arguments used in **func**.
        """
        self.phi.phi = phi
        self.phi.func = func
        self.phi.kwarg = kwarg
