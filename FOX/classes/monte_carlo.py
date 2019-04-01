""" A module fo running MD simulations. """

__all__ = ['MonteCarlo', 'ARMC']

import os
from os.path import (join, dirname, isfile, isdir)

import yaml
import numpy as np

from scm.plams import Molecule
from scm.plams.core.results import Results
from scm.plams.core.settings import Settings
from scm.plams.core.functions import (init, finish, add_to_class)
from scm.plams.interfaces.thirdparty.cp2k import Cp2kJob

from .multi_mol import MultiMolecule


@add_to_class(Results)
def get_xyz_path(self):
    """ Return the path + filename to an .xyz file. """
    for file in self.files:
        if 'pos.xyz' in file:
            return self[file]
    raise FileNotFoundError()


class MonteCarlo():
    """ The base MonteCarlo class.

    :parameter ref: A list with one or more reference PES descriptors derived from , *e.g.*,
        an *Ab Initio* MD simulation.
    :type ref: |list|_
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
    def __init__(self, ref, param, molecule, **kwarg):
        # Set the reference PES descriptor(s) and initial forcefield parameters
        self.ref = ref
        self.param = param

        # Settings for generating PES descriptors
        self.pes = Settings()
        self.pes.func = [MultiMolecule.init_rdf]
        self.pes.kwarg = [{'atom_subset': None}]

        # Settings for generating Monte Carlo moves
        self.move = Settings()
        self.move.range = self.set_move_range()
        self.move.func = np.add
        self.move.kwarg = {}

        # Settings for running the actual MD calculations
        self.job = Settings()
        self.job.molecule = molecule
        self.job.func = Cp2kJob
        self.job.settings = self.get_settings()
        self.job.name = self._get_name()
        self.job.path = os.get_cwd()

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
        return str(Settings(self))

    def _get_name(self):
        """ Return a name derived from **self.job.func**. """
        return str(self.job.func).rsplit("'", 1)[0].rsplit('.', 1)[-1]

    def _sanitize(self):
        """ Sanitize and validate all arguments of __init__(). """
        if not isinstance(self.ref, list):
            self.ref = list(self.ref)

        assert isinstance(self.job.molecule, (Molecule, MultiMolecule))
        if isinstance(self.job.molecule, MultiMolecule):
            self.job.molecule = self.job.molecule.as_Molecule(-1)[0]

        assert isinstance(self.job.settings, (dict, str))
        if isinstance(self.job.settings, str):
            assert isfile(self.job.settings)
            self.job.settings = self.get_settings(path=self.job.settings)

        assert isinstance(self.job.name, str)
        assert isdir(self.job.path)

    def move(self):
        """ Update a random parameter in **self.param** by a random value from **self.move.range**.
        """
        i = np.random.choice(len(self.param), 1)
        j = np.random.choice(self.move.range, 1)
        self.param[i] = self.move.func(self.param[i], j, **self.move.kwarg)

    def set_move_range(self, start=0.005, stop=0.1, step=0.005):
        """ Generate an with array of all allowed moves, the moves spanning both the positive and
        negative range.
        Performs an inplace update of **self.move.range** if **inplace** = *True*.

        :parameter float start: Start of the interval. The interval includes this value.
        :parameter float stop: End of the interval. The interval includes this value.
        :parameter float step: Spacing between values.
        """
        rng_range = np.arange(start, start + step, step)
        self.move_range = np.concatenate((-rng_range, rng_range))

    def run_md(self):
        """ Run an MD job.

        * The MD job is constructed according to the provided settings in **self.job**.

        :return: A MultiMolecule object constructed from the MD trajectory.
        :rtype: |FOX.MultiMolecule|_
        """
        # Run an MD calculation
        job_type = self.job.func
        job = job_type(**self.job)
        results = job.run()
        results.wait()

        # Construct a and return a MultiMolecule object
        return MultiMolecule(filename=results.get_xyz_path())

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
        mol = self.run_md()

        # Create PES descriptors
        func_list = self.pes.func
        kwarg_list = self.pes.kwarg
        values = [func(mol, **kwarg) for func, kwarg in zip(func_list, kwarg_list)]
        history_dict = {key: self.apply_phi(values)}
        return history_dict, key

    def key_in_history(self, history_dict, key):
        """ Check if a **key** is already present in **history_dict**.
        If *True*, return the matching list of PES descriptors;
        If *False*, construct and return a new list of PES descriptors.

        * The PES descriptors are constructed by the provided settings in **self.pes**.

        :parameter history_dict: A dictionary with results from previous iteractions.
        :type history_dict: |dict|_ (keys: |tuple|_, values: |list|_)
        :parameter key: A key in **history_dict**.
        :type key: |tuple|_
        :return: A previous value from **history_dict** or a new value from an MD calculation.
        :rtype: |list|_
        """
        if key in history_dict:
            return history_dict[key]
        else:
            mol = self.run_md()
            func_list = self.pes.func
            kwarg_list = self.pes.kwarg
            return [func(mol, **kwarg) for func, kwarg in zip(func_list, kwarg_list)]

    def get_settings(self, path=None):
        """ Grab the template with default CP2K_ MM-MD settings.
        If **path** is not *None*, read a settings template (.yaml file) from a user-specified path.
        Performs an inplace update of **self.job.settings**.

        :parameter path: An (optional) user-specified to a .yaml file with job settings
        :type path: |None|_ or |str|_
        """
        if isinstance(path, str):
            file_name = path
            assert isfile(path)
        elif isinstance(path, dict):
            self.job.settings = Settings(path)
            return
        else:
            file_name = join(join(dirname(dirname(__file__)), 'data'), 'md_cp2k.yaml')
        with open(file_name, 'r') as file:
            self.job.settings = yaml.load(file)

    def reconfigure_pes_atr(self, func=MultiMolecule.init_rdf, kwarg={'atom_subset': None}):
        """ Reconfigure the attributes in **self.pes**, the latter containing all settings related
        to generating PES descriptors (*e.g.* `radial distribution functions`_).

        :parameter func: A list of type objects of functions used for generating PES
            descriptors. The functions in **func** are applied to MultiMolecule objects.
        :type func: |type|_ or |list|_ [|type|_]
        :parameter dict kwarg: A list of keyword arguments used in **func**.
        :type kwarg: |dict|_ or |list|_ [|dict|_]
        """
        if isinstance(func, type):
            func = list(func)
        if isinstance(kwarg, dict):
            kwarg = list(kwarg)
        self.pes.func = func
        self.pes.kwarg = kwarg

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
                             name=None, path=None):
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
        """
        if molecule is not None:
            self.job.molecule = molecule
        self.job.func = func
        self.job.settings = settings or self.get_settings(settings)
        self.job.name = name or self._get_name()
        self.job.path = path or os.getcwd()


class ARMC(MonteCarlo):
    """ The addaptive rate Monte Carlo (ARMC) class, a subclass of the FOX.MonteCarlo_.

    :Atributes:     * **armc** (|plams.Settings|_) – See :meth:`ARMC.reconfigure_armc_atr`

                    * **phi** (|plams.Settings|_) – See :meth:`ARMC.reconfigure_phi_atr`
    """
    def __init__(self, **kwarg):
        MonteCarlo.__init__(self, **kwarg)

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

    def __dict__(self):
        return vars(self)

    def __str__(self):
        return str(Settings(vars(self))

    def init_armc(self):
        """ Initialize the Addaptive Rate Monte Carlo procedure.

        :return: A new set of parameters.
        :rtype: |np.ndarray|_ [|np.float64|_]
        """
        # Unpack attributes
        acceptance = np.zeros(self.armc.sub_iter_len, dtype=bool)
        sub_iter = self.armc.sub_iter_len
        super_iter = self.armc.iter_len // sub_iter

        # Initialize
        init(path=self.job.path, folder='MM-MD')
        key_new, history_dict = self.run_first_md()
        for _ in range(super_iter):
            for i in range(sub_iter):
                # Step 1: Perform a random move
                key_old = key_new
                self.move()
                key_new = tuple(self.param)

                # Step 2: Check if the move has been performed already
                value_new = self.key_in_history(history_dict, key_new)

                # Step 3: Evaluate the auxilary error
                value_old = history_dict[key_old]
                accept = bool(sum(self.get_aux_error(value_old) - self.get_aux_error(value_new)))

                # Step 4: Update the PES descriptor history
                if accept:
                    acceptance[i] = True
                    history_dict[key_new] = self.apply_phi(value_new)
                else:
                    history_dict[key_old] = self.apply_phi(value_old)
            self.update_phi(acceptance)
            acceptance[:] = False
        finish()

        return self.param

    def get_aux_error(self, values):
        """ Return the auxiliary error of **values** with respect to **self.ref**.

        :parameter values: A list of PES descriptors.
        :type values: |list|_
        :return: An array with *n* auxilary errors
        :rtype: *n* |np.ndarray|_ [|np.float64|_]
        """
        return np.array([np.linalg.norm(i - j, axis=0).sum() for i, j in zip(values, self.ref)])

    def apply_phi(self, values):
        """ Update all values in **values**.

        * The values are updated according to the provided settings in **self.armc**.

        :parameter values: A list of *n* values.
        :type values: *n* |list|_
        """
        phi = self.phi.phi
        func = self.phi.func
        kwarg = self.phi.kwarg
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
