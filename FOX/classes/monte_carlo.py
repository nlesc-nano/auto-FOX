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
    """ The base MonteCarlo class. """
    def __init__(self, param, molecule, ref, **kwarg):
        """ """
        # Reference PES descriptor(s)
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
        self.job.job_type = Cp2kJob
        self.job.settings = self.get_settings()
        self.job.name = str(self.job.job_type).rsplit("'", 1)[0].rsplit('.', 1)[-1]
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

    def get_settings(self, path=None):
        """ Read and return the template with default CP2K MM-MD settings.
        If **path** is not *None*, read a settings template (.yaml file) from a user-sepcified path.
        Performs an inplace update of **self.job.settings**.

        :parameter path: An (optional) user-specified to a .yaml file with job settings
        :type path: |None|_ or |str|_
        """
        file_name = path or join(join(dirname(dirname(__file__)), 'data'), 'md_cp2k.yaml')
        if not isfile(file_name):
            raise FileNotFoundError()
        with open(file_name, 'r') as file:
            self.job.settings = yaml.load(file)

    def set_move_range(self, move_max=0.1, move_min=0.005, move_step=0.005):
        """ Generate and set an array of all allowed moves. """
        rng_range = np.arange(move_min, move_max + move_min, move_step)
        self.move_range = np.concatenate((-rng_range, rng_range))

    def move(self):
        """ Update a random parameter in **self.param** by a random value from **self.move_range**.
        """
        i = np.random.choice(len(self.param), 1)
        j = np.random.choice(self.move.range, 1)
        self.param[i] = self.move.func(self.param[i], j, **self.move.kwarg)

    def run_md(self):
        """ Run a MD job.

        :return: A MultiMolecule object constructed from the MD trajectory.
        :rtype: |FOX.MultiMolecule|_
        """
        # Run an MD calculation
        job_type = self.job.job_type
        job = job_type(**self.job)
        results = job.run()
        results.wait()

        # Construct a and return a MultiMolecule object
        return MultiMolecule(filename=results.get_xyz_path())

    def run_first_md(self):
        """ Run the first MD job before starting the actual main for-loop.

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
        If *True*, return the matching RDF; If *False*, construct and return a new RDF.

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


class ARMC(MonteCarlo):
    """ The addaptive rate Monte Carlo (ARMC) class. """
    def __init__(self, **kwarg):
        """ """
        MonteCarlo.__init__(self, **kwarg)

        # Settings specific to addaptive rate Monte Carlo (ARMC)
        self.armc = Settings()
        self.armc.iter_len = 50000
        self.armc.sub_iter_len = 100
        self.armc.gamma = 2.0
        self.armc.a_target = 0.25

        # Settings specific to handling the phi parameter in ARMC
        self.armc.phi = 1.0
        self.armc.phi_func = np.add
        self.armc.phi_kwarg = {}

        # Set user-specified keywords
        for key in kwarg:
            if key in self.armc:
                self.armc[key] = kwarg[key]

    def get_aux_error(self, values):
        """ Return the auxiliary error.

        :parameter values: A list of PES descriptors.
        :type values: |list|_
        :return: An array of auxilary errors
        :rtype: |float|_
        """
        return np.array([np.linalg.norm(i - j, axis=0).sum() for i, j in zip(values, self.ref)])

    def apply_phi(self, values):
        """ Update all values in **values** with **self.amc.phi**.

        :parameter values: A list of values.
        :type values: |list|_
        """
        phi = self.armc.phi
        func = self.armc.phi_func
        kwarg = self.armc.phi_kwarg
        for i in values:
            func(i, phi, **kwarg, out=i)

    def update_phi(self, acceptance):
        """ Update **self.armc.phi** based on **acceptance**.

        :parameter acceptance: An array denoting the accepted moves within a sub-iteration.
        :type acceptance: |np.ndarray|_ [|bool|_]
        """
        sign = np.sign(self.armc.a_target - np.mean(acceptance))
        self.armc.phi *= self.armc.gamma**sign

    def init_armc(self, ref):
        """ Initialize the Addaptive Rate Monte Carlo procedure.

        :parameter ref: A list with one or more *Ab Initio* reference values. An example would be
            a list with a radial and angular distribution function.
        :type ref: |list|_
        :return: A new set of parameters.
        :rtype: |np.ndarray|_
        """
        # Unpack attributes
        acceptance = np.zeros(self.armc.sub_iter_len, dtype=bool)
        sub_iter = self.armc.sub_iter_len
        super_iter = self.armc.iter_len // sub_iter

        # Set attributes
        self.set_move_range()
        self.ref = ref

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
