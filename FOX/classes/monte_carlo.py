""" A module fo running MD simulations. """

__all__ = ['MonteCarlo', 'ARMC']

import os
from os.path import (join, dirname, isfile, isdir)

import yaml
import numpy as np
import pandas as pd

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
        if '.xyz' in file:
            return self[file]
    raise FileNotFoundError()


class MonteCarlo():
    """ The MonteCarlo class. """
    def __init__(self, param,
                 move_range=None, move_func=np.add, move_func_kwarg={},
                 mol=None, job_type=Cp2kJob, settings=None, atom_subset=None, name=None, path=None):
        """ """
        self.param = param

        self.move = Settings()
        self.move.range = move_range
        self.move.func = move_func
        self.move.kwarg = move_func_kwarg

        self.job = Settings()
        self.job.mol = mol or Molecule()
        self.job.job_type = job_type
        self.job.settings = settings or self.get_settings()
        self.job.name = name or str(job_type).rsplit("'", 1)[0].rsplit('.', 1)[-1]
        self.job.path = path or os.get_cwd()

        self._sanitize()

    def _sanitize(self):
        """ Sanitize and validate all arguments of __init__(). """
        # self.job
        assert isinstance(self.job.mol, (Molecule, MultiMolecule))
        if isinstance(self.job.mol, MultiMolecule):
            self.job.mol = self.job.mol.as_Molecule(0)[0]
        assert isinstance(self.job.settings, (dict, str))
        if isinstance(self.job.settings, str):
            assert isfile(self.job.settings)
            self.job.settings = self.get_settings(path=self.job.settings)
        assert isinstance(self.job.name, str)
        assert isdir(self.job.path)

        # self.rdf
        assert isinstance(self.rdf.rdf_ref, (np.ndarray, pd.Series, pd.DataFrame))

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
        """ Generate an array of all allowed moves. """
        # Create the move range
        move_max = move_max or self.mc.move_max
        move_min = move_min or self.mc.move_min
        move_step = move_step or self.mc.move_step
        rng_range = np.arange(move_min, move_max + move_min, move_step)

        # Set the move range
        self.move_range = np.concatenate((-rng_range, rng_range))

    def move(self):
        """ Update a random parameter in **self.param** by a random value from **self.move_range**.
        """
        i = np.random.choice(len(self.param), 1)
        j = np.random.choice(self.move.range, 1)
        self.param[i] = self.move.func(self.param[i], j, **self.move.kwarg)

    def run_md(self):
        """ Run a MD job. """
        # Run an MD calculation
        job = self.job.job_type(**self.job)
        results = job.run()
        results.wait()

        # Construct a and return a MultiMolecule object
        return MultiMolecule(filename=results.get_xyz_path())

    def run_first_md(self):
        """ Run the first MD calculation before starting the main for-loop.

        :return: A dictionary, containing a list with one or more descriptors of the PES, and
            the first and only key of aforementioned dictionary.
        :rtype: |dict|_ (keys: |tuple|_, values: |list|_) and |tuple|_
        """
        key = tuple(self.param)
        mol = self.run_md()
        values = [mol.init_rdf(self.rdf.atom_subset)]
        history_dict = {key: self.apply_phi(values)}
        return history_dict, key


class ARMC(MonteCarlo):
    """ The Addaptive Rate Monte Carlo (ARMC) class. """
    def __init__(self, iter_len=50000, sub_iter_len=100, gamma=2.0, a_target=0.25,
                 phi=1.0, phi_func=np.add, phi_kwarg={},
                 **kwargs):
        """ """
        MonteCarlo.__init__(self, **kwargs)

        # General ARMC settings
        self.armc = Settings()
        self.armc.iter_len = int(iter_len)
        self.armc.sub_iter_len = int(sub_iter_len)
        self.armc.gamma = gamma
        self.armc.a_target = a_target
        assert iter_len / sub_iter_len >= 1.0

        # Settings specific to handling phi
        self.armc.phi = float(phi)
        self.armc.phi_func = np.add
        self.armc.phi_kwarg = phi_kwarg

    def get_aux_error(self, array):
        """ Return the auxiliary error, defined as dEps = Eps_QM - Eps_MM,
        between two radial distribution functions.

        :parameter rdf: A radial distribution function.
        :type array: |np.ndarray|_, |pd.DataFrame|_ or |pd.Series|_
        :return: The auxiliary error, dEps, between two radial distribution functions.
        :rtype: |float|_
        """
        return np.linalg.norm(array - self.ref, axis=0).sum()

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
            return mol.init_rdf(self.rdf.atom_subset)

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
        """ Update **self.armc.phi** and reset all values in **acceptance** to *False*. """
        sign = np.sign(self.armc.a_target - np.mean(acceptance))
        self.armc.phi *= self.armc.gamma**sign
        acceptance[:] = False

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
        sub_iter = self.mc.sub_iter_len
        super_iter = self.mc.iter_len // sub_iter

        # Set attributes
        self.set_move_range()
        self.ref = ref

        # Initialize
        init(path=self.job.path, folder='MM-MD')
        key, history_dict = self.run_first_md()
        for _ in range(super_iter):
            for i in range(sub_iter):
                # Step 1: Perform a random move
                key_old = key
                key = tuple(self.param)
                self.move()

                # Step 2: Check if the move has been performed already
                new = self.key_in_history(history_dict, key)

                # Step 3: Evaluate the auxilary error
                old = history_dict[key_old]
                accept = bool(self.get_aux_error(old) - self.get_aux_error(new))

                # Step 4: Update the rdf history
                if accept:
                    acceptance[i] = True
                    history_dict[key] = self.apply_phi(new)
                else:
                    history_dict[key_old] = self.apply_phi(old)
            self.update_phi()  # Update phi and reset the acceptance rate
        finish()

        return self.param
