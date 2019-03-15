""" A module fo running MD simulations. """

__all__ = []

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
    def __init__(self, mol, rdf_ref, param, M=50000, phi=1.0, gamma=2.0, omega=100, a_target=0.25,
                 move_min=0.005, move_max=0.1, move_step=0.005,
                 job_type=Cp2kJob, settings=None, atom_subset=None, name=None, path=None):
        """ """
        self.mc = Settings()
        self.mc.param = param
        self.mc.M = int(M)
        self.mc.phi = float(phi)
        self.mc.gamma = float(gamma)
        self.mc.omega = int(omega)
        self.mc.a_target = float(a_target)
        self.mc.move_min = float(move_min)
        self.mc.move_max = float(move_max)
        self.mc.move_step = float(move_step)
        self.job = Settings()
        self.job.mol = mol
        self.job.job_type = job_type
        self.job.settings = settings or self.get_settings()
        self.job.name = name or str(job_type).rsplit("'", 1)[0].rsplit('.', 1)[-1]
        self.job.path = path or os.get_cwd()
        self.rdf = Settings()
        self.rdf.rdf_ref = rdf_ref
        self.rdf.atom_subset = atom_subset

        self._sanitize()

    def _sanitize(self):
        """ Sanitize and validate all arguments of __init__(). """
        # self.mc
        assert isinstance(self.mc.param, (np.ndarray, pd.Series, pd.DataFrame))
        assert self.mc.M // self.mc.omega >= 2
        assert 0.0 < self.mc.a_target < 1.0
        assert self.mc.move_step < self.mc.move_max
        assert (self.mc.move_max - self.mc.move_min) // self.mc.move_step >= 2.0
        assert self.mc.move_min < self.mc.move_max

        # self.job
        assert isinstance(self.job.mol (Molecule, MultiMolecule))
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

    def get_aux_error(self, rdf):
        """ Return the auxiliary error, defined as dEps = Eps_QM - Eps_MM,
        between two radial distribution functions.

        :parameter rdf: A radial distribution function.
        :type rdf: |np.ndarray|_, |pd.DataFrame|_ or |pd.Series|_
        :return: The auxiliary error, dEps, between two radial distribution functions.
        :rtype: |float|_
        """
        return np.linalg.norm(rdf - self.rdf.rdf_ref, axis=0).sum()

    def get_move_range(self):
        """ Generate an array of all allowed moves. """
        rng = np.arange(self.mc.move_min, self.mc.move_max + self.mc.move_min, self.mc.move_step)
        return np.concatenate((-rng, rng))

    def get_settings(self, path=None):
        """ Read and return the template with default CP2K MM-MD settings.
        If **path** is not *None*, read a settings template (.yaml file) from a user-sepcified path.
        Performs an inplace update of **self.job.settings**. """
        file_name = path or join(join(dirname(dirname(__file__)), 'data'), 'md_cp2k.yaml')
        if not isfile(file_name):
            raise FileNotFoundError()
        with open(file_name, 'r') as file:
            self.job.settings = yaml.load(file)

    def run_md(self):
        """ Run a MD job. """
        # Run an MD calculation
        job = self.job.job_type(self.job.mol, settings=self.job.settings, name=self.job.name)
        results = job.run()
        results.wait()

        # Construct a and return a MultiMolecule object
        return MultiMolecule(filename=results.get_xyz_path())

    def run_first_md(self):
        """ Run the first MD calculation before starting the main for-loop. """
        param = self.mc.param
        key = tuple(param)
        mol = self.run_md()
        history_dict = {key: mol.init_rdf(self.rdf.atom_subset) + self.mc.phi}
        assert self.rdf_ref.shape == history_dict[key].shape
        return param, key, history_dict

    def rdf_in_history(self, history_dict, key):
        """ Check if a **key** is already present in **history_dict**.
        If *True*, return the matching RDF; If *False*, construct and return a new RDF. """
        if key in history_dict:
            return history_dict[key]
        else:
            mol = self.run_md()
            return mol.init_rdf(self.rdf.atom_subset)

    def init_mc(self):
        """ """
        init(path=self.job.path, folder='MM-MD')
        move_range = self.get_move_range()
        a = np.zeros(self.mc.omega, dtype=bool)
        phi = self.mc.phi

        # Generate the first RDF
        param, key, history_dict = self.run_first_md()

        # Start the MC parameter optimization
        N = self.mc.M // self.mc.omega
        for _ in range(N):  # Iteration
            for i in range(self.mc.omega):  # Sub-iteration
                # Step 1: Generate a random trial state
                rng = np.random.choice(len(param), 1)
                param[rng] += np.random.choice(move_range, size=1)
                key_old = key
                key = tuple(param)

                # Step 2: Check if the trial state has already been visited; pull its rdf if *True*
                rdf = self.rdf_in_history(history_dict, key)

                # Step 3: Evaluate the auxilary error
                rdf_old = history_dict[key_old]
                accept = self.get_aux_error(rdf_old) - self.get_aux_error(rdf)

                # Step 4: Update the rdf history
                if accept > 0:
                    a[i] = True
                    history_dict[key] = rdf + phi
                else:
                    history_dict[key_old] += phi

            # Update phi and reset the acceptance rate (a)
            phi *= self.mc.gamma**np.sign(self.mc.a_target - np.mean(a))
            a[:] = False

        finish()
        return param
