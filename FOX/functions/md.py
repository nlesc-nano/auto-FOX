""" A module fo running MD simulations. """

__all__ = []

from os.path import (join, dirname, isfile)

import yaml
import numpy as np
import pandas as pd

from scm.plams import Molecule
from scm.plams.core.results import Results
from scm.plams.core.settings import Settings
from scm.plams.core.functions import (init, finish, add_to_class)
from scm.plams.core.basejob import Job
from scm.plams.interfaces.thirdparty.cp2k import Cp2kJob

from ..classes.multi_mol import MultiMolecule


""" ######################### Functions related to running the MM job ######################### """


def run_md(mol=None, job_type=None, settings=None, **kwarg):
    """ """
    # Run an MD calculation
    name = 'MM-MD'
    job = job_type(settings=settings, name=name)
    results = job.run()
    results.wait()

    # Construct a MultiMolecule object
    return MultiMolecule(filename=results.get_xyz_path())


@add_to_class(Results)
def get_xyz_path(self):
    """ Return the path + filename to an .xyz file. """
    for file in self.files:
        if '.xyz' in file:
            return self[file]
    raise FileNotFoundError()


def get_settings(path=None):
    """ Read and return the template with default CP2K MM-MD settings. """
    file_name = path or join(join(dirname(dirname(__file__)), 'data'), 'md_cp2k.yaml')
    if not isfile(file_name):
        raise FileNotFoundError()

    s = Settings()
    s.ignore_molecule = True
    with open(file_name, 'r') as file:
        s.input = Settings(yaml.load(file))


""" ###################### Functions related to the Monte Carlo procedure ##################### """


def get_aux_error(rdf, rdf_ref):
    """ Return the auxiliary error, defined as dEps = Eps_QM - Eps_MM,
    between two radial distribution functions.

    :parameter rdf: A radial distribution function.
    :type rdf: |np.ndarray|_, |pd.DataFrame|_ or |pd.Series|_
    :parameter rdf_ref: A reference radial distribution function.
    :type rdf_ref: |np.ndarray|_, |pd.DataFrame|_ or |pd.Series|_
    :return: The auxiliary error, dEps, between two radial distribution functions.
    :rtype: |float|_
    """
    return np.linalg.norm(rdf - rdf_ref, axis=0).sum()


def init_mc(mol, rdf_ref, start_param,
            M=50000, phi=1.0, gamma=2.0, omega=100, a_target=0.25,
            job=Cp2kJob, settings=None, atom_subset=None):
    """
    :parameter rdf_ref: A reference radial distribution function.
    :type rdf_ref: |np.ndarray|_, |pd.DataFrame|_ or |pd.Series|_
    :parameter start_param: An array with
    :type start_param: |int|_ or |np.ndarray|_
    :parameter int M: The total number of iterations.
    :parameter float phi: The incremenation factor.
    :parameter float gamma: The incremenation factor.
    :parameter int omega: Divides the total number of iterations, **M**, into *N* subiteration
        blocks of length **omega**: *M* = *N*omega*. Should be, at minimum, twice as large as **M**
    :parameter float a_target: The target acceptance rate.
    """
    param, M, omega, phi, gamma, a_target, job_recipe, atom_subset = _sanitize_init_mc(**locals())

    # The acceptance rate
    a = np.zeros(omega, dtype=bool)

    # Generate the first RDF
    key = tuple(param)
    mol = run_md(**job_recipe)
    history_dict = {key: mol.init_rdf(atom_subset) + phi}
    assert rdf_ref.shape == history_dict[key].shape

    # Start the MC parameter optimization
    N = (M // omega)
    for i in range(N):  # Iteration
        for j in range(omega):  # Sub-iteration
            # Step 1: Generate a random trial state
            rng = np.random.choice(len(param), 1)
            param[rng] += np.random.rand(1) - 0.5
            key_old = key
            key = tuple(param)

            # Step 2: Check if the trial state has already been visited
            if key in history_dict:
                rdf = history_dict[key]
            else:
                mol = run_md(**job_recipe)
                rdf = mol.init_rdf(atom_subset)

            # Step 3: Evaluate the auxilary error
            rdf_old = history_dict[key_old]
            accept = get_aux_error(rdf_ref, rdf_old) - get_aux_error(rdf_ref, rdf)

            # Step 4: Update the rdf history
            if accept > 0:
                a[j] = True
                history_dict[key] = rdf + phi
            else:
                history_dict[key_old] += phi

        # Update phi and reset the acceptance rate a
        phi *= gamma**np.sign(a_target - np.average(a))
        a[:] = False

    return param


def _sanitize_init_mc(mol, rdf_ref, start_param,
                      M=50000, phi=1.0, gamma=2.0, omega=100, a_target=0.25,
                      job=Cp2kJob, settings=None, atom_subset=None):
    """ Sanitize the arguments of :func:`FOX.functions.read_xyz.read_multi_xyz`.
    See aforementioned function for a description of the various parameters.

    :return: A sanitized version of: start_param, M, omega, phi, gamma & a_target
    :rtype: |tuple|_
    """
    # Sanitize **mol**
    assert isinstance(mol, (Molecule, MultiMolecule))
    if isinstance(mol, MultiMolecule):
        mol = mol.as_Molecule(mol_subset=0)[0]

    # Sanitize **rdf_ref**
    assert isinstance(rdf_ref, (np.ndarray, pd.DataFrame, pd.Series))

    # Sanitize **start_param**
    assert isinstance(start_param (np.ndarray, int, np.integer))
    if not isinstance(start_param, np.darray):
        start_param = np.random.rand(start_param)
    elif start_param.dtype != np.float:
        start_param = np.array(start_param, dtype=float)

    # Sanitize **M**
    M = int(M)

    # Sanitize **omega**
    omega = int(omega)
    assert (M // omega) > 1

    # Sanitize **phi**
    phi = float(phi)

    # Sanitize **gamma**
    gamma = float(gamma)

    # Sanitize **a_target**
    a_target = float(a_target)
    assert 0.0 < a_target < 1.0

    # Sanitize **job_type**
    assert job is None or isinstance(job, type)
    if job is None:
        job = Cp2kJob

    # Sanitize **settings**
    assert settings is None or isinstance(settings, (str, dict))
    if settings is None:
        settings = get_settings()
    elif isinstance(settings, str):
        settings = get_settings(path=settings)
    elif not isinstance(settings, settings):
        settings = Settings(settings)

    # Sanitize **atom_subset**
    assert atom_subset is None or isinstance(atom_subset, str) or isinstance(atom_subset[0], str)
    at_symbols = [at.symbol for at in mol]
    if atom_subset is None:
        atom_subset = tuple(set(at_symbols))
    elif isinstance(atom_subset, str):
        assert atom_subset in at_symbols
        atom_subset = (atom_subset)
    else:
        for at in atom_subset:
            assert at in at_symbols
        return tuple(atom_subset)

    # Prepare a job recipe
    job_recipe = Settings()
    job_recipe.mol = mol
    job_recipe.job = job
    job_recipe.settings = settings

    return start_param, M, omega, phi, gamma, a_target, job_recipe, atom_subset
