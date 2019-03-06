""" A module fo running MD simulations. """

__all__ = []

from os.path import (join, dirname)

import yaml
import numpy as np

from scm.plams import Molecule
from scm.plams.core.results import Results
from scm.plams.core.settings import Settings
from scm.plams.core.functions import (init, finish, add_to_class)
from scm.plams.interfaces.thirdparty.cp2k import Cp2kJob

from .radial_distribution import get_all_radial
from ..classes.multi_mol import MultiMolecule


@add_to_class(Results)
def get_xyz_path(self):
    """ Return the path + filename to an .xyz file. """
    for file in self.files:
        if '.xyz' in file:
            return self[file]


# Load the job settings
mol = Molecule(None)
job = Cp2kJob


def get_cp2k_settings(path=None):
    """ Read and return the template with default CP2K MM-MD settings. """
    file_name = path or join(join(dirname(dirname(__file__)), 'data'), 'md_cp2k.yaml')
    s = Settings()
    s.ignore_molecule = True
    with open(file_name, 'r') as file:
        s.input = Settings(yaml.load(file))


def get_md(job, s):
    """ Run an MD calculation. """
    # Run an MD calculation
    name = 'job'
    job = job(settings=s, name=name)
    results = job.run()
    results.wait()

    # Construct the radial distribution function
    xyz_file = results.get_multi_xyz(atom_subset=('Cd', 'Se', 'O'))
    return MultiMolecule(filename=xyz_file)


def param_in_history(param, param_history):
    """ Report if a set of parameters has already been visited.
    Return False it has not, return its index in *param_history* if it has. """
    try:
        return np.where(param in param_history)[0][0]
    except IndexError:
        return False


def get_aux_error(rdf_QM, rdf_MM):
    """ Return the auxiliary error defined as dEps = Eps_QM - Eps_MM.

    g_QM & G_MM <np.ndarray>: A m*n numpy arrays of m radial distribution functions of QM & MM
        calculations, respectively.
    return <float>: The auxiliary error dEps.
    """
    return np.linalg.norm(rdf_QM - rdf_MM, axis=0).sum()


def init_mc(i, j):
    # Step 1: Generate a random trial state
    param = np.random.rand()

    # Step 2: Check if the trial state has already been visited
    idx = param_in_history(param, param_history)
    if idx:
        rdf_new = rdf_history[idx]
    else:
        rdf_new = get_new_rdf(mol, job1, s, param)

    # Step 3: Evaluate the auxilary error
    rdf_old = rdf_history[j]
    accept = get_aux_error(rdf_QM, rdf_new) - get_aux_error(rdf_QM, rdf_old)
    accept *= -1

    # Step 4: Update
    phi = None
    if accept:
        rdf_history[i] = rdf_new + phi
    else:
        rdf_history[j] += phi
