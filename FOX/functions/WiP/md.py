""" A module fo running MD simulations. """

__all__ = []

import yaml
import numpy as np

from scm.plams import Molecule
from scm.plams.core.results import Results
from scm.plams.core.settings import Settings
from scm.plams.core.functions import (init, finish, add_to_class)
from scm.plams.interfaces.thirdparty.cp2k import Cp2kJob

from .read_xyz import read_multi_xyz
from .radial_distribution import get_all_radial


@add_to_class(Results)
def get_xyz_path(self):
    """ Return the path + filename to an .xyz file. """
    for file in self.files:
        if '.xyz' in file:
            return self[file]


# Load the job settings
mol = Molecule(None)
job1 = Cp2kJob
file_name = '/Users/bvanbeek/Documents/GitHub/auto-FOX/FOX/data/md_cp2k.yaml'
with open(file_name, 'r') as file:
    s1 = Settings(yaml.load(file))


def get_new_rdf(mol, job1, s1, param):
    """ Run an MD calculation and construct the resulting RDF. """
    # Run an MD calculation
    name = 'job_' + str(i)
    job = job1(mol, settings=s1, name=name)
    results = job.run()
    results.wait()

    # Construct the radial distribution function
    xyz_file = results.get_multi_xyz()
    xyz, idx_dict = read_multi_xyz(xyz_file)
    return get_all_radial(xyz, idx_dict)


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
        rdf_new = get_new_rdf(mol, job1, s1, param)

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


# Set constants
dr = 0.05
r_max = 12.0
atoms = ('Cd', 'Se', 'O')
max_iter = int

rdf_QM = None
shape = max_iter, 1 + int(r_max / dr), np.math.factorial(len(atoms))
rdf_history = np.zeros(shape)

shape = max_iter, np.math.factorial(len(atoms))
param_history = np.zeros(shape, dtype=int)

init(path=None, folder='MD')
for i in max_iter:
    pass
finish()
