""" A module fo running MD simulations. """

__all__ = []

import yaml
import numpy as np

from scm.plams.core.settings import Settings
from scm.plams.core.functions import (init, finish, add_to_class)
from scm.plams.interfaces.thirdparty.cp2k import (Cp2kJob, Cp2kResults)

from .read_xyz import read_multi_xyz
from .radial_distribution import get_all_radial


# Set constants
dr = 0.05
r_max = 12.0
atoms = ('Cd', 'Se', 'O')
max_iter = int
mol = None
file_name = '/Users/bvanbeek/Documents/GitHub/auto-FOX/FOX/data/md_cp2k.yaml'
with open(file_name, 'r') as file:
    s = Settings(yaml.load(file))

shape = max_iter, 1 + int(r_max / dr), np.math.factorial(len(atoms))
rdf_history = np.zeros(shape)


shape =  max_iter, np.math.factorial(len(atoms))
param_history = np.zeros(shape)


@add_to_class(Cp2kResults)
def get_multi_xyz(self):
    for file in self.files:
        if '.xyz' in file:
            return self[file]


init(path=None, folder='MD')
for i in max_iter:
    # Run an MD calculation
    name = 'job_' + str(i)
    job = Cp2kJob(mol, settings=s, name=name)
    results = job.run()
    results.wait()

    # Construct the radial distribution function
    xyz_file = results.get_multi_xyz()
    xyz, idx_dict = read_multi_xyz(xyz_file)
    rdf = get_all_radial(xyz, idx_dict)



finish()
