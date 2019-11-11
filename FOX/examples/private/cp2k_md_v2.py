""" A work in progress recipe for MM-MD parameter optimizations with CP2K. """

from os import remove

from scm.plams import add_to_class
from scm.plams import Cp2kJob

from FOX import ARMC, run_armc


# Prepare the ARMC settings
f = '/Users/basvanbeek/Documents/GitHub/auto-FOX/FOX/examples/private/armc_ivan.yaml'
armc, job_kwargs = ARMC.from_yaml(f)


@add_to_class(Cp2kJob)
def get_runscript(self):
    return 'cp2k.ssmp -i {} -o {}'.format(self._filename('inp'), self._filename('out'))


# Start ARMC
try:
    remove(armc.hdf5_file)
except FileNotFoundError:
    pass


# import pdb; pdb.set_trace()
run_armc(armc, **job_kwargs)
