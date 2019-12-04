"""A work in progress recipe for MM-MD parameter optimizations with CP2K."""

from os import remove

from scm.plams import add_to_class, Cp2kJob

from FOX import ARMC, run_armc


# Prepare the ARMC settings
f = '/Users/basvanbeek/Documents/GitHub/auto-FOX/FOX/examples/private/armc_ivan.yaml'
armc, job_kwargs = ARMC.from_yaml(f)


@add_to_class(Cp2kJob)
def get_runscript(self):
    inp, out = self._filename('inp'), self._filename('out')
    return f'cp2k.ssmp -i {inp} -o {out}'


# Start ARMC
try:
    remove(armc.hdf5_file)
except FileNotFoundError:
    pass

run_armc(armc, **job_kwargs)
