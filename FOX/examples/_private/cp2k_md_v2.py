"""A work in progress recipe for MM-MD parameter optimizations with CP2K."""

import os
import stat
import yaml
import shutil

from scm.plams import add_to_class, Cp2kJob

from FOX.armc import dict_to_armc, run_armc


@add_to_class(Cp2kJob)
def get_runscript(self):
    inp, out = self._filename('inp'), self._filename('out')
    return f'cp2k.ssmp -i {inp} -o {out}'


# Prepare the ARMC settings
file = 'armc_new.yaml'
with open(file, 'r') as f:
    dct = yaml.load(f.read(), Loader=yaml.FullLoader)

try:
    armc, job_kwargs = dict_to_armc(dct)
except Exception as ex:
    exc = ex
    raise ex

# Start ARMC
try:
    workdir = os.path.join(job_kwargs['path'], job_kwargs['folder'])
    shutil.rmtree(workdir)
except FileNotFoundError:
    pass

run_armc(armc, restart=False, **job_kwargs)
