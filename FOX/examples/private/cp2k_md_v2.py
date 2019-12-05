"""A work in progress recipe for MM-MD parameter optimizations with CP2K."""

import os
import stat
import shutil

from scm.plams import add_to_class, Cp2kJob

from FOX import ARMC, run_armc


# Prepare the ARMC settings
auto_fox = '/Users/basvanbeek/Documents/GitHub/auto-FOX'
f = os.path.join(auto_fox, '/FOX/examples/private/armc_ivan.yaml')
armc, job_kwargs = ARMC.from_yaml(f)


@add_to_class(Cp2kJob)
def get_runscript(self):
    inp, out = self._filename('inp'), self._filename('out')
    return f'cp2k.ssmp -i {inp} -o {out}'


# Start ARMC
try:
    os.remove(armc.hdf5_file)
except FileNotFoundError:
    pass
finally:
    shutil.copy2(os.path.join(auto_fox, 'tests/test_files/armc_test.hdf5'), armc.hdf5_file)
    os.chmod(armc.hdf5_file, stat.S_IRWXU)


# import pdb; pdb.set_trace()
run_armc(armc, restart=True, **job_kwargs)
