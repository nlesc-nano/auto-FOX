""" A work in progress recipe for MM-MD parameter optimizations with CP2K. """

from os import remove
from os.path import join

from scm.plams import add_to_class
from scm.plams.interfaces.thirdparty.cp2k import Cp2kJob

from FOX import ARMC


@add_to_class(Cp2kJob)
def get_runscript(self):
    return 'cp2k.ssmp -i {} -o {}'.format(self._filename('inp'), self._filename('out'))


# Prepare the ARMC settings
armc = ARMC.from_yaml('armc_ivan.yaml')
armc.hdf5_file = join('/Users/bvanbeek/Downloads', 'armc.hdf5')


# Start ARMC
try:
    remove(armc.hdf5_file)
except FileNotFoundError:
    pass
armc.init_armc()
