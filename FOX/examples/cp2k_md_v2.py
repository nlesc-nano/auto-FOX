""" A work in progress recipe for MM-MD parameter optimizations with CP2K. """

import os
from os.path import join

from scm.plams import add_to_class
from scm.plams.interfaces.thirdparty.cp2k import Cp2kJob

from FOX import ARMC


@add_to_class(Cp2kJob)
def get_runscript(self):
    return 'cp2k.ssmp -i {} -o {}'.format(self._filename('inp'), self._filename('out'))


path = r'D:\hardd\Downloads'

# Prepare the ARMC settings
armc = ARMC.from_yaml('armc.yaml')


# Start ARMC
try:
    os.remove(join(path, 'armc.hdf5'))
except FileNotFoundError:
    pass
# armc.init_armc()
