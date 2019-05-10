""" A Recipe for MM-MD parameter optimizations with CP2K 3.1 or 6.1. """

from os.path import join

import FOX
from FOX import (ARMC, MultiMolecule, get_template, get_example_xyz)


# Start the MC parameterization
armc = ARMC.from_yaml('armc.yaml')
armc.init_armc()
