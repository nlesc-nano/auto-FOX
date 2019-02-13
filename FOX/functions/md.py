""" A module fo running MD simulations. """

__all__ = []

import yaml

from scm.plams.core.settings import Settings
from scm.plams.core.functions import (init, finish)
from scm.plams.interfaces.thirdparty.cp2k import Cp2kJob


file_name = '/Users/basvanbeek/Documents/GitHub/auto-FOX/FOX/data/md_cp2k.yaml'
with open(file_name, 'r') as file:
    s = Settings(yaml.load(file))
