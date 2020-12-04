"""A Recipe for MM-MD parameter optimizations with CP2K 3.1 or 6.1."""

import yaml
from FOX.armc import dict_to_armc, run_armc

# Prepare the ARMC settings
file = str(...)
with open(file, 'r') as f:
    dct = yaml.load(f.read(), Loader=yaml.SafeLoader)

armc, kwargs = dict_to_armc(dct)
run_armc(armc, **kwargs)
