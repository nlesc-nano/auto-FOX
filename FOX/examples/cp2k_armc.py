"""A Recipe for MM-MD parameter optimizations with CP2K 3.1 or 6.1."""

import yaml
from FOX.armc import dict_to_armc, run_armc


# Start the MC parameterization
armc, job_kwargs = ARMC.from_yaml('armc.yaml')
run_armc(armc, **job_kwargs)

# Prepare the ARMC settings
file = str(...)
with open(file, 'r') as f:
    dct = yaml.load(f.read(), Loader=yaml.FullLoader)

armc, kwargs = dict_to_armc(dct)
run_armc(armc, **kwargs)
