"""A Recipe for MM-MD parameter optimizations with CP2K 3.1 or 6.1."""

from FOX import ARMC, run_armc


# Start the MC parameterization
armc, job_kwargs = ARMC.from_yaml('armc.yaml')
run_armc(armc, **job_kwargs)
