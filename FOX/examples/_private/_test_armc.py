"""Test ARMC input."""

import yaml
import shutil
from FOX.armc import dict_to_armc
# from FOX.armc.run_armc import run_armc


file = 'armc_new.yaml'
with open(file, 'r') as f:
    dct = yaml.load(f.read(), Loader=yaml.FullLoader)


try:
    shutil.rmtree('MM_MD_workdir')
except Exception:
    pass

try:
    armc, kwargs = dict_to_armc(dct)
except Exception as ex:
    exc = ex
    raise ex

# run_armc(armc, **kwargs)
