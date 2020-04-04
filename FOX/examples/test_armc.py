import yaml
import copy
import shutil
from FOX.armc.sanitization import dict_to_armc
from FOX.armc.run_armc import run_armc


file = '/Users/bvanbeek/Documents/GitHub/auto-FOX/FOX/examples/private/armc_new.yaml'
with open(file, 'r') as f:
    dct = yaml.load(f.read(), Loader=yaml.FullLoader)


try:
    shutil.rmtree('/Users/bvanbeek/Downloads/MM_MD_workdir')
except:
    pass

try:
    armc, kwargs = dict_to_armc(dct)
except Exception as ex:
    exc = ex
    raise ex

# run_armc(armc, **kwargs)
