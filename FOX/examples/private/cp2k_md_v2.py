"""A work in progress recipe for MM-MD parameter optimizations with CP2K."""

import os
import stat
import shutil

import numpy as np
import pandas as pd

from scm.plams import add_to_class, Cp2kJob

from FOX import ARMC, run_armc, PRMContainer, estimate_lj
from FOX.ff.lj_dataframe import LJDataFrame
from FOX.ff.lj_uff import UFF_DF
from FOX.ff.lj_calculate import psf_to_atom_dict


# Prepare the ARMC settings
auto_fox = '/Users/bvanbeek/Documents/GitHub/auto-FOX'
f = os.path.join(auto_fox, 'FOX/examples/private/armc_ivan.yaml')
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


# Find all unassigned parameters
prm = PRMContainer.read(armc.md_settings.input.force_eval.mm.forcefield.parm_file_name)
df = LJDataFrame(np.nan, index=set(job_kwargs['psf'].atom_type))
df.overlay_cp2k_settings(armc.md_settings)
df.overlay_prm(prm)
del df['charge']
df = df[df.isna().any(axis=1)]


def uff_guess(df: pd.DataFrame, prm: PRMContainer) -> None:
    nonbonded = prm.nonbonded.set_index(0)
    del nonbonded[1]
    del nonbonded['comment']
    nonbonded.columns = 'epsilon', 'sigma'
    nonbonded[:] = nonbonded.astype(float)

    for i, j in df.index:
        try:
            eps_i, sigma_i = nonbonded.loc[i]
        except KeyError:
            eps_i, sigma_i = UFF_DF.loc[i]

        try:
            eps_j, sigma_j = nonbonded.loc[j]
        except KeyError:
            eps_j, sigma_j = UFF_DF.loc[j]

        sigma = (sigma_i + sigma_j) / 2
        epsilon = np.abs(eps_i * eps_j)**0.5
        df.loc[(i, j)] = epsilon, sigma


def rdf_guess(armc, df: pd.DataFrame) -> None:
    mol = armc.molecule[0]
    mol.atoms = psf_to_atom_dict(armc.psf)
    rdf = mol.init_rdf()
    param = estimate_lj(rdf)


# run_armc(armc, restart=True, **job_kwargs)
