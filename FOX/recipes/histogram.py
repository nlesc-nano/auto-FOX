"""
from pathlib import Path

import numpy as np
import pandas as pd

from scm.plams import Settings, Units

from FOX import MultiMolecule, get_non_bonded

PATH = Path('tests') / 'test_files'
PATH = Path('/Users/basvanbeek/Documents/GitHub/auto-FOX/tests/test_files')

s = Settings()
s.input.force_eval.mm.forcefield.charge = [
    {'atom': 'Cd', 'charge': 0.933347},
    {'atom': 'Se', 'charge': -0.923076}
]
s.input.force_eval.mm.forcefield.nonbonded['lennard-jones'] = [
    {'atoms': 'Cd Cd', 'epsilon': '[kjmol] 0.310100', 'sigma': '[nm] 0.118464'},
    {'atoms': 'Se Se', 'epsilon': '[kjmol] 0.426600', 'sigma': '[nm] 0.485200'},
    {'atoms': 'Se Cd', 'epsilon': '[kjmol] 1.522500', 'sigma': '[nm] 0.294000'}
]

psf = '/Users/basvanbeek/Documents/CdSe/Week_5/qd/asa/QD_MD.011/QD_MD.psf'
prm = '/Users/basvanbeek/Documents/CdSe/Week_5/qd/asa/QD_MD.011/QD_MD.prm'
mol = MultiMolecule.from_xyz(
    '/Users/basvanbeek/Documents/CdSe/Week_5/qd/asa/QD_MD.011/cp2k-pos-1.xyz'
)[500:]

core_set = {'Cd', 'Se'}
elstat_df, lj_df = get_non_bonded(mol, psf, prm=prm, cp2k_settings=s)
for k in elstat_df.columns.copy():
    if not (k[0] in core_set and k[1] in core_set):
        del elstat_df[k]
        del lj_df[k]

df_tot = elstat_df + lj_df
df_tot -= df_tot.min()
for k, v in df_tot.items():
    df_tot[k] = sorted(v)

df_tot -= df_tot.min()


df_tot_hist = pd.DataFrame()

df_tot *= Units.conversion_ratio('au', 'kcal/mol')
df_tot[:] = df_tot.round().astype(int)

df_tot2 = pd.DataFrame(index=pd.RangeIndex(0, 1+df_tot.values.max()), columns=df_tot.columns.copy())
for k, v in df_tot.items():
    values, idx = np.unique(v, return_index=True)
    df_tot2.loc[values, k] = idx
df_tot2.index /= Units.conversion_ratio('au', 'kcal/mol')
df_tot2.index.name = 'dV_elstat+lj - Hartree'

df_tot4 = elstat_df + lj_df
df_tot4 -= df_tot4.min()
df_tot4[:] = (df_tot4*100).round().astype(int)
df_tot3 = pd.DataFrame(0.0, np.arange(0, 1+df_tot4.values.max()), columns=df_tot.columns.copy())
for k, v in df_tot4.items():
    y, x = np.histogram(v, bins=50)
    df_tot3.loc[x[1:].astype(int), k] = y
df_tot3.index /= 100

import matplotlib.pyplot as plt

ncols = 3
figsize = (4 * ncols, 6)
fig, ax_tup = plt.subplots(ncols=ncols, sharex=True, sharey=False)
if ncols == 1:  # Ensure ax_tup is actually a tuple
    ax_tup = (ax_tup,)

# Construct the actual plots
for (key, series), ax in zip(df_tot3.items(), ax_tup):
    if isinstance(key, tuple):
        key = ' '.join(repr(i) for i in key)
    series.plot(ax=ax, title=key, figsize=figsize)

plt.show(block=True)

"""

"""

Examples
--------
A workflow for creating a histogram of non-bonded interactions.

.. code:: python

    >>> import pandas as pd
    >>> from scm.plams import Units
    >>> from FOX import MultiMolecule, PSFContainer, get_non_bonded
    >>> from FOX.recipes plot_descriptor

    >>> mol = MoltiMolecule.read_xyz(...)
    >>> psf = PSFContainer.read(...)

    >>> elstat_df, lj_df = get_non_bonded(mol, psf, ...)  # Creates two DataFrames

    >>> total_df = elstat_df + lj_df
    >>> total_df *= Units.conversion_ratio('au', 'kcal/mol')  # Switch from Hartree to kcal/mol
    >>> set_average_energy(total_df, mol)

    >>> plot_descriptor(total_df, kind='bin', bins=50)

Index
-----
.. currentmodule:: FOX.recipes.param
.. autosummary::
    plot_descriptor

API
---
.. autofunction:: plot_descriptor

"""

from typing import Iterable, Union, MutableMapping, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scm.plams import Settings, Units
from FOX import MultiMolecule, get_non_bonded

NDFrame = pd.DataFrame.__bases__[0]


def set_average_energy(df: Union[NDFrame, MutableMapping[Tuple[str, str], float]],
                       mol: Union[str, MultiMolecule]) -> None:
    """Average all energies in **df** with respect to the number of (valid) atom-pairs.

    Parameters
    ----------
    df : :class:``
    """
    if not isinstance(mol, MultiMolecule):
        mol = MultiMolecule.read_xyz(mol)

    try:
        for at1, at2 in df.keys():
            pair_count = len(mol.atoms[at1]) * len(mol.atoms[at2])
            if at1 == at2:
                pair_count /= 2
            df[at1, at2] /= pair_count
    except (AttributeError, TypeError) as ex:
        pass


NDFrame = pd.DataFrame.__bases__[0]

PATH = Path('/Users/basvanbeek/Documents/GitHub/auto-FOX/tests/test_files')

s = Settings()
s.input.force_eval.mm.forcefield.charge = [
    {'atom': 'Cd', 'charge': 0.933347},
    {'atom': 'Se', 'charge': -0.923076}
]
s.input.force_eval.mm.forcefield.nonbonded['lennard-jones'] = [
    {'atoms': 'Cd Cd', 'epsilon': '[kjmol] 0.310100', 'sigma': '[nm] 0.118464'},
    {'atoms': 'Se Se', 'epsilon': '[kjmol] 0.426600', 'sigma': '[nm] 0.485200'},
    {'atoms': 'Se Cd', 'epsilon': '[kjmol] 1.522500', 'sigma': '[nm] 0.294000'}
]

psf = '/Users/basvanbeek/Documents/CdSe/Week_5/qd/asa/QD_MD.011/QD_MD.psf'
prm = '/Users/basvanbeek/Documents/CdSe/Week_5/qd/asa/QD_MD.011/QD_MD.prm'
mol = MultiMolecule.from_xyz(
    '/Users/basvanbeek/Documents/CdSe/Week_5/qd/asa/QD_MD.011/cp2k-pos-1.xyz'
)[500:]

core_set = {'Cd', 'Se'}
elstat_df, lj_df = get_non_bonded(mol, psf, prm=prm, cp2k_settings=s)
for k in elstat_df.columns.copy():
    i, j = k
    if not (i in core_set and j in core_set):
        del elstat_df[k]
        del lj_df[k]
    else:
        count = len(mol.atoms[i]) * len(mol.atoms[i])
        if i == j:
            count /= 2
        elstat_df[k] /= count
        lj_df[k] /= count

df_tot = elstat_df + lj_df
df_tot *= Units.conversion_ratio('au', 'kcal/mol')
