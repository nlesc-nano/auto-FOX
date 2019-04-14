import FOX

from FOX.classes.multi_mol import MultiMolecule
from FOX.examples.example_xyz import get_example_xyz

import numpy as np
import pandas as pd


atoms = ('Cd', 'Se')
mol = MultiMolecule(filename=get_example_xyz())
dist_mean = []
mol2 = mol.deepcopy()
for at in atoms:
    mol2 -= np.average(mol2[:, mol2.atoms[at]], axis=1)[:, None, :]
    dist = np.linalg.norm(mol2.coords[:, mol2.atoms[at]], axis=2).mean(axis=0)
    dist.sort()
    dist_mean.append(dist)

index = np.arange(0, mol.shape[1])
columns, data = mol2._get_rmsf_columns(dist_mean, index, loop=True, atom_subset=atoms)
df = pd.DataFrame(data=data, index=index, columns=columns)
df.columns.name = 'Distance from center\n  /  Ångström'
df.index.name = 'Arbitrary atomic index'
# df.plot()

mol -= np.average(mol[:, mol.atoms[atoms[-1]]], axis=1)[:, None, :]
b = np.zeros_like(mol.coords[:, 0, :])[:, None, :]
mol.coords = np.hstack((mol.coords, b))
mol.atoms['dummy'] = [mol.shape[1] - 1]
atoms = ('dummy', ) + atoms
rdf = mol.init_rdf(atoms)
del rdf['dummy dummy']
rdf = rdf.loc[rdf.index >= 0.5, [i for i in rdf.columns if 'dummy' in i]]
rdf.plot()
