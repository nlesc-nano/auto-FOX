"""An input file example."""

import time
import pandas as pd

from FOX import (MultiMolecule, get_example_xyz)

__all__ = []

rdf = adf = rmsf = rmsd = None

# Define the atoms of interest and the .xyz path + filename
atoms = ('Cd', 'Se', 'O')
xyz_filename: str = get_example_xyz()

# Optional: start the timer
start: float = time.time()

# Read the .xyz file
mol: MultiMolecule = MultiMolecule.from_xyz(xyz_filename)

# Calculate the RDF, RSMF & RMSD
rdf: pd.DataFrame = mol.init_rdf(atom_subset=atoms)
adf: pd.DataFrame = mol.init_adf(r_max=8.0, atom_subset=['Cd', 'Se'])
rmsf: pd.DataFrame = mol.init_rmsf(atom_subset=atoms)
rmsd: pd.DataFrame = mol.init_rmsd(atom_subset=atoms)
df_dict = {'rdf ': rdf, 'adf ': adf, 'rmsf': rmsf, 'rmsd': rmsd}

# Optional: print the results and try to plot them in a graph (if Matplotlib is installed)
for name, df in df_dict.items():
    try:
        df.plot(name)
    except Exception as ex:
        print(f'{name}: {repr(ex)}')

run_time: float = time.time() - start
print(f'run time: {run_time:.2f} sec')
