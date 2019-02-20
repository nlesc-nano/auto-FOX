__all__ = []

import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from FOX.functions.read_xyz import read_multi_xyz
from FOX.examples.example_xyz import get_example_xyz


xyz_file = get_example_xyz()
xyz_array, idx_dict = read_multi_xyz(xyz_file)

atoms = ('Cd', 'Se', 'O')
atom_pairs = [(at1, at2, at3) for
              i, at1 in enumerate(atoms) for
              j, at2 in enumerate(atoms[i:]) for
              at3 in atoms[i+j:]]

xyz = xyz_array[0]
at1, at2, at3 = atom_pairs[0]
A, B, C = xyz[idx_dict[at1]], xyz[idx_dict[at1]], xyz[idx_dict[at1]]


def get_unit_vec(A, B):
    vec1 = A[None, :] - B[:, None]
    norm = cdist(A, B)
    return vec1 / norm[:, :, None], norm


def get_adf(A, B, C):
    vec1, norm1 = get_unit_vec(A, B)
    vec2, norm2 = get_unit_vec(A, C)

    angle = np.arccos(np.einsum('ijk,ilk->ijl', vec1, vec2)) # Angles in radian (float)
    angle_int = np.clip(np.array(np.degrees(angle), dtype=int), 0, 181)  # Angles in degrees (int)
    particle_count = np.bincount(angle_int.flatten(), minlength=181)
    particle_count[0] = 0

    volume = (4/3) * np.pi * (0.5 * max(norm1.max(), norm2.max()))**3
    dens = particle_count / (volume / 181)  # particle count to density
    dens /= len(A) * len(B)  # Correct for reference particles
    dens /= len(C) / volume  # Normalize with respect to the average density
    return dens

start = time.time()

df = pd.DataFrame(data=np.zeros(181), index=np.arange(0, 181), columns=['ADF'])
for xyz in xyz_array:
    df['ADF'] += get_adf(xyz[idx_dict[at1]], xyz[idx_dict[at1]], xyz[idx_dict[at1]])
df['ADF'] /= len(xyz_array)

print(time.time() - start)

df.plot()
