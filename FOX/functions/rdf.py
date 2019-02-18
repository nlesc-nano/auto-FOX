""" A module for constructing radial distribution functions. """

__all__ = ['get_all_radial', 'get_radial_distr', 'get_radial_distr_lowmem']

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def get_all_radial(xyz_array, idx_dict, atoms=None, dr=0.05, r_max=12.0, low_mem=False):
    """ Calculate and return the radial distribution functions (RDFs) for all possible atom-pairs
    in **atoms** as a dataframe. Accepts both 2d and 3d arrays of cartesian coordinates as input.

    :parameter xyz_array: A 3D or 2D array with cartesian coordinates of *m*
        molecules consisting of *n* atoms.
    :type xyz_array: *m*n*3* or *n*3* |np.ndarray|_ [|np.float64|_]
    :parameter idx_dict: A dictionary with the atomic symbols in **xyz_array** as keys
        and matching atomic indices as values.
    :type idx_dict: |dict|_ (keys: |str|_, values: |list|_ [|int|_])
    :parameter atoms: A tuple of atomic symbols. RDFs will be calculated for all
        possible atom-pairs in **atoms**. If *None*, calculate RDFs for all possible atom-pairs
        in the keys of **idx_dict** (*i.e.* all possible atoms pairs in the molecule).
    :type atoms: None or tuple [str]
    :parameter float dr: The integration step-size in Angstrom, *i.e.* the distance between
        concentric spheres.
    :parameter float r_max: The maximum to be evaluated interatomic distance.
    :parameter float low_mem: If *True*, use a slower but more memory efficient method for
        constructing the RDFs.
    :return: A dataframe of radial distribution functions, averaged over all conformations in
        **xyz_array**. Keys are of the form: at_symbol1 + ' ' + at_symbol2 (*e.g.* 'Cd Cd').
        The radius is used as index.
    :rtype: |pd.DataFrame|_ (keys: |str|_, values: |np.float64|_, indices: |np.float64|_).
    """
    # Make sure we're dealing with a 3d array
    if len(xyz_array.shape) == 2:
        xyz_array = xyz_array[None, :, :]

    # If *atoms* is None: extract atomic symbols from they keys of *idx_dict*
    atoms = atoms or tuple(idx_dict.keys())

    # Construct a list of 2-tuples containing all unique atom pairs
    atom_pairs = [(at1, at2) for i, at1 in enumerate(atoms) for at2 in atoms[i:]]

    # Construct an empty dataframe with appropiate dimensions, indices and keys
    df = get_empty_df(dr, r_max, atom_pairs)

    # Fill the dataframe with RDF's, summed over all conformations in mol_list
    if low_mem:
        # Slower low mem approach
        for xyz in xyz_array:
            for at1, at2 in atom_pairs:
                df[at1 + ' ' + at2] += get_radial_distr_lowmem(xyz[idx_dict[at1]],
                                                               xyz[idx_dict[at2]],
                                                               dr=dr, r_max=r_max)
        df.iloc[0] = 0.0
        df /= xyz_array.shape[0]
    else:
        # Faster high mem approach
        for at1, at2 in atom_pairs:
            df[at1 + ' ' + at2] = get_radial_distr(xyz_array[:, idx_dict[at1]],
                                                   xyz_array[:, idx_dict[at2]],
                                                   dr=dr, r_max=r_max)

    return df


def get_empty_df(dr, r_max, atom_pairs):
    """ Construct and return a pandas dataframe filled with zeros.
    :parameter float dr: The integration step-size in Angstrom, *i.e.* the distance between
        concentric spheres.
    :parameter float r_max: The maximum to be evaluated interatomic distance.
    :parameter atom_pairs: An list of 2-tuples representing the keys of the dataframe.
    :type atom_pairs: list [tuple [str]]"""
    # Prepare the DataFrame arguments
    shape = 1 + int(r_max / dr), len(atom_pairs)
    index = np.arange(0, r_max + dr, dr)
    columns = [at1 + ' ' + at2 for at1, at2 in atom_pairs]

    # Create and return the DataFrame
    df = pd.DataFrame(np.zeros(shape), index=index, columns=columns)
    df.columns.name = 'Atom pairs'
    df.index.name = 'r  /  Ångström'
    return df


def get_radial_distr(array1, array2, dr=0.05, r_max=12.0):
    """ Calculate and return the radial distribution function (RDF) between atoms in
    **array1** and **array2** as an array.

    :parameter array1: A 3D array of cartesian coordinates of *m* molecules with *n* atoms.
    :type array1: *m*n*3* |np.ndarray|_ [|np.float64|_]
    :parameter array2: A 3D array of cartesian coordinates of *m* molecules with *k* atoms.
    :type array2: *m*k*3* |np.ndarray|_ [|np.float64|_]
    :parameter float dr: The integration step-size in Angstrom, *i.e.* the distance between
        concentric spheres.
    :parameter float r_max: The maximum to be evaluated interatomic distance.
    :return: An array with the resulting radial distribution function.
    :rtype: 1D |np.ndarray|_ [|np.float64|_] of length 1 + **r_max** / **dr**.
    """
    r = np.arange(0, r_max + dr, dr)
    idx_max = 1 + int(r_max / dr)
    int_step = 4 * np.pi * r**2 * dr
    int_step[0] = np.nan

    dist = np.array([cdist(i, j) for i, j in zip(array1, array2)])
    dist_int = np.array(dist / dr, dtype=int)
    dist_int.shape = array1.shape[0], array1.shape[1] * array2.shape[1]

    dens_mean = array2.shape[1] / ((4/3) * np.pi * (0.5 * dist.max(axis=(1,2)))**3)
    dens = np.array([np.bincount(i)[:idx_max] for i in dist_int])

    # Plan B: Create a padded array if r_max > dist.max(axis=(1,2)).min()
    if dens.dtype == 'O':
        dens = pad_bincount(dist, dens, idx_max, dr=dr, r_max=r_max)

    dens = dens / array1.shape[1]
    dens /= int_step

    ret = dens / dens_mean[:, None]
    ret[:, 0] = 0.0
    return np.average(ret, axis=0)


def pad_bincount(dist, dens, idx_max, dr=0.05, r_max=12.0):
    """ Create and fill a padded array. """
    error = 'Warning: r_max (' + str(r_max) + 'is larger than one of the observed maximum '
    error += 'distances (' + str(dist.max(axis=(1,2)).min()) + ' to '
    error += str(dist.max(axis=(1,2)).max()) + ' A)'
    print(error)
    print('A padded array will be created')

    shape = idx_max, dist.shape[0]
    dens_tmp = np.zeros(shape, dtype=int)
    for i, j in enumerate(dens):
        dens_tmp[i, 0:len(j)] = j
    return dens_tmp


def get_radial_distr_lowmem(array1, array2, dr=0.05, r_max=12.0):
    """ Calculate and return the radial distribution function (RDF) between atoms in
    **array1** and **array2** as an array. A more memory efficient implementation of
    :func:`FOX.functions.rdf.get_radial_distr`; operating on two 2D arrays instead of two 3D arrays.

    :parameter array1: A 2D array of cartesian coordinates of *n* atoms.
    :type array1: *n*3* |np.ndarray|_ [|np.float64|_]
    :parameter array2: A 2D array of cartesian coordinates of *k* atoms.
    :type array2: *k*3* |np.ndarray|_ [|np.float64|_]
    :parameter float dr: The integration step-size in Angstrom, *i.e.* the distance between
        concentric spheres.
    :parameter float r_max: The maximum to be evaluated interatomic distance.
    :return: An array with the resulting radial distribution function.
    :rtype: 1D |np.ndarray|_ [|np.float64|_] of length 1 + **r_max** / **dr**.
    """
    idx_max = 1 + int(r_max / dr)
    dist = cdist(array1, array2)
    dist_int = np.array(dist / dr, dtype=int).flatten()

    # Calculate the average particle density N / V
    # The diameter of the spherical volume (V) is defined by the largest inter-particle distance
    dens_mean = len(array2) / ((4/3) * np.pi * (0.5 * dist.max())**3)

    # Count the number of occurances of each (rounded) distance
    dens = np.bincount(dist_int)[:idx_max]

    # Correct for the number of reference atoms
    dens = dens / len(array1)
    dens[0] = np.nan

    # Convert the particle count into a partical density
    r = np.arange(0, r_max + dr, dr)
    try:
        dens /= (4 * np.pi * r**2 * dr)
    except ValueError:
        # Plan b: Pad the array with zeros if r_max is larger than dist.max()
        zeros = np.zeros(len(r))
        zeros[:] = np.nan
        zeros[0:len(dens)] = dens
        dens = zeros / (4 * np.pi * r**2 * dr)

    # Normalize and return the particle density
    return dens / dens_mean
