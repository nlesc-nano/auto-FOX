""" A module for constructing radial distribution functions. """

__all__ = ['get_rdf_lowmem', 'get_rdf']

import numpy as np
import pandas as pd


def get_rdf_df(dr, r_max, atom_pairs):
    """ Construct and return a pandas dataframe filled with zeros.
    :parameter float dr: The integration step-size in Angstrom, *i.e.* the distance between
        concentric spheres.
    :parameter float r_max: The maximum to be evaluated interatomic distance.
    :parameter atom_pairs: An list of 2-tuples representing the keys of the dataframe.
    :type atom_pairs: list [tuple [str]]"""
    # Prepare the DataFrame arguments
    shape = 1 + int(r_max / dr), len(atom_pairs)
    index = np.arange(0, r_max + dr, dr)
    try:  # If **atom_pairs** consists of atomic symbols
        columns = [at1 + ' ' + at2 for at1, at2 in atom_pairs]
    except TypeError:  # If **atom_pairs** consists of atomic indices
        columns = ['series ' + str(i) for i, _ in enumerate(atom_pairs, 1)]

    # Create and return the DataFrame
    df = pd.DataFrame(np.zeros(shape), index=index, columns=columns)
    df.columns.name = 'Atom pairs'
    df.index.name = 'r  /  Ångström'
    return df


def get_rdf(dist, dr=0.05, r_max=12.0):
    """ Calculate and return the radial distribution function (RDF) based on the 3D distance matrix
    **dist**.

    :parameter dist: A 3D array representing *m* distance matrices of *n* by *k* atoms.
    :type dist: *m*n*k* |np.ndarray|_ [|np.float64|_]
    :parameter float dr: The integration step-size in Angstrom, *i.e.* the distance between
        concentric spheres.
    :parameter float r_max: The maximum to be evaluated interatomic distance.
    :return: An array with the resulting radial distribution function.
    :rtype: 1D |np.ndarray|_ [|np.float64|_] of length 1 + **r_max** / **dr**.
    """
    r = np.arange(0, r_max + dr, dr)
    idx_max = 1 + int(r_max / dr)
    int_step = 4 * np.pi * dr * r**2
    int_step[0] = np.nan

    dens_mean = dist.shape[2] / ((4/3) * np.pi * (0.5 * dist.max(axis=(1, 2)))**3)
    dist /= dr
    dist = dist.astype(np.int32, copy=False)
    dist.shape = dist.shape[0], dist.shape[1] * dist.shape[2]

    dens = np.array([np.bincount(i, minlength=idx_max)[:idx_max] for i in dist], dtype=float)
    dens /= dist.shape[1]
    dens /= int_step
    dens /= dens_mean[:, None]
    dens[:, 0] = 0.0
    return np.average(dens, axis=0)


def get_rdf_lowmem(dist, dr=0.05, r_max=12.0):
    """ Calculate and return the radial distribution function (RDF) based on the 2D distance matrix
    **dist**. A more memory efficient implementation of :func:`FOX.functions.rdf.get_rdf`,
    which operates on a 3D dstance matrix.

    :parameter dist: A 2D array representing a distance matrix of *n* by *k* atoms.
    :type dist: *1*m*n* |np.ndarray|_ [|np.float64|_]
    :parameter float dr: The integration step-size in Angstrom, *i.e.* the distance between
        concentric spheres.
    :parameter float r_max: The maximum to be evaluated interatomic distance.
    :return: An array with the resulting radial distribution function.
    :rtype: 1D |np.ndarray|_ [|np.float64|_] of length 1 + **r_max** / **dr**.
    """
    idx_max = 1 + int(r_max / dr)
    dist_int = np.array(dist / dr, dtype=int).flatten()

    # Calculate the average particle density N / V
    # The diameter of the spherical volume (V) is defined by the largest inter-particle distance
    dens_mean = dist.shape[2] / ((4/3) * np.pi * (0.5 * dist.max())**3)

    # Count the number of occurances of each (rounded) distance
    dens = np.bincount(dist_int, minlength=idx_max)[:idx_max]

    # Correct for the number of reference atoms
    dens = dens / dist.shape[1]
    dens[0] = np.nan

    # Convert the particle count into a partical density
    r = np.arange(0, r_max + dr, dr)
    dens /= (4 * np.pi * r**2 * dr)

    # Normalize and return the particle density
    return dens / dens_mean
