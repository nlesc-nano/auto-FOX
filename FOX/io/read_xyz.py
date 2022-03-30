"""A module for reading multi-xyz files.

Index
-----
.. currentmodule:: FOX.io.read_xyz
.. autosummary::
    XYZError
    read_multi_xyz
    get_comments
    validate_xyz

API
---
.. autoexception:: XYZError
.. autofunction:: read_multi_xyz
.. autofunction:: get_comments
.. autofunction:: validate_xyz

"""

import reprlib
from typing import Tuple, Dict, Iterable, List, Iterator, Generator, overload
from itertools import islice, chain

import numpy as np
from scm.plams import Units
from nanoutils import Literal, PathType, group_by_values

__all__ = ['read_multi_xyz']


class XYZError(OSError):
    """Raise when there are issues related to parsing .xyz files."""


XYZoutput1 = Tuple[np.ndarray, Dict[str, List[int]]]
XYZoutput2 = Tuple[np.ndarray, Dict[str, List[int]], np.ndarray]


@overload
def read_multi_xyz(filename: PathType, return_comment: Literal[True] = ..., unit: str = ...) -> XYZoutput2: ...  # noqa: E501
@overload
def read_multi_xyz(filename: PathType, return_comment: Literal[False], unit: str = ...) -> XYZoutput1: ...  # noqa: E501
def read_multi_xyz(filename, return_comment=True, unit='angstrom'):  # noqa: E302
    r"""Read a (multi) .xyz file.

    Parameters
    ----------
    filename : str
        The path+filename of a (multi) .xyz file.
    return_comment : bool
        Whether or not the comment line in each Cartesian coordinate block should be returned.
        Returned as a 1D array of strings.
    unit : :class:`str`
        The unit of the to-be returned array.

    Returns
    -------
    :math:`m*n*3` |np.ndarray|_ [|np.float64|_], |dict|_ [|str|_, |list|_ [|int|_]] and\
    (optional) :math:`m` |np.ndarray|_ [|str|_]:
        * A 3D array with Cartesian coordinates of :math:`m` molecules with :math:`n` atoms.
        * A dictionary with atomic symbols as keys and lists of matching atomic indices as values.
        * (Optional) a 1D array with :math:`m` comments.

    Raises
    ------
    :exc:`.XYZError`
        Raised when issues are encountered related to parsing .xyz files.

    """
    # Define constants and construct a dictionary: {atomic symbols: [atomic indices]}
    with open(filename, 'r') as f:
        atom_count = _get_atom_count(f)
        idx_dict = _get_idx_dict(f, atom_count)

    with open(filename, 'r') as f:
        iterator = chain.from_iterable(_xyz_generator(f, atom_count))
        try:
            xyz = np.fromiter(iterator, dtype=np.float64)
        except ValueError as ex:  # Failed to parse the .xyz file
            raise XYZError("Failed to parse the passed xyz file") from ex
    xyz.shape = (-1, atom_count, 3)

    if unit != 'angstrom':
        xyz *= Units.conversion_ratio('angstrom', unit)

    if return_comment:
        return xyz, idx_dict, get_comments(filename, atom_count)
    else:
        return xyz, idx_dict


def _xyz_generator(f: Iterable[str], atom_count: int) -> Generator[Iterator[str], None, None]:
    """Create a Cartesian coordinate generator for :func:`.read_multi_xyz`."""
    stop = 1 + atom_count
    for at_count in f:
        # Allow for empty lines between xyz blocks
        if not at_count.strip():
            continue
        yield chain.from_iterable(at.split()[1:] for at in islice(f, 1, stop))


def get_comments(filename: PathType, atom_count: int) -> np.ndarray:
    """Read and returns all comment lines in an xyz file.

    A single comment line should be located under the atom count of each molecule.

    Parameters
    ----------
    filename : str
        The path+filename of a (multi) .xyz file.

    atom_count : int
        The number of atoms per molecule.

    Returns
    -------
    :math:`m` |np.ndarray|_ [|str|_]:
        A 1D array with :math:`m` comments extracted from **filename**.

    """
    step = 2 + atom_count
    with open(filename, 'r') as f:
        iterator = islice(f, 1, None, step)  # Generator slicing
        return np.array([i.rstrip() for i in iterator])


def _get_atom_count(f: Iterator[str]) -> int:
    """Extract the number of atoms per molecule from the first line in an .xyz file.

    Parameters
    ----------
    f : |io.TextIOWrapper|_
        An opened .xyz file.

    Returns
    -------
    |int|_:
        The number of atoms per molecule.

    Raises
    ------
    :exc:`.XYZError`
        Raised when issues are encountered related to parsing .xyz files.

    """
    ret = next(f)
    try:
        return int(ret)
    except ValueError as ex:
        err = (f"{reprlib.repr(ret)} is not a valid integer, the first line in an .xyz file "
               "should contain the number of atoms per molecule")
        raise XYZError(err) from ex


def _get_idx_dict(f: Iterable[str], atom_count: int) -> Dict[str, List[int]]:
    """Extract atomic symbols and matching atomic indices from **f**.

    Parameters
    ----------
    f : |io.TextIOWrapper|_
        An opened .xyz file.

    atom_count : int
        The number of atoms per molecule.

    Returns
    -------
    |dict|_ [|str|_, |list|_ [|int|_]]:
        A dictionary with atomic symbols and a list of matching atomic indices.

    """
    stop = 1 + atom_count
    atom_list = [at.split(maxsplit=1)[0].capitalize() for at in islice(f, 1, stop)]
    return group_by_values(enumerate(atom_list))
