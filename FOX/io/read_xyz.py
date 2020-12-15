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
from typing import Tuple, Dict, Iterable, List, Union, Iterator, Generator, overload
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
        try:
            line_count = _get_line_count(f, add=[2, atom_count])
        except UnboundLocalError:  # The .xyz file contains a single molecule
            line_count = 2 + atom_count

    # Check if mol_count is fractional, smaller than 1 or if atom_count is smaller than 1
    mol_count = line_count / (2 + atom_count)
    validate_xyz(mol_count, atom_count, filename)

    # Create the to-be returned xyz array
    shape = int(mol_count), atom_count, 3
    with open(filename, 'r') as f:
        iterator = chain.from_iterable(_xyz_generator(f, atom_count))
        try:
            xyz = np.fromiter(iterator, dtype=float, count=np.product(shape))
        except ValueError as ex:  # Failed to parse the .xyz file
            raise XYZError(str(ex)).with_traceback(ex.__traceback__)
    xyz.shape = shape  # From 1D to 3D array

    if unit != 'angstrom':
        xyz *= Units.conversion_ratio('angstrom', unit)

    if return_comment:
        return xyz, idx_dict, get_comments(filename, atom_count)
    else:
        return xyz, idx_dict


def _xyz_generator(f: Iterable[str], atom_count: int) -> Generator[Iterator[str], None, None]:
    """Create a Cartesian coordinate generator for :func:`.read_multi_xyz`."""
    stop = 1 + atom_count
    for _ in f:
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


def validate_xyz(mol_count: float, atom_count: int, filename: PathType) -> None:
    """Validate **mol_count** and **atom_count** in **xyz_file**.

    Parameters
    ----------
    mol_count : float
        The number of molecules in the xyz file.
        Expects float that is finite with integral value
        (*e.g.* :math:`5.0`, :math:`6.0` or :math:`3.0`).

    atom_count : int
        The number of atoms per molecule.

    filename : str
        The path + filename of a (multi) .xyz file.

    Raises
    ------
    :exc:`.XYZError`
        Raised when issues are encountered related to parsing .xyz files.

    """
    if not mol_count.is_integer():
        raise XYZError(f"A non-integer number of molecules was found in {filename!r}; "
                       f"mol count: {mol_count}")
    elif mol_count < 1.0:
        raise XYZError(f"No molecules were found in {filename!r}; mol count: {mol_count}")
    if atom_count < 1:
        raise XYZError(f"No atoms were found in {filename!r}; atom count: {atom_count}")


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


def _get_line_count(f: Iterable, add: Union[int, Iterable[int]] = 0) -> int:
    """Extract the total number lines from **f**.

    Parameters
    ----------
    f : |io.TextIOWrapper|_
        An opened .xyz file.

    add : int or |Iterable|_ [|int|_]
        Add a constant to the to-be returned line count.

    Returns
    -------
    |int|_:
        The total number of lines in **f**.

    """
    start = 1 + np.sum(add)
    for i, _ in enumerate(f, start):
        pass
    return i


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
