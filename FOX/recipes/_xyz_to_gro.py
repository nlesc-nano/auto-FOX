"""Interconvert between .xyz and .gro files.

Examples
--------
This recipe is available from the command line via the ``FOX.recipes.xyz_to_gro`` entry point:

.. code:: bash

    > FOX.recipes.xyz_to_gro file.xyz file.gro


Index
-----
.. currentmodule:: FOX.recipes
.. autosummary::
    xyz_to_gro
    gro_to_xyz

API
---
.. autofunction:: xyz_to_gro
.. autofunction:: gro_to_xyz

"""

from __future__ import annotations

import os
import argparse
import itertools
import warnings

import FOX
from FOX.utils import LicenseAction

__all__ = ["xyz_to_gro", "gro_to_xyz"]


def xyz_to_gro(
    xyz_path: str | os.PathLike[str] | FOX.MultiMolecule,
    gro_path: str | os.PathLike[str],
) -> None:
    """Convert the passed .xyz file into a .gro file.

    Parameters
    ----------
    xyz_path : path-like object
        The name of the to-be read .xyz file.
    gro_path : path-like object
        The name of the to-be created .gro file.

    """
    if isinstance(xyz_path, FOX.MultiMolecule):
        mol = xyz_path
    else:
        mol = FOX.MultiMolecule.from_xyz(xyz_path, read_comment=True)

    if mol.shape[1] != 1:
        warnings.warn(
            "The passed .xyz contains multiple molecules; "
            "only the first will be written to the .gro file"
        )
    mol.as_gro(gro_path)


def gro_to_xyz(
    gro_path: str | os.PathLike[str],
    xyz_path: str | os.PathLike[str],
) -> None:
    """Convert the passed .xyz file into a .gro file.

    Parameters
    ----------
    gro_path : path-like object
        The name of the to-be created .gro file.
    xyz_path : path-like object
        The name of the to-be read .xyz file.

    """
    with open(gro_path, "r", encoding="utf8") as f_inp, open(xyz_path, "w", encoding="utf8") as f_out:  # noqa: E501
        header = next(f_inp)
        atom_count_str = next(f_inp)
        try:
            atom_count = int(atom_count_str)
        except ValueError as ex:
            raise ValueError(f"Invalid atom count: {atom_count_str}") from ex

        f_out.write(f"{atom_count_str}")
        f_out.write(f"{header}")
        for item in itertools.islice(f_inp, atom_count):
            f_out.write(f"{item[10:15]} {item[20:44]}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        usage="FOX.recipes.xyz_to_gro file.xyz file.gro",
        description=__doc__ if __doc__ is None else __doc__.split("\n")[0],
    )
    parser.add_argument("--license", dest="license", action=LicenseAction)
    parser.add_argument("--version", action="version", version=f"%(prog)s {FOX.__version__}")
    parser.add_argument("xyz_path", help="The .xyz file")
    parser.add_argument("gro_path", help="The .gro file")
    parser.add_argument(
        "--gro_to_xyz", action="store_true",
        help="Whether to convert the .gro file into .xyz rather than the other way around"
    )

    args = parser.parse_args()
    if args.gro_to_xyz:
        gro_to_xyz(args.gro_path, args.xyz_path)
    else:
        xyz_to_gro(args.xyz_path, args.gro_path)


if __name__ == "__main__":
    main()
