"""Recipe for creating GROMACS .top files from an .xyz and CHARMM .rtf and .str files.

Index
-----
.. currentmodule:: FOX.recipes
.. autosummary::
    create_top

API
---
.. autofunction:: create_top

"""

from __future__ import annotations

import os
import operator
import sys
import math
import functools
import itertools
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from scm.plams import PT, Molecule

import FOX
from FOX.functions.molecule_utils import get_bonds, get_angles, get_dihedrals, get_impropers

if sys.version_info >= (3, 9):
    from collections.abc import Iterator
else:
    from typing import Iterator

if TYPE_CHECKING or sys.version_info < (3, 10):
    from builtins import zip as strict_zip
else:
    strict_zip = functools.partial(zip, strict=True)

if TYPE_CHECKING:
    from typing_extensions import TypeAlias, Self, NotRequired, TypedDict, SupportsIndex

    StrPath: TypeAlias = str | os.PathLike[str]

    class _RTFDict(TypedDict):
        atnum: int
        mass: float
        charge: NotRequired[float]
        epsilon: NotRequired[float]
        sigma: NotRequired[float]

__all__ = ["create_top"]


class _FileIter(Iterator[str]):
    """Enumerate through the passed ``iterable`` and remove all empty and commented lines."""

    __slots__ = ("__weakref__", "_enumerator", "_name", "_index")

    _name: str
    _index: None | int

    @property
    def index(self) -> None | int:
        """Get the index within the current iterator."""
        return self._index

    @property
    def name(self) -> str:
        """Get the name of the iterator."""
        return self._name

    def __init__(self, iterable: Iterable[str], start=1) -> None:
        self._enumerator = ((i, j.strip()) for i, j in enumerate(iterable, start=start))
        self._name = getattr(iterable, "name", "<unknown>")
        self._index = None

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> str:
        self._index, value = next(self._enumerator)
        return value

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self._name!r} index={self._index!r}>"


def _parse_rtf(file: Iterator[str]) -> tuple[list[str], dict[str, _RTFDict]]:
    file_iter = _FileIter(file)
    prefix = f"Error parsing {file_iter.name!r}"
    dct: dict[str, _RTFDict] = {}

    try:
        i = next(file_iter)
        while not i.startswith("MASS"):
            i = next(file_iter)
    except StopIteration:
        raise ValueError(f'{prefix}: Failed to find any "MASS" blocks')

    while i.startswith("MASS"):
        args = i.partition("!")[0].split()
        if len(args) != 5:
            raise ValueError(
                f'{prefix}: Expected 5 columns in the "MASS" mass block at line {file_iter.index}'
            )
        dct[args[2]] = {"atnum": PT.get_atomic_number(args[4]), "mass": float(args[3])}
        i = next(file_iter)

    try:
        while not i.startswith("ATOM"):
            i = next(file_iter)
    except StopIteration:
        raise ValueError(f'{prefix}: Failed to find any "ATOM" blocks')

    iterator = (i.partition("!")[0].split() for i in file_iter if i.startswith("ATOM"))
    atoms = []
    for args in itertools.chain([i.partition("!")[0].split()], iterator):
        if len(args) != 4:
            raise ValueError(
                f'{prefix}: Expected 4 columns in the "ATOM" mass block at line {file_iter.index}'
            )

        atom_name, charge = args[2], float(args[3])
        atoms.append(atom_name)
        charge_old = dct[atom_name].get("charge")
        if charge_old is not None and charge_old != charge:
            raise ValueError(
                f'{prefix}: Found multiple distinct charges for atom type {atom_name!r}: '
                f'{charge} and {charge_old}'
            )
        dct[atom_name]["charge"] = charge
    return atoms, dct


def create_top(
    out_path: StrPath,
    *,
    xyz_files: Mapping[str, StrPath | Molecule],
    mol_count: Iterable[SupportsIndex],
    rtf_files: Iterable[StrPath],
    prm_files: Iterable[StrPath | FOX.PRMContainer],
) -> None:
    """Construct a GROMACS .top file from the passed CHARMM .rtf and .prm files.

    Parameters
    ----------
    out_path : :term:`python:path-like` object
        The name of the to-be created output file. Will be overriden if it already exists
    xyz_files : dictionary of :term:`python:path-like` objects
        A dictionary mapping residue names to .xyz files and/or plams molecules
    mol_count : ``list[int]``
        The number
    rtf_files : list of :term:`python:path-like` objects
        The names of all to-be converted .rtf files
    prm_files : list of :term:`python:path-like` and/or :class:`FOX.PRMContainer` objects
        The names of all to-be converted .prm files

    """
    with open(out_path, "w", encoding="utf8") as f_out:
        rtf_dict: dict[str, _RTFDict] = {}
        atom_types_list: list[list[str]] = []
        for rtf_file in rtf_files:
            with open(rtf_file, "r", encoding="utf8") as f_rtf:
                atom_names, _rtf_dct = _parse_rtf(f_rtf)
                atom_types_list.append(atom_names)
                rtf_dict.update(_rtf_dct)

        prm_list: list[FOX.PRMContainer] = []
        for _prm in prm_files:
            prm = _prm if isinstance(_prm, FOX.PRMContainer) else FOX.PRMContainer.read(_prm)
            prm_list.append(prm)
            if prm.nonbonded is None:
                raise ValueError
            for atom_name, series in prm.nonbonded.iterrows():
                rtf_dict[atom_name].update({
                    "epsilon": series[2],
                    "sigma": series[3] * 2 / 2**(1/6),
                })

        f_out.write("[ defaults ]\n")
        f_out.write("1 2 yes 1.0 1.0\n")
        f_out.write("\n[ atomtypes ]\n")
        for atom_type, dct in rtf_dict.items():
            f_out.write(
                f"{atom_type:>4} {dct['atnum']:>3} {dct['mass']:>16.9f} {dct['charge']:>16.9f} "
                f"A {dct['sigma']:>16.9f} {dct['epsilon']:>16.9f}\n"
            )

        f_out.write("\n[ bondtypes ]\n")
        for prm in prm_list:
            if prm.bonds is not None:
                for ((at1, at2), series) in prm.bonds.iterrows():
                    k = series[2]
                    rij = series[3] / 10
                    f_out.write(f"{at1:>4} {at2:>4} 1 {rij:>16.9f} {k:>16.9f}\n")

        f_out.write("\n[ angletypes ]\n")
        for prm in prm_list:
            if prm.angles is not None:
                for ((at1, at2, at3), series) in prm.angles.iterrows():
                    k, theta, k_ub, rij_ub = series
                    if not math.isnan(k_ub):
                        f_out.write(
                            f"{at1:>4} {at2:>4} {at3:>4} 5 {theta:>16.9f} "
                            f"{k:>16.9f} {rij_ub:>16.9f} {k_ub:>16.9f}\n"
                        )
                    else:
                        f_out.write(f"{at1:>4} {at2:>4} {at3:>4} 1 {theta:>16.9f} {k:>16.9f}\n")

        f_out.write("\n[ dihedraltypes ]\n")
        for prm in prm_list:
            if prm.dihedrals is not None:
                for ((at1, at2, at3, at4), series) in prm.dihedrals.iterrows():
                    k, mult, phi = series
                    f_out.write(
                        f"{at1:>4} {at2:>4} {at3:>4} 9 {phi:>16.9f} {k:>16.9f} {mult:>2n}\n"
                    )

        f_out.write("\n[ dihedraltypes ]\n")
        for prm in prm_list:
            if prm.impropers is not None:
                for ((at1, at2, at3, at4), series) in prm.impropers.iterrows():
                    k, _, xi = series
                    f_out.write(f"{at1:>4} {at2:>4} {at3:>4} 2 {xi:>16.9f} {k:>16.9f}\n")

        f_out.write("\n[ nonbond_params ]\n")
        for prm in prm_list:
            if prm.nbfix is not None:
                for ((at1, at2), series) in prm.nbfix.iterrows():
                    epsilon = series[2]
                    sigma = series[3] * 2 / 2**(1/6)
                    f_out.write(f"{at1:>4} {at2:>4} 1 {sigma:>16.9f} {epsilon:>16.9f}\n")

        enumerator = enumerate(strict_zip(atom_types_list, xyz_files.items()), start=1)
        for i, (atom_types, (res_name, xyz)) in enumerator:
            mol = Molecule(xyz) if not isinstance(xyz, Molecule) else xyz
            if not mol.bonds:
                mol.guess_bonds()
            f_out.write("\n[ moleculetype ]\n")
            f_out.write(f"molecule{i:<5n} 3\n")

            f_out.write("\n[ atoms ]\n")
            for j, (at, at_name) in enumerate(strict_zip(mol, atom_types), start=1):
                charge, mass = rtf_dict[at_name]["charge"], rtf_dict[at_name]["mass"]
                f_out.write(
                    f"{j:>5n} {at_name:>4} 1 {res_name:>4} {at.symbol:>2} {1:>5n} "
                    f"{charge:>16.9f} {mass:>16.9f}\n"
                )

            bonds = get_bonds(mol)
            f_out.write("\n[ bonds ]\n")
            for at_i, at_j in bonds:
                f_out.write(f"{at_i:>5n} {at_j:>5n} {1:>5n}\n")

            angles = get_angles(mol)
            f_out.write("\n[ angles ]\n")
            for at_i, at_j, at_k in angles:
                f_out.write(f"{at_i:>5n} {at_j:>5n} {at_k:>5n} {1:>5n}\n")

            dihedrals = get_dihedrals(mol)
            f_out.write("\n[ dihedrals ]\n")
            for at_i, at_j, at_k, at_l in dihedrals:
                f_out.write(f"{at_i:>5n} {at_j:>5n} {at_k:>5n} {at_l:>5n} {9:>5n}\n")

            impropers = get_impropers(mol)
            f_out.write("\n[ dihedrals ]\n")
            for at_i, at_j, at_k, at_l in impropers:
                f_out.write(f"{at_i:>5n} {at_j:>5n} {at_k:>5n} {at_l:>5n} {2:>5n}\n")

        f_out.write("\n[ system ]\n")
        f_out.write("system1\n")

        f_out.write("\n[ molecules ]\n")
        for i, k in enumerate(mol_count, start=1):
            f_out.write(f"molecule{i:<5n} {operator.index(k)}\n")
