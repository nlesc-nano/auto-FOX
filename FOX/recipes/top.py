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
import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

from scm.plams import PT

import FOX


if TYPE_CHECKING:
    from typing_extensions import TypeAlias, NotRequired, TypedDict, SupportsIndex

    StrPath: TypeAlias = str | os.PathLike[str]

    class _AtomTypesDict(TypedDict):
        atnum: int
        symbol: str
        mass: float
        charge: float
        epsilon: NotRequired[float]
        sigma: NotRequired[float]

__all__ = ["create_top"]


def create_top(
    out_path: StrPath,
    *,
    mol_count: Iterable[SupportsIndex],
    rtf_files: Iterable[StrPath | FOX.io.RTFContainer],
    prm_files: Iterable[StrPath | FOX.PRMContainer],
) -> None:
    """Construct a GROMACS .top file from the passed CHARMM .rtf and .prm files.

    Parameters
    ----------
    out_path : :term:`python:path-like` object
        The name of the to-be created output file. Will be overriden if it already exists
    mol_count : ``list[int]``
        The number of molecules of a given residue.
        Note that rtf files *may* contain multiple residues.
    rtf_files : list of :term:`python:path-like` objects
        The names of all to-be converted .rtf files
    prm_files : list of :term:`python:path-like` and/or :class:`FOX.PRMContainer` objects
        The names of all to-be converted .prm files

    """
    with open(out_path, "w", encoding="utf8") as f_out:
        rtf_list: list[FOX.io.RTFContainer] = []
        atomtypes_dct: dict[str, _AtomTypesDict] = {}
        for _rtf in rtf_files:
            rtf = _rtf if isinstance(_rtf, FOX.RTFContainer) else FOX.RTFContainer.from_file(_rtf)
            rtf.auto_to_explicit()
            rtf_list.append(rtf)

            charge_dict = rtf.collapse_charges()
            for _, (at_type, symbol, mass) in rtf.mass.iterrows():
                atomtypes_dct[at_type] = {
                    "atnum": PT.get_atomic_number(symbol),
                    "symbol": symbol,
                    "mass": mass,
                    "charge": charge_dict[at_type],
                }

        prm_list: list[FOX.PRMContainer] = []
        for _prm in prm_files:
            prm = _prm if isinstance(_prm, FOX.PRMContainer) else FOX.PRMContainer.read(_prm)
            prm_list.append(prm)
            if prm.nonbonded is None:
                raise ValueError
            for at_type, series in prm.nonbonded.iterrows():
                atomtypes_dct[at_type].update({
                    "epsilon": series[2],
                    "sigma": series[3] * 2 / 2**(1/6),
                })

        f_out.write("[ defaults ]\n")
        f_out.write("1 2 yes 1.0 1.0\n")
        f_out.write("\n[ atomtypes ]\n")
        for at_type, dct in atomtypes_dct.items():
            f_out.write(
                f"{at_type:>4} {dct['atnum']:>3} {dct['mass']:>16.9f} {dct['charge']:>16.9f} "
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
                for ((at1, at2, at3), (k, theta, k_ub, rij_ub)) in prm.angles.iterrows():
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
                for ((at1, at2, at3, at4), (k, mult, phi)) in prm.dihedrals.iterrows():
                    f_out.write(
                        f"{at1:>4} {at2:>4} {at3:>4} 9 {phi:>16.9f} {k:>16.9f} {mult:>2n}\n"
                    )

        f_out.write("\n[ dihedraltypes ]\n")
        for prm in prm_list:
            if prm.impropers is not None:
                for ((at1, at2, at3, at4), (k, _, xi)) in prm.impropers.iterrows():
                    f_out.write(f"{at1:>4} {at2:>4} {at3:>4} 2 {xi:>16.9f} {k:>16.9f}\n")

        f_out.write("\n[ nonbond_params ]\n")
        for prm in prm_list:
            if prm.nbfix is not None:
                for ((at1, at2), series) in prm.nbfix.iterrows():
                    epsilon = series[2]
                    sigma = series[3] * 2 / 2**(1/6)
                    f_out.write(f"{at1:>4} {at2:>4} 1 {sigma:>16.9f} {epsilon:>16.9f}\n")

        enumerator = enumerate(((res, rtf) for rtf in rtf_list for res in rtf.residues), start=1)
        for i, (res, rtf) in enumerator:
            f_out.write("\n[ moleculetype ]\n")
            f_out.write(f"molecule{i:<5n} 3\n")

            f_out.write("\n[ atoms ]\n")
            for _, (at1, at_type, charge) in rtf.atom.loc[[res], :].iterrows():
                mass, symbol = atomtypes_dct[at_type]["mass"], atomtypes_dct[at_type]["symbol"]
                f_out.write(
                    f"{at1:>5n} {at_type:>4} 1 {res:>4} {symbol:>2} {1:>5n} "
                    f"{charge:>16.9f} {mass:>16.9f}\n"
                )

            f_out.write("\n[ bonds ]\n")
            for at_i, at_j in rtf.bond.loc[[res], :].values:
                f_out.write(f"{at_i:>5n} {at_j:>5n} {1:>5n}\n")

            f_out.write("\n[ angles ]\n")
            for at_i, at_j, at_k in rtf.angles.loc[[res], :].values:
                f_out.write(f"{at_i:>5n} {at_j:>5n} {at_k:>5n} {1:>5n}\n")

            f_out.write("\n[ dihedrals ]\n")
            for at_i, at_j, at_k, at_l in rtf.dihedrals.loc[[res], :].values:
                f_out.write(f"{at_i:>5n} {at_j:>5n} {at_k:>5n} {at_l:>5n} {9:>5n}\n")

            f_out.write("\n[ dihedrals ]\n")
            for at_i, at_j, at_k, at_l in rtf.impropers.loc[[res], :].values:
                f_out.write(f"{at_i:>5n} {at_j:>5n} {at_k:>5n} {at_l:>5n} {2:>5n}\n")

        f_out.write("\n[ system ]\n")
        f_out.write("system1\n")

        f_out.write("\n[ molecules ]\n")
        for i, k in enumerate(mol_count, start=1):
            f_out.write(f"molecule{i:<5n} {operator.index(k)}\n")
