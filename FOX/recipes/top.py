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
from collections.abc import Iterable, Generator
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
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


def _get_bonds(rtf: FOX.RTFContainer) -> pd.DataFrame:
    bonds = rtf.bond.reset_index(drop=False)
    bonds["func"] = 1
    return bonds


def _get_angles(rtf: FOX.RTFContainer) -> pd.DataFrame:
    angles = rtf.angles.reset_index(drop=False)
    angles["func"] = 1
    return angles


def _get_dihedrals(rtf: FOX.RTFContainer) -> pd.DataFrame:
    dihedrals = (
        rtf.dihedrals.reset_index(drop=False),
        rtf.impropers.reset_index(drop=False),
    )
    dihedrals[0]["func"], dihedrals[1]["func"] = 9, 2
    return pd.concat(dihedrals)


def _get_bond_types(prm: FOX.PRMContainer) -> None | dict[int, pd.DataFrame]:
    if prm.bonds is None:
        return None

    bondtypes = prm.bonds.reset_index(drop=False)
    bondtypes.insert(2, "func", 1)
    bondtypes[3] /= 10
    bondtypes.columns = FOX.TOPContainer.DF_DICT_DTYPES["bondtypes"][1].names
    return {1: bondtypes}


def _get_angle_types(prm: FOX.PRMContainer) -> None | dict[int, pd.DataFrame]:
    if prm.angles is None:
        return None

    angletypes = prm.angles.reset_index(drop=False)
    angletypes.insert(3, "func", 5)
    angletypes.loc[np.isnan(angletypes[5]), 5] = 0.0
    angletypes.loc[np.isnan(angletypes[6]), 6] = 0.0
    angletypes.columns = FOX.TOPContainer.DF_DICT_DTYPES["angletypes"][5].names
    return {5: angletypes}


def _get_dihedral_types(prm: FOX.PRMContainer) -> None | dict[int, pd.DataFrame]:
    if prm.dihedrals is None and prm.impropers is None:
        return None

    dihedraltypes: dict[int, pd.DataFrame] = {}
    if prm.dihedrals is not None:
        _dihedraltypes = prm.dihedrals.reset_index(drop=False)
        _dihedraltypes.insert(4, "func", 9)
        _dihedraltypes[[5, 6, 4]] = _dihedraltypes[[4, 5, 6]]
        _dihedraltypes.columns = FOX.TOPContainer.DF_DICT_DTYPES["dihedraltypes"][9].names
        dihedraltypes[9] = _dihedraltypes
    if prm.impropers is not None:
        _impropertypes = prm.impropers.reset_index(drop=False)
        _impropertypes.insert(4, "func", 2)
        del _impropertypes[5]
        _impropertypes[4], _impropertypes[6] = _impropertypes[6], _impropertypes[4]
        _impropertypes.columns = FOX.TOPContainer.DF_DICT_DTYPES["dihedraltypes"][2].names
        dihedraltypes[2] = _impropertypes
    return dihedraltypes


def _get_nonbonded_params(prm: FOX.PRMContainer) -> None | dict[int, pd.DataFrame]:
    if prm.nbfix is None:
        return None

    nonbond_params = prm.nbfix.reset_index(drop=False)
    del nonbond_params[4]
    del nonbond_params[5]
    nonbond_params[3] *= 2 / 2**(1/6)
    nonbond_params[[2, 3]] = nonbond_params[[3, 2]]
    nonbond_params.insert(2, "func", 1)
    nonbond_params.columns = FOX.TOPContainer.DF_DICT_DTYPES["nonbond_params"][1].names
    return {1: nonbond_params}


def _yield_atoms(
    rtf: FOX.RTFContainer,
    atomtypes_dict: dict[str, _AtomTypesDict],
) -> Generator[tuple[Any, ...], None, None]:
    for res in rtf.residues:
        for _, (at1, at_type, charge) in rtf.atom.loc[[res], :].iterrows():
            dct = atomtypes_dict[at_type]
            yield (
                res,
                at1,
                at_type,
                1,
                res,
                dct["symbol"],
                1,
                charge,
                dct["mass"],
            )


def _yield_atomtypes(
    atomtypes_dict: dict[str, _AtomTypesDict],
) -> Generator[tuple[Any, ...], None, None]:
    for at_type, dct in atomtypes_dict.items():
        yield (
            at_type,
            dct["atnum"],
            dct["mass"],
            dct["charge"],
            "A",
            dct.get("sigma", 0.0),
            dct.get("epsilon", 0.0),
        )


def _get_atoms_and_atom_types(
    prm: FOX.PRMContainer,
    rtf: FOX.RTFContainer,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    atomtypes_dict: dict[str, _AtomTypesDict] = {}
    charge_dict = rtf.collapse_charges()
    for _, (at_type, mass, symbol) in rtf.mass.iterrows():
        atomtypes_dict[at_type] = {
            "atnum": PT.get_atomic_number(symbol),
            "symbol": symbol,
            "mass": mass,
            "charge": charge_dict[at_type],
        }

    if prm.nonbonded is not None:
        for at_type, series in prm.nonbonded.iterrows():
            atomtypes_dict[at_type].update({
                "epsilon": series[2],
                "sigma": series[3] * 2 / 2**(1/6),
            })

    atomtypes = np.array(
        list(_yield_atomtypes(atomtypes_dict)),
        dtype=FOX.TOPContainer.DF_DTYPES["atomtypes"],
    )

    atoms = np.array(
        list(_yield_atoms(rtf, atomtypes_dict)),
        dtype=FOX.TOPContainer.DF_DTYPES["atoms"]
    )
    return pd.DataFrame.from_records(atomtypes), pd.DataFrame.from_records(atoms)


def create_top(
    *,
    mol_count: Iterable[SupportsIndex],
    rtf_files: Iterable[StrPath | FOX.RTFContainer],
    prm_files: Iterable[StrPath | FOX.PRMContainer],
) -> FOX.TOPContainer:
    """Construct a :class:`FOX.TOPContainer` object from the passed CHARMM .rtf and .prm files.

    Examples
    --------
    .. code:: python

        >>> from FOX.recipes import create_top

        >>> output_path: str = ...
        >>> rtf_files = ["ligand1.rtf", "ligand2.rtf"]
        >>> prm_files = ["ligand1.prm", "ligand2.prm"]
        >>> mol_count = [30, 15]  # 30 ligand1 residues and 15 ligand2 residues

        >>> top = create_top(
        ...     mol_count=mol_count, rtf_files=rtf_files, prm_files=prm_files,
        ... )
        >>> top.to_file(output_path)

    Parameters
    ----------
    mol_count : ``list[int]``
        The number of molecules of a given residue.
        Note that rtf files *may* contain multiple residues.
    rtf_files : list of path-like objects
        The names of all to-be converted .rtf files
    prm_files : list of path-like and/or :class:`FOX.PRMContainer` objects
        The names of all to-be converted .prm files

    Returns
    -------
    FOX.TOPContainer
        A new .top container object

    """
    # Parse and concatenate the RTF files
    rtf_list: list[FOX.RTFContainer] = []
    for r in rtf_files:
        rtf_list.append(r if isinstance(r, FOX.RTFContainer) else FOX.RTFContainer.from_file(r))
    assert len(rtf_list) > 0
    rtf = rtf_list[0].concatenate(rtf_list[1:])

    # Parse the PRM files
    prm_list: list[FOX.PRMContainer] = []
    for p in prm_files:
        prm_list.append(p if isinstance(p, FOX.PRMContainer) else FOX.PRMContainer.read(p))
    assert len(prm_list) > 0
    prm = prm_list[0].concatenate(prm_list[1:])

    atomtypes, atoms = _get_atoms_and_atom_types(prm, rtf)
    return FOX.TOPContainer(
        defaults=pd.DataFrame(
            data=[[1, 2, "yes", 1.0, 1.0]],
            index=[0], columns=FOX.TOPContainer.DF_DTYPES["defaults"].names,
        ),
        system=pd.DataFrame(
            data=[["system1"]],
            index=[0], columns=FOX.TOPContainer.DF_DTYPES["system"].names,
        ),
        molecules=pd.DataFrame(
            data=zip(rtf.residues, (operator.index(i) for i in mol_count)),
            columns=FOX.TOPContainer.DF_DTYPES["molecules"].names,
        ),
        moleculetype=pd.DataFrame(
            data=[(i, 3) for i in rtf.residues],
            columns=FOX.TOPContainer.DF_DTYPES["moleculetype"].names,
        ),
        bonds=_get_bonds(rtf),
        angles=_get_angles(rtf),
        dihedrals=_get_dihedrals(rtf),
        bondtypes=_get_bond_types(prm),
        angletypes=_get_angle_types(prm),
        dihedraltypes=_get_dihedral_types(prm),
        nonbond_params=_get_nonbonded_params(prm),
        atomtypes=atomtypes,
        atoms=atoms,
    )
