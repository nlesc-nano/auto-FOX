"""
FOX.armc.guess
==============

A module with functions for guessing ARMC parameters.

"""

import reprlib
from typing import Union, Iterable, Mapping, Tuple, List, Collection, Optional, Dict, Set, TYPE_CHECKING
from itertools import chain

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from qmflows.cp2k_utils import prm_to_df

from ..type_hints import Literal
from ..io import PSFContainer, PRMContainer
from ..ff.lj_param import estimate_lj
from ..ff.lj_uff import UFF_DF
from ..ff.shannon_radii import SIGMA_DF
from ..ff.lj_dataframe import LJDataFrame

if TYPE_CHECKING:
    from os import PathLike
    from scm.plams import Settings  # type: ignore
    from .armc import ARMC
    from ..classes import MultiMolecule
else:
    from ..type_alias import PathLike, Settings, ARMC, MultiMolecule

__all__ = ['guess_param']

Param = Literal['epsilon', 'sigma']
Mode = Literal[
    'ionic_radius',
    'ion_radius',
    'ionic_radii',
    'ion_radii',
    'rdf',
    'uff',
    'crystal_radius',
    'crystal_radii'
]


def guess_param(mol_list: Iterable[MultiMolecule], settings: Settings,
                mode: Mode = 'rdf', param: Param = 'epsilon',
                prm: Union[None, str, bytes, PathLike, PRMContainer] = None,
                psf_list: Optional[Iterable[Union[str, bytes, PathLike, PSFContainer]]] = None
                ) -> Dict[Tuple[str, str], float]:
    # Construct a set with all valid atoms types
    if psf_list is None:
        atoms: Set[str] = set(chain.from_iterable(mol.keys() for mol in mol_list))
    else:
        atoms = set()
        for _psf in psf_list:
            psf: PSFContainer = PSFContainer.read(_psf) if not isinstance(_psf, PSFContainer) else _psf
            atoms.update(psf.to_atom_dict().keys())

    # Construct a DataFrame and update it with all available parameters
    df = LJDataFrame(np.nan, index=atoms)
    df.overlay_cp2k_settings(settings)
    if prm is not None:
        df.overlay_prm(prm)

    # Extract the relevant parameter Series
    _series = df[param]
    series = _series[~_series.isnull()]

    mode = mode.lower()  # type: ignore
    if mode == 'rdf':
        rdf(...)
    elif mode == 'uff':
        uff(series, series.loc)
    elif mode in {'ionic_radius', 'ion_radius', 'ionic_radii', 'ion_radii'}:
        ion_radius(series, series.loc)
    elif mode in {'crystal_radius', 'crystal_radii'}:
        crystal_radius(series, series.loc)
    else:
        raise ValueError(f"'mode' is of invalid value: {mode!r:.100};\naccepted values: "
                         "'rdf', 'uff', 'crystal_radius' and 'ion_radius'")
    return series.as_dict()


def uff(series: pd.Series, prm_mapping: Mapping[str, float]) -> None:
    """Guess parameters in **df** using UFF parameters."""
    uff_loc = UFF_DF[series.name].loc
    _set_radii(series, prm_mapping, uff_loc)


def ion_radius(series: pd.Series, prm_mapping: Mapping[str, float]) -> None:
    """Guess parameters in **df** using ionic radii."""
    if series.name == 'epsilon':
        raise ValueError(f"'epsilon' guessing is not supported with `guess='ion_radius'`")

    ion_loc = SIGMA_DF['ionic_sigma'].loc
    _set_radii(series, prm_mapping, ion_loc)


def crystal_radius(series: pd.Series, prm_mapping: Mapping[str, float]) -> None:
    """Guess parameters in **df** using crystal radii."""
    if series.name == 'epsilon':
        raise ValueError(f"'epsilon' guessing is not supported with `guess='crystal_radius'`")

    ion_loc = SIGMA_DF['crystal_sigma'].loc
    _set_radii(series, prm_mapping, ion_loc)


def rdf(series: pd.Series, mol_list: Iterable[MultiMolecule]) -> None:
    """Guess parameters in **df** using the Boltzmann-inverted radial distribution function."""
    # Construct the RDF and guess the parameters
    rdf_gen = (mol.init_rdf() for mol in mol_list)
    for rdf in rdf_gen:
        guess = estimate_lj(rdf)
        guess.index = pd.MultiIndex.from_tuples(sorted(i.split()) for i in guess.index)
        series.update(guess[series.name], overwrite=False)


def _set_radii(series: pd.Series,
               prm_mapping: Mapping[str, float],
               ref_mapping: Mapping[str, float]) -> None:
    if series.name == 'epsilon':
        func = lambda a, b: np.abs(a * b)**0.5
    elif series.name == 'sigma':
        func = lambda a, b: (a + b) / 2
    else:
        raise ValueError(f"series.name: {series.name!r:.100}")

    for i, j in series.index:  # pd.MultiIndex
        try:
            value_i = prm_mapping[i]
        except KeyError:
            value_i = ref_mapping[i]

        try:
            value_j = prm_mapping[j]
        except KeyError:
            value_j = ref_mapping[j]

        series[i, j] = func(value_i, value_j)
