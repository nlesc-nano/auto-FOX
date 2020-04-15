"""
FOX.armc.guess
==============

A module with functions for guessing ARMC parameters.

"""

import reprlib
from typing import Union, Iterable, Mapping, Tuple, List, Collection, TYPE_CHECKING
from itertools import chain

import numpy as np
import pandas as pd

from ..type_hints import Literal
from ..io.read_psf import PSFContainer
from ..io.read_prm import PRMContainer
from ..ff.lj_param import estimate_lj
from ..ff.lj_uff import UFF_DF
from ..ff.shannon_radii import SIGMA_DF
from ..ff.lj_dataframe import LJDataFrame

if TYPE_CHECKING:
    from .armc import ARMC
    from ..classes import MultiMolecule
else:
    from ..type_alias import ARMC, MultiMolecule

__all__ = ['guess_param']

Columns = Literal['epsilon', 'sigma']

Mode = Literal['ionic_radius', 'ion_radius', 'ionic_radii', 'ion_radii',
               'rdf', 'uff', 'crystal_radius', 'crystal_radii']


def guess_param(armc: ARMC, mode: Mode = 'rdf',
                columns: Union[Columns, Iterable[Columns]] = ('epsilon', 'sigma'),
                frozen: Union[None, str, Iterable[str]] = None) -> None:
    r"""Guess Lennard-Jones parameters absent from **armc**.

    Parameters
    ----------
    armc : :class:`ARMC`
        An ARMC instances.

    mode : :class:`str`
        The guessing method.
        Accepted values are ``"uff"`` and ``"rdf"``.

        ``"uff"`` substitute all missing parameters with those from the Universal Forcefield (UFF);
        new :math:`\sigma` and :math:`\varepsilon` parameters are created using the standard
        combination rules.

        ``"rdf"`` utilizes the radial distribution function;
        the based of the first peak and the well-depth of the (Boltzmann-inverted) RDF are used
        for estimating :math:`\sigma` and :math:`\varepsilon`, respectively.

    columns : :class:`str` or :class:`~collections.abc.Iterable` [:class:`str`]
        The type of to-be guessed parameters.
        Accepted values are ``"epsilon"`` and/or ``"sigma"``.

    frozen : :class:`str` or :class:`~collections.abc.Iterable` [:class:`str`], optional
        Which parameter-types in **columns** are to-be treated as constants rather than variables.

    See Also
    --------
    :data:`UFF_DF`
        A DataFrame with UFF Lennard-Jones parameters.

    :func:`estimate_lj`
        Estimate the Lennard-Jones :math:`\sigma` and :math:`\varepsilon` parameters using an RDF.

    """
    # Parse and sort the columns
    column_list = sorted(columns) if not isinstance(columns, str) else [columns]
    df, mol_list = _process_df(armc)

    # Populate df with the guess parameters
    try:
        prm_df = _process_prm(armc)
        if mode.lower() == 'rdf':
            rdf(df, mol_list, columns=column_list)
        elif mode.lower() == 'uff':
            uff(df, prm_df.loc, columns=column_list)
        elif mode.lower() in {'ionic_radius', 'ion_radius', 'ionic_radii', 'ion_radii'}:
            ion_radius(df, prm_df.loc, columns=column_list)
        elif mode.lower() in {'crystal_radius', 'crystal_radii'}:
            crystal_radius(df, prm_df.loc, columns=column_list)
        else:
            raise AttributeError
    except AttributeError as ex:
        raise ValueError(f"'mode' is of invalid value: {reprlib.repr(mode)};\naccepted values: "
                         "'rdf', 'uff', 'crystal_radius' and 'ion_radius'") from ex

    # Update all ARMC Settings instacnes with the new parameters
    df_transform = _transform_df(df[column_list])

    # Define which guessed parameters are variables and constants
    if frozen is None:
        unfrozen = column_list
    else:
        frozen = sorted(frozen) if not isinstance(frozen, str) else [frozen]
        unfrozen = list(set(column_list).difference(frozen))

    # Update the DataFrame with containing all variables
    param = armc.param
    for ij, value in df_transform.loc[unfrozen].iterrows():
        param.loc[ij, :] = None
        param.loc[ij, ['unit', 'param', 'keys']] = value
        param.loc[ij, ['min', 'max']] = -np.inf, np.inf
    param.sort_index(inplace=True)


def _transform_df(df: pd.DataFrame) -> pd.DataFrame:
    """Cast **df** into the same structure as :attr:`ARMC.param`."""
    # Construct a new dataframe
    at_pairs = [f'{i} {j}' for i, j in df.index]
    ret = pd.DataFrame(
        index=pd.MultiIndex.from_product([df.columns, at_pairs])
    )

    # Populate the dataframe
    units = []
    for key in df:
        units += len(df) * ['[kcalmol] {:f}' if key == 'epsilon' else '[angstrom] {:f}']
    ret['unit'] = units

    prm = []
    for value in df.values:
        prm += value.tolist()
    ret['param'] = prm

    key_list = ['input', 'force_eval', 'mm', 'forcefield', 'nonbonded', 'lennard-jones']
    ret['keys'] = [key_list.copy() for _ in range(len(ret))]

    return ret


def _process_prm(armc: ARMC) -> pd.DataFrame:
    r"""Extract a DataFrame from a .prm file with all :math:`\varepsilon` and :math:`\sigma` values."""  # noqa
    prm = PRMContainer.read(armc.md_settings[0].input.force_eval.mm.forcefield.parm_file_name)
    nonbonded = prm.nonbonded[[2, 3]].copy()
    nonbonded.columns = ['epsilon', 'sigma']  # kcal/mol and Angstrom
    nonbonded['sigma'] *= 2 / 2**(1/6)  # Conversion factor between (R / 2) and sigma
    return nonbonded


def _process_df(armc: ARMC) -> Tuple[LJDataFrame, List['MultiMolecule']]:
    """Prepare a DataFrame with the to-be guessed parameters."""
    # Update all atom based on the .psf file
    mol_list = [mol.copy() for mol in armc.molecule]
    for mol, s in zip(mol_list, armc.md_settings):
        psf_name = s.input.force_eval.subsys.topology.conn_file_name
        if psf_name:
            psf = PSFContainer.read(psf_name)
            mol.atoms = psf.to_atom_dict()

    # COnstruct a dataframe with all to-be guessed parameters
    idx = set(chain.from_iterable(mol.atoms.keys() for mol in mol_list))
    df = LJDataFrame(np.nan, index=sorted(idx))
    for s in armc.md_settings:
        prm_name = s.input.force_eval.mm.forcefield.parm_file_name
        if prm_name:
            df.overlay_prm(prm_name)
        df.overlay_cp2k_settings(s)
    del df['charge']
    return df[df.isna().any(axis=1)], mol_list


def _radii(df: pd.DataFrame,
           ref_map: Mapping[str, Tuple[float, float]],
           prm_mapping: Mapping[str, Tuple[float, float]],
           columns: Collection[str] = frozenset({'epsilon', 'sigma'})) -> None:
    for i, j in df.index:  # pd.MultiIndex
        try:
            eps_i, sig_i = prm_mapping[i]
        except KeyError:
            eps_i, sig_i = ref_map[i]

        try:
            eps_j, sig_j = prm_mapping[j]
        except KeyError:
            eps_j, sig_j = ref_map[j]

        if 'epsilon' in columns:
            df.at[(i, j), 'epsilon'] = np.abs(eps_i * eps_j)**0.5
        if 'sigma' in columns:
            df.at[(i, j), 'sigma'] = ((sig_i + sig_j) / 2)


def uff(df: pd.DataFrame, prm_mapping: Mapping[str, Tuple[float, float]],
        columns: Collection[str] = frozenset({'epsilon', 'sigma'})) -> None:
    """Guess parameters in **df** using UFF parameters."""
    column_set = set(columns)
    _radii(df, UFF_DF.loc, prm_mapping, column_set)


def ion_radius(df: pd.DataFrame, prm_mapping: Mapping[str, Tuple[float, float]],
               columns: Collection[str] = frozenset({'sigma'})) -> None:
    """Guess parameters in **df** using ionic radii."""
    column_set = set(columns)
    if 'epsilon' in column_set:
        raise ValueError(f"'epsilon' guessing is not supported with `guess='ion_radius'`")

    ion_loc = SIGMA_DF[['ionic_sigma', 'ionic_sigma']].loc
    _radii(df, ion_loc, prm_mapping, column_set)


def crystal_radius(df: pd.DataFrame, prm_mapping: Mapping[str, Tuple[float, float]],
                   columns: Collection[str] = frozenset({'sigma'})) -> None:
    """Guess parameters in **df** using crystal radii."""
    column_set = set(columns)
    if 'epsilon' in column_set:
        raise ValueError(f"'epsilon' guessing is not supported with `guess='crystal_radius'`")

    ion_loc = SIGMA_DF[['crystal_sigma', 'crystal_sigma']].loc
    _radii(df, ion_loc, prm_mapping, column_set)


def rdf(df: pd.DataFrame, mol_list: Iterable[MultiMolecule],
        columns: List[str] = ['epsilon', 'sigma']) -> None:
    """Guess parameters in **df** using the Boltzmann-inverted radial distribution function."""
    # Construct the RDF and guess the parameters
    rdf_gen = (mol.init_rdf() for mol in mol_list)
    for rdf in rdf_gen:
        guess = estimate_lj(rdf)
        guess.index = pd.MultiIndex.from_tuples(sorted(i.split()) for i in guess.index)
        df.update(guess[columns], overwrite=False)