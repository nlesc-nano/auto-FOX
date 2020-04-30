"""
FOX.armc.guess
==============

A module with functions for guessing ARMC parameters.

"""

from functools import wraps
from itertools import chain
from typing import (
    Union,
    Type,
    Callable,
    TypeVar,
    Iterable,
    Mapping,
    Tuple,
    Optional,
    Dict,
    Set,
    Sequence,
    cast,
    TYPE_CHECKING
)

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from ..type_hints import Literal
from ..utils import prepend_exception
from ..io import PSFContainer, PRMContainer
from ..ff.lj_param import estimate_lj
from ..ff.lj_uff import UFF_DF
from ..ff.shannon_radii import SIGMA_DF
from ..ff.lj_dataframe import LJDataFrame

if TYPE_CHECKING:
    from os import PathLike
    from scm.plams import Settings  # type: ignore
    from ..classes import MultiMolecule
else:
    from ..type_alias import PathLike, Settings, MultiMolecule

__all__ = ['guess_param']

FT = TypeVar('FT', bound=Callable)

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

ION_SET = frozenset({'ionic_radius', 'ion_radius', 'ionic_radii', 'ion_radii'})
CRYSTAL_SET = frozenset({'crystal_radius', 'crystal_radii'})


def guess_param(mol_list: Sequence[MultiMolecule], settings: Settings, param: Param,
                mode: Mode = 'rdf',
                prm: Union[None, str, bytes, PathLike, PRMContainer] = None,
                psf_list: Optional[Sequence[Union[str, bytes, PathLike, PSFContainer]]] = None
                ) -> Dict[Tuple[str, str], float]:
    """Estimate all missing forcefield parameters.

    Parameters
    ----------
    mol_list : :class:`~collections.abc.Sequence` [:class:`~FOX.MultiMolecule`]
        A sequence of molecules.

    settings : :class:`~scm.plams.core.settings.Settings`
        The CP2K input settings.

    param : :class:`str`
        The to-be estimated parameter.
        Accepted values are ``"epsilon"`` and ``"sigma"``.

    mode : :class:`str`
        The procedure for estimating the parameters.
        Accepted values are ``"rdf"``, ``"uff"``, ``"crystal_radius"`` and ``"ion_radius"``.

    prm : path-like_ or :class:`~FOX.PRMContainer`, optional
        An optional .prm file.

    psf_list : :class:`~collections.abc.Sequence` [path-like_ or :class:`~FOX.PSFContainer`], optional
        An optional list of .psf files.

    .. _`path-like`: https://docs.python.org/3/glossary.html#term-path-like-object

    Returns
    -------
    :class:`dict` [:class:`tuple` [:class:`str`, :class:`str`], :class:`float`]
        A dictionary with atom-pairs as keys and the estimated parameters as values.

    """  # noqa: E501
    # Validate param
    param = param.lower()  # type: ignore
    if param not in {'epsilon', 'sigma'}:
        raise ValueError("Invalid 'param' value: {param!r:.100}")

    # Construct a set with all valid atoms types
    if psf_list is not None:
        for mol, p in zip(mol_list, psf_list):
            psf: PSFContainer = PSFContainer.read(p) if not isinstance(p, PSFContainer) else p
            mol.atoms = psf.to_atom_dict()
    atoms: Set[str] = set(chain.from_iterable(mol.atoms.keys() for mol in mol_list))

    # Construct a DataFrame and update it with all available parameters
    df = LJDataFrame(np.nan, index=atoms)
    df.overlay_cp2k_settings(settings)
    if prm is not None:
        prm_: PRMContainer = prm if isinstance(prm, PRMContainer) else PRMContainer.read(prm)
        df.overlay_prm(prm_)
        prm_dict = _process_prm(prm_, param=param)
    else:
        prm_dict = {}

    # Extract the relevant parameter Series
    _series = df[param]
    series = _series[~_series.isnull()]

    mode = mode.lower()  # type: ignore
    if mode == 'rdf':
        rdf(series, mol_list)
    elif mode == 'uff':
        uff(series, prm_dict)
    elif mode in ION_SET:
        ion_radius(series, prm_dict)
    elif mode in CRYSTAL_SET:
        crystal_radius(series, prm_dict)
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


@prepend_exception('No reference parameters available for atom type: ', exception=KeyError)
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
        if i in prm_mapping:
            value_i = prm_mapping[i]
        else:
            value_i = ref_mapping[i]

        if j in prm_mapping:
            value_j = prm_mapping[j]
        else:
            value_j = ref_mapping[j]

        series[i, j] = func(value_i, value_j)


def _process_prm(prm: PRMContainer, param: Param) -> Dict[str, float]:
    r"""Extract a dict from **prm** with all :math:`\varepsilon` or :math:`\sigma` values."""
    if prm.nonbonded is None:
        return {}
    nonbonded = prm.nonbonded[[2, 3]].copy()
    nonbonded.columns = ['epsilon', 'sigma']  # kcal/mol and Angstrom
    nonbonded['sigma'] *= 2 / 2**(1/6)  # Conversion factor between (R / 2) and sigma
    return nonbonded[param].as_dict()
