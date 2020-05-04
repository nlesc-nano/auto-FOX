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
    Iterable,
    Mapping,
    Tuple,
    Optional,
    Dict,
    Set,
    Any,
    Container,
    NoReturn,
    cast,
    overload,
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


def guess_param(mol_list: Iterable[MultiMolecule], param: Param,
                mode: Mode = 'rdf',
                cp2k_settings: Optional[Mapping] = None,
                prm: Union[None, str, bytes, PathLike, PRMContainer] = None,
                psf_list: Optional[Iterable[Union[str, bytes, PathLike, PSFContainer]]] = None
                ) -> Dict[Tuple[str, str], float]:
    """Estimate all missing forcefield parameters.

    Examples
    --------
    .. code:: python

        >>> from FOX import MultiMolecule
        >>> from FOX.armc import guess_Param

        >>> mol_list = [MultiMolecule(...), ...]
        >>> prm = str(...)
        >>> psf_list = [str(...), ...]

        >>> epsilon_dict = guess_Param(mol_list, 'epsilon', prm=prm, psf_list=psf_list)
        >>> sigma_dict = guess_Param(mol_list, 'sigma', prm=prm, psf_list=psf_list)


    Parameters
    ----------
    mol_list : :class:`~collections.abc.Iterable` [:class:`~FOX.MultiMolecule`]
        An iterable of molecules.

    param : :class:`str`
        The to-be estimated parameter.
        Accepted values are ``"epsilon"`` and ``"sigma"``.

    mode : :class:`str`
        The procedure for estimating the parameters.
        Accepted values are ``"rdf"``, ``"uff"``, ``"crystal_radius"`` and ``"ion_radius"``.

    cp2k_settings : :class:`~collections.abc.Mapping`
        The CP2K input settings.

    prm : path-like_ or :class:`~FOX.PRMContainer`, optional
        An optional .prm file.

    psf_list : :class:`~collections.abc.Iterable` [path-like_ or :class:`~FOX.PSFContainer`], optional
        An optional list of .psf files.

    .. _`path-like`: https://docs.python.org/3/glossary.html#term-path-like-object

    Returns
    -------
    :class:`dict` [:class:`tuple` [:class:`str`, :class:`str`], :class:`float`]
        A dictionary with atom-pairs as keys and the estimated parameters as values.
        Units are in kj/mol (``param="epsilon"``) or nm (``param="sigma"``).

    """  # noqa: E501
    # Validate param and mode
    param = _validate_arg(param, name='param', ref={'epsilon', 'sigma'})  # type: ignore
    mode = _validate_arg(mode, name='mode', ref=ION_SET | CRYSTAL_SET)  # type: ignore

    # Construct a set with all valid atoms types
    mol_list = [mol.copy() for mol in mol_list]
    if psf_list is not None:
        for mol, p in zip(mol_list, psf_list):
            psf: PSFContainer = PSFContainer.read(p) if not isinstance(p, PSFContainer) else p
            mol.atoms = psf.to_atom_dict()
    atoms: Set[str] = set(chain.from_iterable(mol.atoms.keys() for mol in mol_list))

    # Construct a DataFrame and update it with all available parameters
    df = LJDataFrame(np.nan, index=atoms)
    if cp2k_settings is not None:
        df.overlay_cp2k_settings(cp2k_settings)
    if prm is not None:
        prm_: PRMContainer = prm if isinstance(prm, PRMContainer) else PRMContainer.read(prm)
        df.overlay_prm(prm_)
        prm_dict = _nb_from_prm(prm_, param=param)
    else:
        prm_dict = {}

    # Extract the relevant parameter Series
    _series = df[param]
    series = _series[~_series.isnull()]
    return _guess_param(series, mode, mol_list=mol_list, prm_dict=prm_dict)  # type: ignore


def _validate_arg(value: str, name: str, ref: Container[str]) -> str:
    """Check if **value** is in **ref**.

    Returns
    -------
    :class:`str`
        The lowered version of **value**.

    """
    try:
        ret = value.lower()
        assert ret in ref
    except (TypeError, AttributeError) as ex:
        raise TypeError(f"Invalid {name!r} type: {value.__class__.__name__!r}") from ex
    except AssertionError:
        raise ValueError(f"Invalid {name!r} value: {value!r:.100}") from ex
    return ret


def _guess_param(series: pd.Series, mode: Mode,
                 mol_list: Iterable[MultiMolecule],
                 prm_dict: Mapping[str, float]) -> Dict[Tuple[str, str], float]:
    """Perform the parameter guessing as specified by **mode**

    Returns
    -------
    :class:`dict`
        A dictionary with atom-pairs as keys (2-tuples) and the estimated parameters as values.

    """
    if mode == 'rdf':
        rdf(series, mol_list)
    elif mode == 'uff':
        uff(series, prm_dict)
    elif mode in ION_SET:
        ion_radius(series, prm_dict)
    elif mode in CRYSTAL_SET:
        crystal_radius(series, prm_dict)
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


def _nb_from_prm(prm: PRMContainer, param: Param) -> Dict[str, float]:
    r"""Extract a dict from **prm** with all :math:`\varepsilon` or :math:`\sigma` values."""
    if prm.nonbonded is None:
        return {}
    nonbonded = prm.nonbonded[[2, 3]].copy()
    nonbonded.columns = ['epsilon', 'sigma']  # kcal/mol and Angstrom
    nonbonded['sigma'] *= 2 / 2**(1/6)  # Conversion factor between (R / 2) and sigma
    return nonbonded[param].as_dict()
