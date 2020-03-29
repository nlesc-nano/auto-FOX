"""A module with miscellaneous functions related to CP2K."""

from types import MappingProxyType
from typing import Mapping, Union, Optional, TypeVar, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import constants

from scm.plams import Settings, Units

from .charge_utils import assign_constraints

if TYPE_CHECKING:
    from scm.plams import Results
    from qmflows.packages import Result
else:
    from ..type_lias import Results, Result

__all__ = ['set_keys', 'parse_cp2k_value']

# Multiplicative factor for converting Hartree into Kelvin
Units.energy['k'] = Units.energy['kelvin'] = (
    constants.physical_constants['Hartree energy'][0] / constants.Boltzmann
)

#: Map CP2K units to PLAMS units.
UNIT_MAP: Mapping[str, str] = MappingProxyType({
    '[hartree]': 'hartree',
    '[ev]': 'eV',
    '[kcalmol]': 'kcal/mol',
    '[kjmol]': 'kj/mol',
    '[k_e]': 'kelvin',

    '[bohr]': 'bohr',
    '[pm]': 'pm',
    '[nm]': 'nm',
    '[angstrom]': 'angstrom',

    '[rad]': 'radian',
    '[deg]': 'degree'
})

T = TypeVar('T', float, np.ndarray)


def parse_cp2k_value(param: Union[str, T], unit: str, default_unit: Optional[str] = None) -> T:
    """Parse and return the provided CP2K input parameter **param**.

    Examples
    --------
    Examples of valid input values for **param**.

    .. code:: python

        >>> param1: str = '[angstrom] 2.0'
        >>> param2: str = '2.5'
        >>> param3: float = 9.3
        >>> param4: int = 2
        ...


    Parameters
    ----------
    param : :class:`str`, :class:`float` or :class:`numpy.ndarray`
        The parameter.

    unit : :class:`str`
        The desired output unit of **param**.

    default_unit :class:`str`, optional
        The default input unit of **param** for when its unit is not explicitly specified.
        Will be ignored if ``None``.

    Returns
    -------
    :class:`float` or :class:`numpy.ndarray`
        The value **param** expressed in **unit**.

    Raises
    ------
    :exc:`ValueError`
        Raised if

    """
    default_unit = unit if default_unit is None else default_unit

    # Identify the unit and the quantity of interest
    if not isinstance(param, str):
        value_unit = None
    else:
        value_unit, param = param.split()
        param = float(param)

    if not value_unit:
        return param * Units.conversion_ratio(default_unit, unit)

    # Correct the units
    try:  # Convert from CP2K to PLAMS Units
        value_unit_parsed = UNIT_MAP[value_unit.lower()]
    except KeyError as ex:
        raise ValueError(f"Invalid unit {value_unit.lower()!r};\naccepted units: "
                         f"{tuple(UNIT_MAP.keys())!r}") from ex
    return param * Units.conversion_ratio(value_unit_parsed, unit)


def get_xyz_path(results: Union[Results, Result]) -> str:
    """Return the path + filename to an .xyz file."""
    for file in results.files:
        if '-pos' in file and '.xyz' in file:
            return results.files[file]
    raise FileNotFoundError(f'No .xyz files found in {results.job.path!r}')


def set_subsys_kind(settings: Settings, df: pd.DataFrame) -> None:
    """Set the FORCE_EVAL/SUBSYS/KIND_ keyword(s) in CP2K job settings.

    Performs an inplace update of the input.force_eval.subsys key in **settings**.

    .. _KIND: https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/SUBSYS/KIND.html

    Examples
    --------
    .. code:: python

        >>> print(df)
           atom name  atom type
        1          O      OG2D2
        2          O      OG2D2
        3          C      CG2O3
        4          C      CG2O3
        5          H      HGR52
        6          H      HGR52

        >>> set_subsys_kind(settings, df)
        >>> print(settings.input.force_eval)
        input:
              force_eval:
                         subsys:
                                kind OG2D2:
                                           element:        O
                                kind CG2O3:
                                           element:        C
                                kind HGR52:
                                           element:        H

    Parameters
    ----------
    settings : |plams.Settings|_
        CP2K settings.

    df : |pd.DataFrame|_
        A dataframe with atom names (*e.g.* ``"O"``, ``"H"`` & ``"C"``)
        and atom types (*e.g.* ``"OG2D2"``, ``"HGR52"`` & ``"CG2O3"``).

    """
    subsys = settings.input.force_eval.subsys
    for at_name, at_type in df[['atom name', 'atom type']].values:
        at_type = f'kind {at_type}'
        if not subsys[at_type]:
            subsys[at_type] = {'element': at_name}


def set_keys(settings: Settings, param: pd.DataFrame) -> None:
    r"""Parse **param** and populate all appropiate blocks in **settings**.

    Updates/creates three columns in **param**:

        * ``"param"``: Contains the initial parameter value.
        * ``"unit"``: Contains an fstring for ``"param"``, including the appropiate unit.
        * ``"keys"``: A list keys pointing to the parameter block of interest.

    See the CP2K_ documentation for more details regarding available units.

    .. _CP2K: https://manual.cp2k.org/trunk/units.html

    Examples
    --------
    .. code:: python

        >>> print(param)
                                                                   param
        charge  Br                                                  -1.0
                Cs                                                   1.0
                keys         [input, force_eval, mm, forcefield, charge]
        epsilon Cs Br                                             0.4447
                keys   [input, force_eval, mm, forcefield, nonbonded,...
                unit                                               kjmol
        sigma   Cs Br                                               0.38
                keys   [input, force_eval, mm, forcefield, nonbonded,...
                unit                                                  nm

        >>> set_keys(settings, param)
        >>> print(param)
                        param          unit                                               keys
        charge  Br    -1.0000          {:f}  [input, force_eval, mm, forcefield, charge, 0,...
                Cs     1.0000          {:f}  [input, force_eval, mm, forcefield, charge, 1,...
        epsilon Cs Br  0.4447  [kjmol] {:f}  [input, force_eval, mm, forcefield, nonbonded,...
        sigma   Cs Br  0.3800     [nm] {:f}  [input, force_eval, mm, forcefield, nonbonded,...


    Parameters
    ----------
    param : |pd.DataFrame|_
        A dataframe with MM parameters and parameter names as 2-level multiindex.

    settings : |plams.Settings|_
        CP2K Job settings.

    """
    key_list = []
    param['unit'] = None
    param['max'] = np.inf
    param['min'] = -np.inf

    # Create and fill a column with units (fstrings) and key paths
    for k in param.index.levels[0]:
        # Idenify if units are specified
        if 'unit' in param.loc[k].index:
            unit = param.at[(k, 'unit'), 'param']
            param.loc[[k], 'unit'] = '[{}]'.format(unit) + ' {:f}'
            param.drop(index=(k, 'unit'), inplace=True)
        else:
            param.loc[k, 'unit'] = '{:f}'

        # Identify and parse constraints
        if 'constraints' in param.loc[k].index:
            constraints = param.at[(k, 'constraints'), 'param']
            param.drop(index=(k, 'constraints'), inplace=True)
            assign_constraints(constraints, param, k)

        # Identify and parse the path
        keys = param.at[(k, 'keys'), 'param']
        key_list += [keys.copy() for _ in param.loc[k, 'param']][1:]
        param.drop(index=(k, 'keys'), inplace=True)

    param['keys'] = key_list
    param['param'] = param['param'].astype(float)
    param.index = _sort_index(param.index)
    _populate_keys(settings, param)


def _sort_index(index: pd.MultiIndex) -> pd.MultiIndex:
    ret = []
    for i, j in index:
        j_split = j.split()
        if len(j_split) == 2:
            ret.append((i, ' '.join(k for k in sorted(j_split))))
        else:
            ret.append((i, j))

    return pd.MultiIndex.from_tuples(ret, names=index.names)


def _populate_keys(settings: Settings, param: pd.DataFrame, update_keys: bool = True) -> None:
    """Populate the settings blocks specified in :func:`.set_keys`.

    Examples
    --------
    .. code:: python

        >>> print(param)
            param  unit                                         keys
        Br   -1.0  {:f}  [input, force_eval, mm, forcefield, charge]
        Cs    1.0  {:f}  [input, force_eval, mm, forcefield, charge]
        Pb    2.0  {:f}  [input, force_eval, mm, forcefield, charge]

        >>> _populate_keys(settings, param)
        >>> print(settings.input.force_eval.mm.forcefield.charge)
        [atom:  Br
        charge:         -1.000000
        , atom:         Cs
        charge:         1.000000
        , atom:         Pb
        charge:         2.000000
        ]

    Parameters
    ----------
    param : |pd.DataFrame|_
        A dataframe with MM parameters and parameter names as 2-level multiindex.

    settings : |plams.Settings|_
        CP2K Job settings.

    """
    for i in param.index.levels[0]:
        try:
            keys = param.loc[i, 'keys'].iloc[0]
        except KeyError:
            continue
        if not isinstance(settings.get_nested(keys), list):
            settings.set_nested(keys, [])

    for (k, at), (keys, prm, fstring) in param[['keys', 'param', 'unit']].iterrows():
        # User either 'atoms' or `atom' as key depending on the number of atoms
        if len(at.split()) != 1:
            atom = 'atoms'
        else:
            atom = 'atom'

        # Evaluate if **param** consists of intersecting or disjoint sets of input blocks
        nested_value: list = settings.get_nested(keys)
        idx = None
        for i, j in enumerate(nested_value):
            if atom in j and at == j[atom]:
                idx = i
                break

        # Populate the blocks specified in the ``"keys"`` column
        if idx is None:  # Disjoint set of input blocks
            idx = len(nested_value)
            dict_ = Settings({atom: at, k: fstring.format(prm)})
            nested_value.append(dict_)
            if update_keys:
                keys += [idx, k]
        else:  # Intersecting set of input blocks
            if update_keys:
                keys += [idx, k]
                settings.set_nested(keys, fstring.format(prm))
            else:
                settings.set_nested(keys + [idx, k], fstring.format(prm))
