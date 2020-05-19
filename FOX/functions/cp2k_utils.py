"""A module with miscellaneous functions related to CP2K.

Index
-----
.. currentmodule:: FOX.functions.cp2k_utils
.. autosummary::
    parse_cp2k_value
    get_xyz_path
    UNIT_MAP

API
---
.. autofunction:: parse_cp2k_value
.. autofunction:: update_charge
.. autodata:: UNIT_MAP

"""

import os
from types import MappingProxyType
from typing import Mapping, Union, Optional, TypeVar

import numpy as np
from scipy import constants

from scm.plams import Units

__all__ = ['UNIT_MAP', 'parse_cp2k_value', 'get_xyz_path']

# Multiplicative factor for converting Hartree into Kelvin
Units.energy['k'] = Units.energy['kelvin'] = (
    constants.physical_constants['Hartree energy'][0] / constants.Boltzmann
)

#: Map CP2K units to PLAMS units.
UNIT_MAP: Mapping[str, str] = MappingProxyType({
    'hartree': 'hartree',
    'ev': 'eV',
    'kcalmol': 'kcal/mol',
    'kjmol': 'kj/mol',
    'k_e': 'kelvin',

    'bohr': 'bohr',
    'pm': 'pm',
    'nm': 'nm',
    'angstrom': 'angstrom',

    'rad': 'radian',
    'deg': 'degree'
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
        value_unit, _param = param.split()
        param = float(_param)

    if not value_unit:
        return param * Units.conversion_ratio(default_unit, unit)

    # Correct the units
    value_unit_lower = value_unit.lower().strip('[').rstrip(']')
    try:  # Convert from CP2K to PLAMS Units
        value_unit_parsed = UNIT_MAP[value_unit_lower]
    except KeyError as ex:
        raise ValueError(f"Invalid unit {value_unit_lower!r};\naccepted units: "
                         f"{tuple(UNIT_MAP.keys())!r}") from ex
    return param * Units.conversion_ratio(value_unit_parsed, unit)


def get_xyz_path(path: Union[str, os.PathLike]) -> str:
    """Return the path + filename to an .xyz file."""
    for file in os.listdir(path):
        if '-pos' in file and '.xyz' in file:
            return os.path.join(path, file)
    raise FileNotFoundError(f'No .xyz files found in {path!r}')
