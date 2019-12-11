"""
FOX.armc_functions.df_to_dict
=============================

A module for converting ARMC parameter dataframes to ARMC-compatible dictionaries.

"""

from collections import abc
from typing import Type, Mapping

import numpy as np
import pandas as pd


def df_to_dict(df: pd.DataFrame, mapping_type: Type[Mapping] = dict) -> Mapping:
    """Convert an :class:`ARMC` parameter DataFrame to :meth:`ARMC.from_yaml`-compatible mapping.

    Example
    -------
    .. code:: python

        >>> import pandas as pd
        >>> from scm.plams import Settings

        >>> df = pd.DataFrame(...)
        >>> print(df)
                        param  param_old  ... constraints  count
        charge  Cd     0.9768        NaN  ...        None     68
                Se    -0.9768        NaN  ...        None     55
        epsilon Cd Cd  0.3101        NaN  ...        None   2278
        sigma   Cd Cd  0.1234        NaN  ...        None   2278
                Cd Se  0.2940        NaN  ...        None   3740
                Se Se  0.4852        NaN  ...        None   1485

        [6 rows x 8 columns]

        >>> s = df_to_dict(df, mapping_type=Settings)
        >>> print(s)
        charge:
               constraints:     []
               keys:    ['input', 'force_eval', 'mm', 'forcefield', 'charge']
               Cd:      0.9768
               Se:      -0.9768
        epsilon:
                constraints:    []
                keys:   ['input', 'force_eval', 'mm', 'forcefield', 'nonbonded', 'lennard-jones']
                unit:   kjmol
                Cd Cd:  0.3101
        sigma:
              constraints:      []
              keys:     ['input', 'force_eval', 'mm', 'forcefield', 'nonbonded', 'lennard-jones']
              unit:     nm
              Cd Cd:    0.1234
              Cd Se:    0.294
              Se Se:    0.4852

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        A DataFrame with variable ARMC parameters; see :attr:`ARMC.param`.

    mapping_type : :class:`type` [:class:`Mapping<collections.abc.Mapping>`]
        The to-be returned mapping type.

    Returns
    -------
    :class:`Mapping<collections.abc.Mapping>`
        An :meth:`ARMC.from_yaml`-compatible mapping constructed from **df**.

    """
    if isinstance(mapping_type, abc.MutableMapping):
        ret = mapping_type()
        mutable = True
    else:
        ret = {}
        mutable = False

    prm_set = set()
    for (i, j), series in df.iterrows():
        try:
            ret_i = ret[i]
        except KeyError:
            ret_i = ret[i] = {}

        if i not in prm_set:
            prm_set.add(i)
            ret_i['constraints'] = []
            ret_i['keys'] = series['keys'][:-2]
            if series['unit'] != '{:f}':
                ret_i['unit'] = series['unit'].split()[0][1:-1]
            if series['constraints'] is not None:
                ret_i['constraints'].append(
                    ' == '.join(f'{at} * {k}' for at, k in series['constraints'].items())
                )

        ret_i[j] = series['param']
        if series['max'] != np.inf:
            ret_i['constraints'].append(f'{j} < {series["max"]}')
        if series['min'] != -np.inf:
            ret_i['constraints'].append(f'{j} < {series["min"]}')

    return ret if mutable else mapping_type(ret)
