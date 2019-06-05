"""A module with miscellaneous functions related to CP2K."""

from typing import (List, Tuple, Hashable)

import pandas as pd

from scm.plams import Settings

__all__ = ['set_keys']


def set_subsys_kind(settings: Settings,
                    df: pd.DataFrame) -> None:
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
    for at_name, at_type in df[['atom name', 'atom type']].values:
        if not settings.input.force_eval.subsys['kind ' + at_type]:
            settings.input.force_eval.subsys['kind ' + at_type] = {'element': at_name}


def set_keys(settings: Settings,
             param: pd.DataFrame) -> List[Tuple[Hashable]]:
    r"""Find and return the keys in **settings** matching all parameters in **param**.

    Units can be specified under the *unit* key (see the CP2K_ documentation for more details).
    A column of fstrings (``"unit"``) is added to the **param**,
    simplifying the process of prepending values with a unit without altering the key.


    .. _CP2K: https://manual.cp2k.org/trunk/units.html

    Examples
    --------
    .. code:: python

        >>> print(param)
                                      param
        charge                Br         -1
                              Cs          1
        lennard-jones epsilon unit    kjmol
                              Br Br  1.0501
                              Cs Br  0.4447
        lennard-jones sigma   unit       nm
                              Br Br    0.42
                              Cs Br    0.38

        >>> key_list = set_keys(settings, param)
        >>> print(param)
                                      param          unit
        charge                Br    -1.0000          {:f}
                              Cs     1.0000          {:f}
        lennard-jones epsilon Br Br  1.0501  [kjmol] {:f}
                              Cs Br  0.4447  [kjmol] {:f}
        lennard-jones sigma   Br Br  0.4200     [nm] {:f}
                              Cs Br  0.3800     [nm] {:f}

        >>> print(key_list)
        [('input', 'force_eval', 'mm', 'forcefield', 'charge', 0, 'charge'),
         ('input', 'force_eval', 'mm', 'forcefield', 'charge', 1, 'charge'),
         ('input', 'force_eval', 'mm', 'forcefield', 'nonbonded', 'lennard-jones', 0, 'epsilon'),
         ('input', 'force_eval', 'mm', 'forcefield', 'nonbonded', 'lennard-jones', 1, 'epsilon'),
         ('input', 'force_eval', 'mm', 'forcefield', 'nonbonded', 'lennard-jones', 0, 'sigma'),
         ('input', 'force_eval', 'mm', 'forcefield', 'nonbonded', 'lennard-jones', 1, 'sigma')]


    Parameters
    ----------
    param : |pd.DataFrame|_
        A dataframe with MM parameters and parameter names as 2-level multiindex.

    settings : |plams.Settings|_
        CP2K Job settings.

    Returns
    -------
    |list|_ [|tuple|_ [|str|_]]:
        A list of (flattened) CP2K input-settings keys.


    """
    # Create a new column in **param** with the quantity units
    param['unit'] = None
    for key in param.index.levels[0]:
        if 'unit' in param.loc[key].index:
            unit = param.at[(key, 'unit'), 'param']
            param.loc[[key], 'unit'] = '[{}]'.format(unit) + ' {:f}'
            param.drop(index=(key, 'unit'), inplace=True)
        else:
            param.loc[key, 'unit'] = '{:f}'
    param['param'] = param['param'].astype(float)

    return _get_key_list(settings, param)


def _get_key_list(settings: Settings,
                  param: pd.DataFrame) -> List[Tuple[Hashable]]:
    """Prepare the list of to-be returned keys for :func:`.set_keys`.

    The returned list consists of tuples with a path to specific nested values.

    Examples
    --------
    .. code:: python

        >>> print(param)
                                      param          unit
        charge                Br    -1.0000          {:f}
                              Cs     1.0000          {:f}
        lennard-jones epsilon Br Br  1.0501  [kjmol] {:f}
                              Cs Br  0.4447  [kjmol] {:f}
        lennard-jones sigma   Br Br  0.4200     [nm] {:f}
                              Cs Br  0.3800     [nm] {:f}

        >>> key_list = _get_key_list(settings, param)
        >>> print(key_list)
        [('input', 'force_eval', 'mm', 'forcefield', 'charge', 0, 'charge'),
         ('input', 'force_eval', 'mm', 'forcefield', 'charge', 1, 'charge'),
         ('input', 'force_eval', 'mm', 'forcefield', 'nonbonded', 'lennard-jones', 0, 'epsilon'),
         ('input', 'force_eval', 'mm', 'forcefield', 'nonbonded', 'lennard-jones', 1, 'epsilon'),
         ('input', 'force_eval', 'mm', 'forcefield', 'nonbonded', 'lennard-jones', 0, 'sigma'),
         ('input', 'force_eval', 'mm', 'forcefield', 'nonbonded', 'lennard-jones', 1, 'sigma')]

    Parameters
    ----------
    param : |pd.DataFrame|_
        A dataframe with MM parameters and parameter names as 2-level multiindex.

    settings : |plams.Settings|_
        CP2K Job settings.

    Returns
    -------
    |list|_ [|tuple|_ [|str|_]]:
        A list of (flattened) keys.

    """
    ret = []
    forcefield = ('input', 'force_eval', 'mm', 'forcefield')

    for (key, at), (value, fstring) in param[['param', 'unit']].iterrows():
        # Identify the keys in **settings**
        if 'charge' in key:
            super_key = key
            s = settings.input.force_eval.mm.forcefield
            atom = 'atom'
        else:
            super_key, key = key.split()
            s = settings.input.force_eval.mm.forcefield.nonbonded
            atom = 'atoms'

        # Add a new super_key (*e.g.* lennard-jones)
        if super_key not in s:
            s[super_key] = []
            i = 0

        # Update an existing value if possible
        new_block = True
        for i, dict_ in enumerate(s[super_key], 1):
            if at in dict_.values():
                i -= 1
                dict_[key] = fstring.format(value)
                new_block = False
                break

        # Create a new value otherwise
        if new_block:
            s[super_key].append(Settings({atom: at, key: fstring.format(value)}))

        # Update the list of to-be returned keys
        if key == 'charge':
            ret.append(forcefield + (super_key, i, key))
        else:
            ret.append(forcefield + ('nonbonded', super_key, i, key))

    return ret
