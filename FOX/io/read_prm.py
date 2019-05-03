""" A module for reading CHARMM .prm files. """

__all__ = ['read_prm', 'write_prm', 'rename_atom_types']

import pandas as pd
import numpy as np


def rename_atom_types(prm_dict, rename_dict):
    """ Rename atom types in a CHARM parameter file (prm_).

    An example is provided below, one where the atom type *H_old* is renamed to *H_new*:

    .. code:: python

        >>> 'H_old' in atom_dict['ATOMS'].index
        True

        >>> rename_dict = {'H_old': 'H_new'}
        >>> rename_atom_types(prm_dict, rename_dict)

        >>> 'H_old' in atom_dict['ATOMS'].index
        False

        >>> 'H_new' in atom_dict['ATOMS'].index
        True

    :parameter prm_dict: A dictionary with **key** as key and a dataframe of accumulated data as
        value.
    :parameter rename_dict: A dictionary or series with old atom types as keys and new atom types
        as values
    :type atom_dict: |dict|_ or |pd.Series|_ (keys: |str|_, values: |str|_)

    .. _prm: https://mackerell.umaryland.edu/charmm_ff.shtml
    """
    for key, df in prm_dict.items():
        idx = np.array(df.index.tolist())
        for at_old, at_new in rename_dict.items():
            try:
                idx[idx == at_old] = np.array(at_new, dtype=idx.dtype)
            except ValueError:
                pass

        if idx.ndim == 1:
            df.index = pd.Index(idx, names=df.index.name)
        else:
            df.index = pd.MultiIndex.from_arrays(idx.T, names=df.index.names)


def write_prm(prm_dict, filename='charmm_out.prm'):
    """ Create a new CHARM parameter file (prm_) out of **prm_dict**.

    :parameter prm_dict: A dictionary with block names as keys and a dataframe of matching
        parameters as value. Atom types should be used as (multi-)index.
    :type prm_dict: |dict|_ (keys: |str|_, values: |pd.DataFrame|_)
    :parameter str filename: The path+filename of the to-be created .prm file.

    .. _prm: https://mackerell.umaryland.edu/charmm_ff.shtml
    """
    with open(filename, 'w') as f:
        for key, df in prm_dict.items():
            if 'HBOND' in key:
                f.write(key + '\n\n')
                continue

            df = df.reset_index()
            f.write(key + '\n')

            line = _get_empty_line(df)
            for _, row in df.iterrows():
                line_format = line.format(*row)
                if 'nan' in line_format:
                    line_format = line_format.replace('nan', '   ')
                f.write(line_format)

            f.write('\n')
        f.write('END\n')


def _get_empty_line(df):
    """ Create a string with a sufficient amount of curly brackets to hold all items from a single
    row in **df**.

    :parameter df: A Pandas dataframe.
    :type df: |pd.DataFrame|_
    :return: Given a dataframe, **df**, with :math:`n` columns, return a string with :math:`n`
        sets of curly brackets.
    :rtype: |str|_
    """
    ret = ''
    for column in df:
        if df[column].dtype == np.dtype('O'):
            ret += ' {:10.10}'
        elif df[column].dtype == np.dtype('float64'):
            ret += ' {:>10.5f}'
        else:
            ret += ' {:>10.0f}'
    return ret[1:] + '\n'


def read_prm(filename='charmm.prm'):
    """ Read a CHARM parameter file (prm_), returning a dictionary of dataframes.

    The .prm file is expected to possess one or more of the following blocks:

        * ATOMS
        * BONDS
        * ANGLES
        * DIHEDRALS
        * IMPROPERS
        * NONBONDED
        * NBFIX
        * HBOND

    Blocks not mentioned herein are unsupported.

    :parameter str filename: The path+filename of the parameter file.
    :return: A dictionary with block names as keys and a dataframe of matching parameters as value.
        Atom types are used as (multi-)index.
    :rtype: |dict|_ (keys: |str|_, values: |pd.DataFrame|_)

    .. _prm: https://mackerell.umaryland.edu/charmm_ff.shtml
    """
    ret = {}
    with open(filename, 'r') as f:
        for i in f:
            key = i.rstrip('\n')
            if key == 'ATOMS':
                while(key):
                    dict_, key = read_blocks(f, key)
                    ret.update(dict_)
                    next
    return _proccess_prm_df(ret)


def read_blocks(f, key):
    """ Read the content of a .prm block.

    The following, and only the following, blocks are currently supported:

        * ATOMS
        * BONDS
        * ANGLES
        * DIHEDRALS
        * NBFIX
        * IMPROPERS
        * NONBONDED
        * HBOND

    :parameter f: An opened .prm file.
    :type f: |io.TextIOWrapper|_
    :parameter str key: The key of the current .prm block.
    :return: A dictionary with **key** as key and a dataframe of accumulated data as value.
        In addition, the key of the next .prm block is returned.
    :rtype: |dict|_ (key: |str|_, value: |pd.DataFrame|_) and |str|_
    """
    stop = ('ATOMS', 'BONDS', 'ANGLES', 'DIHEDRALS', 'NBFIX', 'IMPROPERS')
    ret = []
    for j in f:
        item = j.rstrip('\n')
        if 'NONBONDED' in item:  # Prepare for the NONBONDED block
            item = _get_nonbonded(f, item)
            kwarg = {'data': ret, 'index': np.arange(len(ret))}
            return {key: pd.DataFrame(**kwarg)}, item

        elif 'HBOND' in item:  # Prepare for the HBOND block
            kwarg = {'data': ret, 'index': np.arange(len(ret))}
            return {key: pd.DataFrame(**kwarg)}, item.split('!')[0].rstrip()

        elif item in stop:  # Prepare for the next (regular) block
            kwarg = {'data': ret, 'index': np.arange(len(ret))}
            return {key: pd.DataFrame(**kwarg)}, item

        elif item == 'END':  # The end of the .prm file has been reached
            kwarg = {'data': ret, 'index': np.arange(len(ret))}
            return {key: pd.DataFrame(**kwarg)}, False

        elif item and item[0] != '!':  # Iterate within a block
            item2 = item.split('!')[0].split()
            ret.append(item2)


def _get_nonbonded(f, item):
    """ Get the key of the NONBONDED block.

    :parameter f: An opened .prm file.
    :type f: |io.TextIOWrapper|_
    :parameter str item: The part of the .prm NONBONDED key.
    :return: The complete .prm NONBONDED key.
    :rtype: |str|_
    """
    item2 = '\n'
    while(item2):
        item += item2
        k = next(f)
        item2 = k.rstrip('\n')
    return item + '\n'


def _proccess_prm_df(prm_dict):
    """ Process the dataframes produced by :func:`.read_prm`, re-assigning columns,
    fixing data types and using atom types as (multi-)index.

    :parameter prm_dict: A dictionary with **key** as key and a dataframe of accumulated data as
        value.
    :return: **prm_dict** with new columns, indices and datatypes.
    :rtype: |dict|_ (key: |str|_, value: |pd.DataFrame|_)
    """
    float_blacklist = {'ATOMS': ('-1'), 'DIHEDRALS': ('param 2'), 'IMPROPERS': ('param 2')}

    # A dictionary with column keys
    column_dict = {
        'ATOMS': ['MASS', '-1', 1, 'mass'],
        'BONDS': [1, 2],
        'ANGLES': [1, 2, 3],
        'DIHEDRALS': [1, 2, 3, 4],
        'NBFIX': [1, 2],
        'IMPROPERS': [1, 2, 3, 4],
        'NONBONDED': [1],
    }

    for key, df in prm_dict.items():
        if 'NONBONDED' in key:
            key = 'NONBONDED'
        elif key == 'ATOMS':
            column_dict[key] = _reorder_column_dict(df)
        elif 'HBOND' in key:  # Ignore, this is an empty dataframe
            continue

        # Prepare new columns
        columns = column_dict[key]
        i = df.shape[1] - len(columns)
        columns += ['param {:d}'.format(j) for j in range(1, i+1)]
        df.columns = pd.Index(columns, name='parameters')

        # Prepare a new index
        df.set_index([i for i in df.columns if isinstance(i, int)], inplace=True)
        df.index.set_names(['Atom {:d}'.format(i) for i in df.index.names], inplace=True)
        df.sort_index(inplace=True)
        update_dtype(df, float_blacklist.setdefault(key, []))
    return prm_dict


def _reorder_column_dict(df):
    ret = []
    for i, (key, column) in enumerate(df.items()):
        if (column == '-1').all():
            ret.append('-1')
            continue

        elif (column == 'MASS').all():
            ret.append('MASS')
            continue

        try:
            float(column.iloc[0])
            ret.append('mass')
        except ValueError:
            ret.append(1)
    return ret


def update_dtype(df, float_blacklist=[]):
    """ Update the dtype of all columns in **df**.

    All columns will be turned into dtype('float64') unless a value error is raised, in which case
    the current dtype will remain unchanged.
    An exception is made for columns whose name is present in **float_blacklist**, in wich case the
    dtype of the respective column(s) is converted into dtype('int64').
    Performs an inplace update of all columns in **df**.

    :parameter df: A Pandas dataframe.
    :type df: |pd.DataFrame|_
    :parameter list float_blacklist: A list of column names of columns whose desired dtype is
        dtype('int64') rather than dtype('float64').
    """
    for column in df:
        if column in float_blacklist:
            df[column] = df[column].astype(int, copy=False)
            continue
        try:
            df[column] = df[column].astype(float, copy=False)
        except ValueError:
            pass
