""" A module with miscellaneous functions. """

__all__ = ['get_template']

from os.path import join
import pkg_resources as pkg

import yaml
import pandas as pd

from scm.plams import Settings


def get_template(name, path=None):
    """ Grab a .yaml template and turn it into a Settings object.

    :parameeter str name: The name of the template file.
    :parameter str path: The path where **name** is located.
        Will default to the FOX.data directory if *None*.
    :return: A settings object as constructed from the template file.
    :rtype: |plams.Settings|_
    """
    if path is None:
        path = pkg.resource_filename('FOX', join('data', name))
    else:
        path = join(path, name)

    with open(path, 'r') as f:
        return Settings(yaml.load(f, Loader=yaml.FullLoader))


def serialize_array(array, items_per_row=4, indent=9):
    """ Serialize an array into a single string.

    :parameter array: An array.
    :type array: |np.ndarray|_
    :parameter int items_per_row: The number of values per row before switching to a new line.
    :parameter int indent: The number of spaces used for indentation at
        the begining of each new line.
    :return: A serialized array.
    :rtype: |str|_
    """
    if len(array) == 0:
        return ''

    ret = ''
    for _ in range(indent):
        ret += ' '

    k = 0
    for i in array:
        for j in i:
            ret += '{:8.8}'.format(str(j)) + ' '
        k += 1
        if (i != array[-1]).all() and k == items_per_row:
            k = 0
            ret += '\n'
            for _ in range(indent):
                ret += ' '

    return ret


def read_param(name):
    """ Read a CHARMM parameter file.

    :parameter str name: the path+filename of the CHARMM parameter file.
    :return: A settings object consisting of 5 dataframes assigned to the following keys:
        *bonds*, *angles*, *dihedrals*, *improper* & *nonbonded*.
    :rtype: |plams.Settings|_ (keys: |str|_, values: |pd.DataFrame|_)
     """
    with open(name, 'r') as file:
        str_list = file.read().splitlines()
    str_gen = (i for i in str_list if '*' not in i and '!' not in i)

    headers = ['BONDS', 'ANGLES', 'DIHEDRALS', 'IMPROPER', 'NONBONDED']
    df_dict = {}
    for i in str_gen:
        if i in headers:
            tmp = []
            for j in str_gen:
                if j:
                    tmp.append(j.split())
                else:
                    df_dict[i.lower()] = pd.DataFrame(tmp)
                    break

    return Settings(df_dict)
