""" A module with miscellaneous functions. """

__all__ = ['serialize_array']

import numpy as np
import pandas as pd

from scm.plams.core.settings import Settings


def serialize_array(array, items_per_row=4, indent=9):
    """ """
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


def read_param(filename):
    """ Read a CHARMM parameter file. """
    with open(filename, 'r') as file:
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
