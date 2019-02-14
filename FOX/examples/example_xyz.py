""" A module for grabbing the example multi-xyz file. """

__all__ = []

from os.path import (join, dirname)


def get_example_xyz(path=None, name='Cd68Se55_26COO_MD_trajec.xyz'):
    """ Return the path + name of the example multi-xyz file. """
    path = path or join(dirname(dirname(__file__)), 'data')
    return join(path, name)
