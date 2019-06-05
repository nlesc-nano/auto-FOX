"""Various templates for validating ARMC inputs."""
from typing import Callable

import numpy as np
from collections import abc
from schema import (And, Optional, Or, Schema, Use)

from FOX.classes.multi_mol import MultiMolecule
from FOX.functions.charge_utils import get_charge_constraints

__all__ = []


def str_to_callable(string: str) -> Callable:
    """Create a callable object from a string.

    Accepts string-representations of functions, classes and methods, returning the respective
    callable.

    Examples
    --------
    An example with a builtin function:

    .. code:: python

        >>> callable_ = str_to_callable('len')
        >>> print(callable_)
        <function len(obj, /)>

        >>> out = callable_([True, True, True])
        >>> print(out)
        3

    An example with a third-party function from NumPy:

    .. code:: python

        >>> callable_ = str_to_callable('numpy.add')
        >>> print(callable_)
        <ufunc 'add'>

        >>> out = callable_(10, 5)
        >>> print(out)
        15

    Another example with a third-party method from the :class:`.MultiMolecule` class in Auto-FOX:

    .. code:: python

        >>> from FOX import get_example_xyz

        >>> callable_ = str_to_callable('FOX.MultiMolecule.from_xyz')
        >>> print(callable_)
        <bound method MultiMolecule.from_xyz of <class 'FOX.classes.multi_mol.MultiMolecule'>>

        >>> out = callable_(get_example_xyz())
        >>> print(type(out))
        <class 'FOX.classes.multi_mol.MultiMolecule'>


    Parameters
    ----------
    string : str
        A string represnting a callable object.
        The path to the callable should be included in the string (see examples).

    Returns
    -------
    |Callable|_:
        A callable object (*e.g.* function, class or method).

    """
    if '.' not in string:  # Builtin function or class
        return eval(string)

    elif string.count('.') == 1:
        try:  # Builtin method
            return eval(string)
        except NameError:  # Non-builtin function or class
            package, func = string.split('.')
            exec('from {} import {}'.format(package, func))
            return eval(func)

    else:
        try:  # Non-builtin function or class
            package, func = string.rsplit('.', 1)
            exec('from {} import {}'.format(package, func))
            return eval(func)
        except ImportError:  # Non-builtin method
            package, class_, method = string.rsplit('.', 2)
            exec('from {} import {}'.format(package, class_))
            return eval('.'.join([class_, method]))


_float = Or(int, np.integer, float, np.float)
_int = Or(int, np.integer)

schema_armc = Schema({
    ('a_target',): And(_float, Use(float), lambda x: 0 < x <= 1,
                       error='armc.a_target expects a float between 0.0 and 1.0'),

    ('gamma',): And(_float, Use(float), error='armc.gamma expects a float'),

    ('iter_len',): And(_int, lambda x: x > 1,
                       error='armc.iter_len expects an integer larger than 2'),

    ('phi',): And(_float, Use(float), error='armc.phi expects a float'),

    ('sub_iter_len',): And(_int, lambda x: x > 1,
                           error='armc.sub_iter_len expects an integer smaller than armc.iter_len')
})

schema_move = Schema({
    ('charge_constraint',): Or(None, And(str, get_charge_constraints),
                               error='move.charge_constrain expects a string or None'),

    ('range', 'start'): And(_float, Use(float), lambda x: x > 0.0,
                            error='move.range.start expects a float larger than 0.0'),

    ('range', 'step'): And(_float, Use(float), lambda x: x > 0.0,
                           error='move.range.step expects a float larger than 0.0'),

    ('range', 'stop'): And(_float, Use(float), lambda x: x > 0.0,
                           error='move.range.stop expects a float larger than 0.0'),
})

schema_job = Schema({
    ('folder',): And(str, error='job.folder expects a string'),

    ('func',): Or(abc.Callable, And(str, Use(str_to_callable)),
                  error='job.func expects a callable or string-representation of a callable'),

    ('keep_files',): And(bool, error='job.keep_files expects a boolean'),

    ('logfile',): And(str, error='job.logfile expects a string'),

    ('name',): And(str, error='job.name expects a string'),

    ('path',): And(str, error='job.path expects a string'),

    ('rmsd_threshold',): And(_float, Use(float), error='job.rmsd_threshold expects a float')
})


schema_param = Schema({
    tuple: Or(float, np.float, int, np.integer, str,
              error='param expects a (nested) dictionary of floats and/or integers')
})

schema_hdf5 = Schema(str, error='hdf5_file expects a string')

schema_molecule = Schema(Or(
    MultiMolecule, And(str, Use(MultiMolecule.from_xyz)),
    error='molecule expects a FOX.MultiMolecule instance or a string with an .xyz filename'
))

schema_psf = Schema({
    ('str_file',): And(str, error='psf.str_file expects a string'),

    ('ligand_atoms',): And(abc.Sequence, lambda x: all([isinstance(i, str) for i in x]),
                           error='psf.ligand_atoms expects a sequence of strings')
})


def get_pes_schema(key: str) -> Schema:
    err = 'pes.{}.func expects a callable or string-representation of a callable'
    schema_pes = Schema({
        'func': Or(abc.Callable, And(str, Use(str_to_callable)),
                   error=err.format(key)),

        Optional('kwarg', default={}): And(dict, lambda x: all([isinstance(i, str) for i in x]),
                                           error='pes.{}.kwarg expects a dictionary'.format(key)),

        Optional('arg', default=[]): And(abc.Sequence,
                                         error='pes.{}.arg expects a sequence'.format(key))
    })
    return schema_pes
