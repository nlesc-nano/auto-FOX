"""Various templates for validating ARMC inputs."""

import numpy as np
from collections import abc
from schema import (And, Optional, Or, Schema, Use)

from FOX.classes.multi_mol import MultiMolecule
from FOX.functions.utils import str_to_callable
from FOX.functions.charge_utils import get_charge_constraints

__all__ = [
    'get_pes_schema', 'schema_armc', 'schema_move', 'schema_job', 'schema_param',
    'schema_hdf5', 'schema_molecule', 'schema_psf'
]


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
