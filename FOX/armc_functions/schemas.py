"""
FOX.armc_functions.sanitization
===============================

Various templates for validating ARMC inputs.

Index
-----
.. currentmodule:: FOX.armc_functions.schemas
.. autosummary::
    get_pes_schema
    schema_armc
    schema_move
    schema_job
    schema_param
    schema_hdf5
    schema_molecule
    schema_psf

API
---
.. autofunction:: FOX.armc_functions.schemas.get_pes_schema

.. autodata:: FOX.armc_functions.schemas.schema_armc
    :annotation: = schema.Schemas

.. autodata:: FOX.armc_functions.schemas.schema_move
    :annotation: = schema.Schemas

.. autodata:: FOX.armc_functions.schemas.schema_job
    :annotation: = schema.Schemas

.. autodata:: FOX.armc_functions.schemas.schema_param
    :annotation: = schema.Schemas

.. autodata:: FOX.armc_functions.schemas.schema_hdf5
    :annotation: = schema.Schemas

.. autodata:: FOX.armc_functions.schemas.schema_molecule
    :annotation: = schema.Schemas

.. autodata:: FOX.armc_functions.schemas.schema_psf
    :annotation: = schema.Schemas

"""

from schema import And, Optional, Or, Schema, Use
from collections import abc

import numpy as np

from ..functions.utils import str_to_callable

__all__ = [
    'get_pes_schema', 'schema_armc', 'schema_move', 'schema_job', 'schema_param',
    'schema_hdf5', 'schema_psf'
]


def get_pes_schema(key: str) -> Schema:
    """Return a :class:`schema.Schema` instance for validating a PES block (see **key**)."""
    err = 'pes.{}.func expects a callable or string-representation of a callable'
    schema_pes = Schema({
        'func': Or(abc.Callable, And(str, Use(str_to_callable)),
                   error=err.format(key)),

        Optional('kwargs', default={}): And(dict, lambda x: all([isinstance(i, str) for i in x]),
                                            error='pes.{}.kwargs expects a dictionary'.format(key)),

        Optional('args', default=[]): And(abc.Sequence,
                                          error='pes.{}.args expects a sequence'.format(key))
    })
    return schema_pes


_float = Or(int, np.integer, float, np.float)
_int = Or(int, np.integer)

#: Schema for validating the ``"armc"`` block.
schema_armc: Schema = Schema({
    ('a_target',): And(_float, Use(float), lambda x: 0 < x <= 1,
                       error='armc.a_target expects a float between 0.0 and 1.0'),

    ('gamma',): And(_float, Use(float), error='armc.gamma expects a float'),

    ('iter_len',): And(_int, lambda x: x > 1,
                       error='armc.iter_len expects an integer larger than 2'),

    ('phi',): And(_float, Use(float), error='armc.phi expects a float'),

    ('sub_iter_len',): And(_int, lambda x: x > 1,
                           error='armc.sub_iter_len expects an integer smaller than armc.iter_len')
})

#: Schema for validating the ``"move"`` block.
schema_move: Schema = Schema({
    ('func',): Or(abc.Callable, And(str, Use(str_to_callable)),
                  error='move.func expects a callable or string-representation of a callable'),

    ('args',): And(abc.Sequence, error='move.arg expects a sequence'),

    ('kwargs',): And(dict, lambda x: all([isinstance(i, str) for i in x]),
                     error='move.kwarg expects a dictionary'),

    ('range', 'start'): And(_float, Use(float), lambda x: x > 0.0,
                            error='move.range.start expects a float larger than 0.0'),

    ('range', 'step'): And(_float, Use(float), lambda x: x > 0.0,
                           error='move.range.step expects a float larger than 0.0'),

    ('range', 'stop'): And(_float, Use(float), lambda x: x > 0.0,
                           error='move.range.stop expects a float larger than 0.0'),
})

#: Schema for validating the ``"job"`` block.
schema_job: Schema = Schema({
    ('folder',): And(str, error='job.folder expects a string'),

    ('job_type',): Or(abc.Callable, And(str, Use(str_to_callable)),
                      error='job.func expects a callable or string-representation of a callable'),

    ('keep_files',): And(bool, error='job.keep_files expects a boolean'),

    ('logfile',): And(str, error='job.logfile expects a string'),

    ('name',): And(str, error='job.name expects a string'),

    ('path',): And(str, error='job.path expects a string'),

    ('rmsd_threshold',): And(_float, Use(float), error='job.rmsd_threshold expects a float')
})

#: Schema for validating the ``"param"`` block.
schema_param: Schema = Schema({
    tuple: Or(float, np.float, int, np.integer, str, None, list,
              error='param expects a (nested) dictionary of floats and/or integers')
})

#: Schema for validating the ``"hdf5"`` block.
schema_hdf5: Schema = Schema(str, error='hdf5_file expects a string')

#: Schema for validating the ``"psf"`` block.
schema_psf: Schema = Schema({
    ('str_file',): Or(None, And(str), error='psf.str_file expects a string'),

    ('rtf_file',): Or(None, And(str), error='psf.rtf_file expects a string'),

    ('ligand_atoms',): Or(None, And(abc.Sequence, lambda x: all([isinstance(i, str) for i in x])),
                          error='psf.ligand_atoms expects a sequence of strings')
})
