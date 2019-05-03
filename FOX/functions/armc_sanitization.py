""" A module for parsing and sanitizing :class:`FOX.classes.monte_carlo.ARMC` settings. """

__all__ = ['init_armc_sanitization']

from os.path import isfile, isdir, split

import numpy as np

from scm.plams import Settings
from scm.plams.core.basejob import Job
from scm.plams.interfaces.thirdparty.cp2k import Cp2kJob

from .utils import (get_template, _get_move_range, dict_to_pandas)
from .cp2k_utils import set_keys
from .charge_utils import get_charge_constraints
from ..classes.multi_mol import MultiMolecule


TYPE_ERR = "{} expects an object of type '{}', not '{}'"
TYPE_ERR2 = "{} expects an object of type '{}' or '{}', not '{}'"
TYPE_DICT = {
    'rdf': MultiMolecule.init_rdf,
    'adf': MultiMolecule.init_adf,
    'rmsd': MultiMolecule.init_rmsd,
    'rmsf': MultiMolecule.init_rmsf,
    'init_rdf': MultiMolecule.init_rdf,
    'init_adf': MultiMolecule.init_adf,
    'init_rmsd': MultiMolecule.init_rmsd,
    'init_rmsf': MultiMolecule.init_rmsf,
    'multimolecule.init_rdf': MultiMolecule.init_rdf,
    'multimolecule.init_adf': MultiMolecule.init_adf,
    'multimolecule.init_rmsd': MultiMolecule.init_rmsd,
    'multimolecule.init_rmsf': MultiMolecule.init_rmsf
}


def init_armc_sanitization(dict_):
    """ Initialize the armc input settings sanitization. """
    s = Settings(dict_)

    s.job = sanitize_job(s.job)
    s.pes = sanitize_pes(s.pes, s.job.molecule)
    s.hdf5_file = sanitize_hdf5_file(s.hdf5_file)
    s.param = sanitize_param(s.param, s.job.settings)
    s.move = sanitize_move(s.move)
    s.armc, s.phi = sanitize_armc(s.armc)

    mol = s.job.pop('molecule')
    param = s.pop('param')
    return mol, param, s


def get_name(item):
    """ Return the class name of **item**. """
    return item.__class__.__name__


def assert_type(item, item_type, name='argument'):
    if isinstance(item_type, tuple):
        type1 = item_type[0].__name__
    else:
        type1 = item_type.__name__

    if not isinstance(item, item_type):
        raise TypeError(TYPE_ERR.format(name, type1, get_name(item)))


def sanitize_armc(armc):
    """ Sanitize the armc block. """
    assert_type(armc.iter_len, (int, np.integer), 'armc.iter_len')
    assert_type(armc.sub_iter_len, (int, np.integer), 'armc.sub_iter_len')
    if armc.sub_iter_len > armc.iter_len:
        raise ValueError('armc.sub_iter_len is larger than armc.iter_len')

    if isinstance(armc.gamma, (int, np.integer)):
        armc.gamma = float(armc.gamma)
    assert_type(armc.gamma, (float, np.float), 'armc.gamma')

    if isinstance(armc.a_target, (int, np.integer)):
        armc.a_target = float(armc.a_target)
    assert_type(armc.a_target, (float, np.float), 'armc.a_target')

    if isinstance(armc.phi, (int, np.integer)):
        armc.phi = float(armc.phi)
    assert_type(armc.phi, (float, np.float), 'armc.phi')

    phi = Settings()
    phi.phi = armc.pop('phi')
    phi.kwarg = {}
    phi.func = np.add

    return armc, phi


def sanitize_param(param, settings):
    """ Sanitize the param block. """
    for key1, value1 in param.items():
        if not isinstance(key1, str):
            error = TYPE_ERR.format('param.{}'.format(str(key1)) + ' key', 'str', get_name(key1))
            raise TypeError(error)
        elif not isinstance(value1, dict):
            error = TYPE_ERR.format('param.{}'.format(str(key1)) + ' value',
                                    'dict', get_name(value1))
            raise TypeError(error)

        for key2, value2 in value1.items():
            if not isinstance(key2, str):
                error = TYPE_ERR.format('param.{}.{}'.format(str(key1), str(key2)) +
                                        ' key', 'str', get_name(key2))
                raise TypeError(error)
            elif isinstance(value2, (int, np.int)):
                param[key1][key2] = float(value2)
            elif not isinstance(value2, (float, np.float)):
                error = TYPE_ERR.format('param.{}.{}'.format(str(key1), str(key2)),
                                        'float', get_name(value2))
                raise TypeError(error)

    param = dict_to_pandas(param.as_dict(), 'param')
    param['key'] = set_keys(settings, param)
    param['param_old'] = np.nan
    return param


def sanitize_pes(pes, ref):
    """ Sanitize the pes block. """
    for key in pes:
        assert_type(key, str)
        if isinstance(pes[key].func, str):
            try:
                pes[key].func = TYPE_DICT[pes[key].func.lower()]
            except KeyError:
                raise KeyError("No type conversion available for '{}', consider directly passing"
                               " function as type object".format(pes[key].func.__name__))
        assert_type(pes[key].kwarg, dict, 'pes'+str(key)+'kwarg')
        pes[key].ref = pes[key].func(ref, **pes[key].kwarg)
    return pes


def sanitize_hdf5_file(hdf5_file):
    """ Sanitize the hdf5_file block. """
    if not isinstance(hdf5_file, str):
        raise TypeError(TYPE_ERR.format('hdf5_file', 'str', get_name(hdf5_file)))
    return hdf5_file


def sanitize_job(job):
    """ Sanitize the job block. """
    if isinstance(job.molecule, MultiMolecule):
        pass
    elif isinstance(job.molecule, str):
        job.molecule = MultiMolecule.from_xyz(job.molecule)
    else:
        raise TypeError(TYPE_ERR2.format('job.molecule', 'FOX.MultiMolecule',
                                         'str', get_name(job.molecule)))

    if job.type.lower() == 'cp2kjob':
        job.type = Cp2kJob
    elif not isinstance(job.type, Job):
        raise TypeError(TYPE_ERR.format('job.type', 'Job', get_name(job.type)))

    if isinstance(job.settings, dict):
        job.settings = Settings(job.Settings)
    elif isinstance(job.settings, str):
        if isfile(job.settings):
            head, tail = split(job.settings)
            job.settings = get_template(tail, path=head)
        else:
            job.settings = get_template(job.settings)
    else:
        raise TypeError(TYPE_ERR2.format('job.settings', 'dict', 'str', get_name(job.settings)))

    assert isdir(job.path)

    assert_type(job.name, str, 'job.name')
    assert_type(job.folder, str, 'job.workdir')
    assert_type(job.keep_files, bool, 'job.keep_files')
    return job


def sanitize_move(move):
    """ Sanitize the move block. """
    move.range = _get_move_range(**move.range)
    move.func = np.multiply
    move.kwarg = {}
    move.charge_constraints = get_charge_constraints(move.charge_constraints)
    return move
