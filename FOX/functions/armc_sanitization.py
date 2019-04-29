""" A module for parsing and sanitizing :class:`FOX.ARMC` settings. """

__all__ = []

import os

import schema
import numpy as np

from scm.plams import Molecule, Settings
from scm.plams.core.basejob import Job
from scm.plams.interfaces.thirdparty import Cp2kJob

from FOX import MultiMolecule, get_template


TYPE_ERR = "{} expects an object of type '{}', not '{}'"
TYPE_ERR2 = "{} expects an object of type '{}' or '{}', not '{}'"


def get_name(item):
    return item.__class__.__name__


def sanitize_armc(armc):
    if not isinstance(armc.iter_len, (int, np.integer)):
        raise TypeError(TYPE_ERR.format('armc.iter_len', 'int', get_name(armc.iter_len)))

    if not isinstance(armc.sub_iter_len, (int, np.integer)):
        raise TypeError(TYPE_ERR.format('armc.iter_len', 'int', get_name(armc.sub_iter_len)))
    elif armc.sub_iter_len > armc.iter_len:
        raise ValueError('armc.sub_iter_len is larger than armc.iter_len')

    if isinstance(armc.gamma, (int, np.integer)):
        armc.gamma = float(armc.gamma)
    elif not isinstance(armc.gamma, (float, np.float)):
        raise TypeError(TYPE_ERR.format('armc.gamma', 'float', get_name(armc.gamma)))

    if isinstance(armc.a_target, (int, np.integer)):
        armc.a_target = float(armc.a_target)
    elif not isinstance(armc.a_target, (float, np.float)):
        raise TypeError(TYPE_ERR.format('armc.a_target', 'float', get_name(armc.a_target)))

    if isinstance(armc.phi, (int, np.integer)):
        armc.phi = float(armc.phi)
    elif not isinstance(armc.phi, (float, np.float)):
        raise TypeError(TYPE_ERR.format('armc.phi', 'float', get_name(armc.phi)))

    return armc


def sanitize_ref(ref):
    if isinstance(ref, MultiMolecule):
        return ref
    elif isinstance(ref, str):
        return MultiMolecule.from_xyz(ref)
    else:
        raise TypeError(TYPE_ERR2.format('ref', 'FOX.MultiMolecule', 'str', get_name(ref)))


def sanitize_param():
    pass


def sanitize_pes():
    pass


def sanitize_hdf5_path(hdf5_path):
    if not isinstance(hdf5_path, str):
        raise TypeError(TYPE_ERR.format('hdf5_path', 'str', get_name(hdf5_path)))

    elif hdf5_path.lower() in ('cwd', '.', 'none', 'pwd', '$pwd'):
        hdf5_path = os.getcwd()
    elif os.path.isdir(hdf5_path):
        return hdf5_path
    elif os.path.isdir(os.path.dirname(hdf5_path)):
        return os.path.dirname(hdf5_path)
    else:
        raise FileNotFoundError(hdf5_path + ' not found')


def sanitize_job(job):
    if job.func.lower() == 'cp2kjob':
        job.func = Cp2kJob
    elif not isinstance(job.func, Job):
        raise TypeError(TYPE_ERR.format('job.func', 'Job', get_name(job.func)))

    if isinstance(job.settings, dict):
        job.settings = Settings(job.Settings)
    elif isinstance(job.settings, str):
        job.settings = get_template(job.settings)
    else:
        raise TypeError(TYPE_ERR2.format('job.settings', 'dict', 'str', get_name(job.settings)))

    if not isinstance(job.name, str):
        raise TypeError(TYPE_ERR.format('job.name', 'str', get_name(job.name)))

    if not isinstance(job.psf, str):
        raise TypeError(TYPE_ERR.format('job.psf', 'str', get_name(job.psf)))

    if isinstance(job.path, str):
        if job.path.lower() in ('cwd', '.', 'none', 'pwd', '$pwd'):
            job.path = os.getcwd()
        elif not os.path.isdir(job.path):
            raise FileNotFoundError(job.path + ' not found')
    else:
        raise TypeError(TYPE_ERR.format('job.path', 'str', get_name(job.path)))

    if not isinstance(job.keep_files, bool):
        raise TypeError(TYPE_ERR.format('job.keep_files', 'bool', get_name(job.keep_files)))

    return job


def sanitize_move():
    pass
