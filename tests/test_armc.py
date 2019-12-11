"""A module for testing :class:`FOX.classes.armc.ARMC`."""

import os
from pathlib import Path

import numpy as np
from assertionlib import assertion

from FOX import ARMC

PATH = Path('tests') / 'test_files' / 'armc'


def test_to_yaml() -> None:
    """Test :meth:`ARMC.to_yaml` and :meth:`ARMC.from_yaml`."""
    armc1, job_kwarg = ARMC.from_yaml(PATH / 'armc1.yaml')

    for psf in job_kwarg['psf']:
        psf.write(None)

    armc1.to_yaml(PATH / 'armc2.yaml', path=PATH)
    armc2, _ = ARMC.from_yaml(PATH / 'armc2.yaml')

    try:
        assertion.eq(armc1.sub_iter_len, armc2.sub_iter_len)
        assertion.allclose(armc1.rmsd_threshold, armc2.rmsd_threshold)
        assertion.is_(armc1.preopt_settings, armc2.preopt_settings)

        for mol1, mol2 in zip(armc1._plams_molecule, armc2._plams_molecule):
            np.testing.assert_allclose(mol1.as_array(), mol2.as_array())
        assertion.allclose(armc1.phi, armc2.phi)

        for k, v1 in armc1.pes.items():
            v2 = armc2.pes[k]
            for p1, p2 in zip(v1, v2):
                assertion.is_(p1.func, p2.func)
                assertion.eq(p1.keywords, p2.keywords)
                np.testing.assert_allclose(p1.ref.values, p2.ref.values)

        np.testing.assert_allclose(armc1.param['param'], armc2.param['param'])
        np.testing.assert_allclose(armc1.param['min'], armc2.param['min'])
        np.testing.assert_allclose(armc1.param['max'], armc2.param['max'])
        np.testing.assert_array_equal(armc1.param['unit'], armc2.param['unit'])
        np.testing.assert_array_equal(armc1.param['keys'], armc2.param['keys'])
        np.testing.assert_array_equal(armc1.param['count'], armc2.param['count'])
        assertion(armc1.param['constraints'].isnull().all())
        assertion(armc2.param['constraints'].isnull().all())

        np.testing.assert_allclose(armc1.move_range, armc2.move_range)
        for mol1, mol2 in zip(armc1.molecule, armc2.molecule):
            np.testing.assert_allclose(mol1, mol2)

        assertion.is_(armc1.keep_files, armc2.keep_files)
        assertion.is_(armc1.job_type.func, armc2.job_type.func)
        assertion.eq(armc1.job_type.keywords, armc2.job_type.keywords)
        assertion.eq(armc1.job_cache, armc2.job_cache)
        assertion.eq(armc1.iter_len, armc2.iter_len)
        assertion.eq(armc1.history_dict, armc2.history_dict)
        assertion.eq(armc1.hdf5_file, armc2.hdf5_file)
        assertion.allclose(armc1.gamma, armc2.gamma)
        assertion.is_(armc1.apply_phi, armc2.apply_phi)
        assertion.is_(armc1.apply_move.func, armc2.apply_move.func)
        assertion.allclose(armc1.a_target, armc2.a_target)

    finally:
        os.remove(PATH / 'mol.0.xyz') if os.path.isfile(PATH / 'mol.0.xyz') else None
        os.remove(PATH / 'mol.0.psf') if os.path.isfile(PATH / 'mol.0.psf') else None
        os.remove(PATH / 'armc2.yaml') if os.path.isfile(PATH / 'armc2.yaml') else None
