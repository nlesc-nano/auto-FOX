"""A module for testing :mod:`FOX.armc_functions.guess`."""

from pathlib import Path

import numpy as np
from assertionlib import assertion

from FOX import MultiMolecule
from FOX.armc import guess_param

PATH = Path('tests') / 'test_files'

MOL_LIST = [MultiMolecule.from_xyz(PATH / 'Cd68Se55_26COO_MD_trajec.xyz')]
PSF_LIST = [PATH / 'Cd68Se55_26COO_MD_trajec.psf']
PRM = PATH / 'Cd68Se55_26COO_MD_trajec.prm'


def test_guess_param() -> None:
    """Test :func:`FOX.armc.guess.guess_param`."""
    ar = np.array([
        guess_param(MOL_LIST, 'sigma', mode='rdf', prm=PRM, psf_list=PSF_LIST),
        guess_param(MOL_LIST, 'sigma', mode='uff', prm=PRM, psf_list=PSF_LIST),
        guess_param(MOL_LIST, 'sigma', mode='crystal_radius', prm=PRM, psf_list=PSF_LIST),
        guess_param(MOL_LIST, 'sigma', mode='ion_radius', prm=PRM, psf_list=PSF_LIST),
        guess_param(MOL_LIST, 'epsilon', mode='rdf', prm=PRM, psf_list=PSF_LIST),
        guess_param(MOL_LIST, 'epsilon', mode='uff', prm=PRM, psf_list=PSF_LIST)
    ])

    ref = np.load(PATH / 'guess_param.npy')
    np.testing.assert_allclose(ar, ref)

    assertion.assert_(guess_param, MOL_LIST, 'epsilon', 'crystal_radius',
                      exception=NotImplementedError)
    assertion.assert_(guess_param, MOL_LIST, 'epsilon', 'ion_radius', exception=NotImplementedError)
    assertion.assert_(guess_param, MOL_LIST, 'bob', 'crystal_radius', exception=ValueError)
    assertion.assert_(guess_param, MOL_LIST, 'epsilon', int, exception=TypeError)
