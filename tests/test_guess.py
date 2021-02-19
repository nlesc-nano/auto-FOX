"""A module for testing :mod:`FOX.armc_functions.guess`."""

from pathlib import Path
from typing import Mapping, Any, Type

import pytest
import numpy as np

from FOX import MultiMolecule
from FOX.armc import guess_param

PATH = Path('tests') / 'test_files'

MOL_LIST = [MultiMolecule.from_xyz(PATH / 'Cd68Se55_26COO_MD_trajec.xyz')]
PSF_LIST = [PATH / 'Cd68Se55_26COO_MD_trajec.psf']
PRM = PATH / 'Cd68Se55_26COO_MD_trajec.prm'

REF = np.load(PATH / 'guess_param.npy')


class TestGuesParam:
    """Test :func:`FOX.armc.guess.guess_param`."""

    @pytest.mark.parametrize(
        "i,kwargs",
        [
            (0, dict(param='sigma', mode='rdf', prm=PRM, psf_list=PSF_LIST)),
            (1, dict(param='sigma', mode='uff', prm=PRM, psf_list=PSF_LIST)),
            (2, dict(param='sigma', mode='crystal_radius', prm=PRM, psf_list=PSF_LIST)),
            (3, dict(param='sigma', mode='ion_radius', prm=PRM, psf_list=PSF_LIST)),
            (4, dict(param='epsilon', mode='rdf', prm=PRM, psf_list=PSF_LIST)),
            (5, dict(param='epsilon', mode='uff', prm=PRM, psf_list=PSF_LIST)),
        ],
        ids=["sigma/rdf", "sigma/uff", "sigma/crystal_radius", "sigma/ion_radius",
             "epsilon/rdf", "epsilon/uff"],
    )
    def test_passes(self, i: int, kwargs: Mapping[str, Any]) -> None:
        ar = guess_param(MOL_LIST, **kwargs)
        ref = REF[i]
        np.testing.assert_allclose(ar, ref, rtol=1e-05)

    @pytest.mark.parametrize(
        "kwargs,exc",
        [
            (dict(param='epsilon', mode='crystal_radius'), NotImplementedError),
            (dict(param='epsilon', mode='ion_radius'), NotImplementedError),
            (dict(param='bob', mode='crystal_radius'), ValueError),
            (dict(param='epsilon', mode=int), TypeError),
        ]
    )
    def test_raises(self, kwargs: Mapping[str, Any], exc: Type[Exception]) -> None:
        with pytest.raises(exc):
            guess_param(MOL_LIST, **kwargs)
