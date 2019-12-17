"""A module for testing :mod:`FOX.armc_functions.guess`."""

from pathlib import Path

import yaml
import numpy as np
from assertionlib import assertion

from FOX import ARMC as ARMCType
from FOX.armc_functions.guess import guess_param

PATH = Path('tests') / 'test_files'
ARMC, _ = ARMCType.from_yaml(PATH / 'armc.yaml')
ARMC.md_settings[0].input.force_eval.subsys.topology.conn_file_name = str(PATH / 'mol.0.psf')


def test_guess_param() -> None:
    """Test :func:`FOX.armc_functions.guess.guess_param`."""
    armc = ARMC.copy(deep=True)
    assertion.assert_(guess_param, armc, mode='bob', exception=ValueError)
    assertion.assert_(guess_param, armc, mode=1, exception=ValueError)

    guess_param(armc, mode='rdf')
    param1 = armc.param['param']
    ref1 = np.load(PATH / 'guess_param_rdf.npy')
    np.testing.assert_allclose(param1, ref1, rtol=1e-06)
    with open(PATH / 'guess_param_rdf.yaml', 'r') as f:
        s1 = armc.md_settings[0].input.force_eval.mm.forcefield.nonbonded
        s1_ref = yaml.load(f, Loader=yaml.Loader)
        assertion.eq(s1, s1_ref)

    armc = ARMC.copy(deep=True)
    guess_param(armc, mode='uff')
    param2 = armc.param['param']
    ref2 = np.load(PATH / 'guess_param_uff.npy')
    np.testing.assert_allclose(param2, ref2, rtol=1e-06)
    with open(PATH / 'guess_param_uff.yaml', 'r') as f:
        s2 = armc.md_settings[0].input.force_eval.mm.forcefield.nonbonded
        s2_ref = yaml.load(f, Loader=yaml.Loader)
        assertion.eq(s2, s2_ref)

    armc = ARMC.copy(deep=True)
    guess_param(armc, mode='rdf', frozen=['sigma', 'epsilon'])
    np.testing.assert_allclose(armc.param['param'], ARMC.param['param'].sort_index())

    armc = ARMC.copy(deep=True)
    guess_param(armc, mode='uff', frozen=['sigma', 'epsilon'])
    np.testing.assert_allclose(armc.param['param'], ARMC.param['param'].sort_index())
