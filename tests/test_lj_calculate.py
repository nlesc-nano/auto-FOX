"""A module for testing :mod:`FOX.functions.lj_calculate`."""

from pathlib import Path

import numpy as np

from scm.plams import Settings

from FOX import MultiMolecule, get_non_bonded

PATH = Path('tests') / 'test_files'


def test_get_non_bonded() -> None:
    """Test :func:`FOX.functions.lj_calculate.get_non_bonded`."""
    psf = PATH / 'Cd68Se55_26COO_MD_trajec.psf'
    prm = PATH / 'Cd68Se55_26COO_MD_trajec.prm'
    mol = MultiMolecule.from_xyz(PATH / 'Cd68Se55_26COO_MD_trajec.xyz')

    s = Settings()
    s.input.force_eval.mm.forcefield.charge = [
        {'atom': 'Cd', 'charge': 0.933347},
        {'atom': 'Se', 'charge': -0.923076}
    ]
    s.input.force_eval.mm.forcefield.nonbonded['lennard-jones'] = [
        {'atoms': 'Cd Cd', 'epsilon': '[kjmol] 0.310100', 'sigma': '[nm] 0.118464'},
        {'atoms': 'Se Se', 'epsilon': '[kjmol] 0.426600', 'sigma': '[nm] 0.485200'},
        {'atoms': 'Se Cd', 'epsilon': '[kjmol] 1.522500', 'sigma': '[nm] 0.294000'}
    ]

    ref = np.load(PATH / 'get_non_bonded.npy')
    df = get_non_bonded(mol, psf, prm=prm, cp2k_settings=s)
    np.testing.assert_allclose(df.values, ref)
