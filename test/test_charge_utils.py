""" A module for testing files in the :mod:`FOX.functions.charge_utils` module. """

__all__ = []

from os.path import join

import numpy as np

from scm.plams import Settings

import FOX
from FOX import (MultiMolecule, get_example_xyz)
from FOX.functions.charge_utils import (
    update_charge, update_constrained_charge, update_unconstrained_charge,
    get_charge_constraints, invert_ufunc
)


MOL = MultiMolecule.from_xyz(get_example_xyz())
MOL.guess_bonds(atom_subset=['C', 'O', 'H'])
MOL.update_atom_type(join(FOX.__path__[0], 'examples/ligand.str'))
MOL.update_atom_charge('Cd', 2.0)
MOL.update_atom_charge('Se', -2.0)
DF = MOL.properties.psf


def test_update_charge():
    """ Test :func:`FOX.functions.charge_utils.update_charge`. """
    df = DF.copy()

    net_charge = df['charge'].sum()
    constrain_dict = get_charge_constraints('Cd = -0.5 * O_1 = -1 * Se')
    exclude = ['H_1']
    update_charge('Cd', 1.9, df, constrain_dict, exclude)

    Cd_charge = df.loc[df['atom type'] == 'Cd', 'charge'].iloc[0]
    O_charge = df.loc[df['atom type'] == 'O_1', 'charge'].iloc[0]
    Se_charge = df.loc[df['atom type'] == 'Se', 'charge'].iloc[0]
    H_charge = df.loc[df['atom type'] == 'H_1', 'charge'].iloc[0]

    assert Cd_charge == 1.9
    assert Se_charge == -1 * Cd_charge
    assert O_charge == -0.5 * Cd_charge
    assert H_charge <= 10**-8
    assert np.abs(df['charge'].sum() - net_charge) <= 10**-8

    net_charge = df['charge'].sum()
    exclude = ['H_1']
    update_charge('C_1', -1.0, df, constrain_dict, exclude)

    Cd_charge = df.loc[df['atom type'] == 'Cd', 'charge'].iloc[0]
    O_charge = df.loc[df['atom type'] == 'O_1', 'charge'].iloc[0]
    Se_charge = df.loc[df['atom type'] == 'Se', 'charge'].iloc[0]
    H_charge = df.loc[df['atom type'] == 'H_1', 'charge'].iloc[0]
    C_charge = df.loc[df['atom type'] == 'C_1', 'charge'].iloc[0]

    assert C_charge == -1.0
    assert Se_charge == -1 * Cd_charge
    assert O_charge == -0.5 * Cd_charge
    assert H_charge <= 10**-8
    assert np.abs(df['charge'].sum() - net_charge) <= 10**-8


def test_update_constrained_charge():
    """ Test :func:`FOX.functions.charge_utils.update_constrained_charge`. """
    df = DF.copy()

    constrain_dict = get_charge_constraints('Cd = -0.5 * O_1 = -1 * Se')
    df.loc[df['atom type'] == 'Cd', 'charge'] = 2.5
    update_constrained_charge('Cd', df, constrain_dict)

    Cd_charge = df.loc[df['atom type'] == 'Cd', 'charge'].iloc[0]
    O_charge = df.loc[df['atom type'] == 'O_1', 'charge'].iloc[0]
    Se_charge = df.loc[df['atom type'] == 'Se', 'charge'].iloc[0]
    assert Cd_charge == 2.5
    assert Se_charge == -1 * Cd_charge
    assert O_charge == -0.5 * Cd_charge

    constrain_dict = get_charge_constraints('Cd = 0.5 + O_1 = -1 + Se = H_1 * 0.9')
    df.loc[df['atom type'] == 'Cd', 'charge'] = 1.5
    update_constrained_charge('Cd', df, constrain_dict)

    Cd_charge = df.loc[df['atom type'] == 'Cd', 'charge'].iloc[0]
    O_charge = df.loc[df['atom type'] == 'O_1', 'charge'].iloc[0]
    Se_charge = df.loc[df['atom type'] == 'Se', 'charge'].iloc[0]
    H_charge = df.loc[df['atom type'] == 'H_1', 'charge'].iloc[0]
    assert Cd_charge == 1.5
    assert Se_charge == -1 + Cd_charge
    assert O_charge == 0.5 + Cd_charge
    assert H_charge == 0.9 * Cd_charge


def test_update_unconstrained_charge():
    """ Test :func:`FOX.functions.charge_utils.update_unconstrained_charge`. """
    df = DF.copy()

    update_unconstrained_charge(-1, df)
    assert df['charge'].sum() - -1.0 <= 10**-8

    update_unconstrained_charge(2, df)
    assert df['charge'].sum() - 2.0 <= 10**-8

    condition = df['atom type'] == 'Cd'
    ref = df['charge']
    update_unconstrained_charge(-10.0, df, exclude=['Cd'])
    assert df['charge'].sum() - 10.0 <= 10**-8
    np.testing.assert_allclose(df.loc[condition, 'charge'], ref[condition])


def test_get_charge_constraints():
    """ Test :func`:FOX.functions.charge_utils.get_charge_constraints`. """
    constrain1 = 'Cd = -0.5 * O = -1 * Se'
    constrain2 = 'Cd = H + -1 = O + 1.5 = 0.5 * Se'

    ref1 = Settings()
    ref1['Cd'] = {'arg': 1.0, 'func': np.multiply}
    ref1['O'] = {'arg': -0.5, 'func': np.multiply}
    ref1['Se'] = {'arg': -1.0, 'func': np.multiply}

    ref2 = Settings()
    ref2['Cd'] = {'arg': 1.0, 'func': np.multiply}
    ref2['H'] = {'arg': -1.0, 'func': np.add}
    ref2['O'] = {'arg': 1.5, 'func': np.add}
    ref2['Se'] = {'arg': 0.5, 'func': np.multiply}

    s1 = get_charge_constraints(constrain1)
    s2 = get_charge_constraints(constrain2)

    assert s1 == ref1
    assert s2 == ref2


def test_invert_ufunc():
    """ Test :func:`FOX.functions.charge_utils.invert_ufunc`. """
    assert invert_ufunc(np.add) == np.subtract
    assert invert_ufunc(np.multiply) == np.divide

    try:
        invert_ufunc(np.stack)
    except KeyError:
        pass
    else:
        raise AssertionError
