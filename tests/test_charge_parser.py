"""A module for testing :mod:`FOX.functions.charge_parser` ."""

from assertionlib import assertion

from FOX.functions.charge_parser import assign_constraints


def test_assign_constraints() -> None:
    """Test :func:`assign_constraints`."""
    constraints = [
        'H < 1',
        'C >2.0',
        '-3.0> N',
        '- 4<O',
        '1 < F< 2.0',
        '2 > P >1.0',
        'S == 2 * Cl == 0.5*Br + -1.5*K - 1*Na == 1* I',
        '1 < H H < 2',
    ]
    extremite, constrain_list = assign_constraints(constraints)

    extremite_ref = {
        ('H', 'max'): 1.0,
        ('C', 'min'): 2.0,
        ('N', 'max'): -3.0,
        ('O', 'min'): -4.0,
        ('F', 'min'): 1.0,
        ('F', 'max'): 2.0,
        ('P', 'max'): 2.0,
        ('P', 'min'): 1.0,
        ('H H', 'min'): 1.0,
        ('H H', 'max'): 2.0
    }
    assertion.eq(extremite, extremite_ref)

    constrain_ref = [
        {"S": 1.0},
        {"Cl": 2.0},
        {"Br": 0.5, "K": -1.5, "Na": -1.0},
        {"I": 1.0}
    ]
    assertion.eq(constrain_list, constrain_ref)
