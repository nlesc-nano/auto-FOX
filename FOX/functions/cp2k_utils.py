""" A module with miscellaneous functions related to CP2K. """

__all__ = ['set_subsys_kind', 'set_lennard_jones', 'set_atomic_charges', 'update_cp2k_settings']

import itertools


def set_subsys_kind(settings, df):
    """ Set the FORCE_EVAL/SUBSYS/KIND_ keyword(s) in CP2K job settings.
    Performs an inplace update of the input.force_eval.subsys key in **settings**.

    .. _KIND: https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/SUBSYS/KIND.html

    :parameter settings: CP2K settings.
    :type settings: |plams.Settings|_
    :parameter df: A dataframe with atom names (*e.g.* *O*, *H* & *C*) and atom types
        (*e.g.* *OG2D2*, *HGR52* & *CG2O3*).
    :type df: |pd.DataFrame|_
    """
    for at_name, at_type in df[['atom name', 'atom type']].values:
        if not settings.input.force_eval.subsys['kind ' + at_type]:
            settings.input.force_eval.subsys['kind ' + at_type] = {'element': at_name}


def set_lennard_jones(settings, lj_dict):
    """ Set the FORCE_EVAL/MM/FORCEFIELD/NONBONDED/LENNARD-JONES_ keyword(s) in CP2K job settings.
    Performs an inplace update of the input.mm.forcefield.nonbonded key in **settings**.

    .. _LENNARD-JONES: https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/MM/\
    FORCEFIELD/NONBONDED/LENNARD-JONES.html

    :parameter settings: CP2K settings.
    :type settings: |plams.Settings|_
    :parameter dict lj_dict: A nested dictionary with atom pairs as keys
        (*e.g.* *Se Se* or *Cd OG2D2*). An overview of accepted values is provided in the
        LENNARD-JONES_ section of the CP2K documentation. The value assigned to the *rcut* key will
        default to 12.0 Ångström if not provided by the user.
    """
    ret = {}
    key_map = map(''.join, itertools.product(*zip('LENNARD-JONES', 'lennard-jones')))
    for key1, key2 in zip(key_map, lj_dict):
        settings.input.mm.forcefield.nonbonded[key1] = {'name': key2, 'rcut': 12.0}
        settings.input.mm.forcefield.nonbonded[key1].update(lj_dict[key2])
        ret[key2] = key1
    return ret


def set_atomic_charges(settings, charge_dict):
    """ Set the FORCE_EVAL/MM/FORCEFIELD/CHARGE_ keyword(s) in CP2K job settings.
    Performs an inplace update of the input.mm.forcefield key in **settings**.

    .. _CHARGE: https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/MM/FORCEFIELD/CHARGE.html

    :parameter settings: CP2K settings.
    :type settings: |plams.Settings|_
    :parameter dict charge_dict: A dictionary with atom types as keys
        (*e.g.* *Se*, *Cd* or *COG2D2*) and matching atomic charge as values.
    """
    ret = {}
    key_map = map(''.join, itertools.product(*zip('CHARGE', 'charge')))
    for key1, key2 in zip(key_map, charge_dict):
        settings.input.mm.forcefield[key1] = {'atom': key2, 'charge': charge_dict[key2]}
        ret[key2] = key1
    return ret


def update_cp2k_settings(settings, param):
    """ Update CP2K job settings with those provided in param.

    :parameter settings: The CP2K job settings.
    :type settings: |plams.Settings|_
    :parameter param: A dataframe with (variable) forcefield parameters.
    :type param: |pd.DataFrame|_
    """
    # Update atomic charges
    charge = param.loc['charge', :]
    for i, j in zip(charge['keys'], charge['param']):
        settings.input.mm.forcefield[i].charge = j

    # Update the Lennard-Jones epsilon parameters
    epsilon = param.loc['epsilon', :]
    for i, j in zip(epsilon['keys'], epsilon['param']):
        settings.input.mm.forcefield.nonbonded[i].epsilon = j

    # Update the Lennard-Jones sigma parameters
    sigma = param.loc['sigma', :]
    for i, j in zip(sigma['keys'], sigma['param']):
        settings.input.mm.forcefield.nonbonded[i].sigma = j
