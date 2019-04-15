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


def set_lennard_jones(settings, lj_df):
    """ Set the FORCE_EVAL/MM/FORCEFIELD/NONBONDED/LENNARD-JONES_ keyword(s) in CP2K job settings.
    Performs an inplace update of **lj_df** and the input.mm.forcefield.nonbonded key in
    **settings**.

    .. _LENNARD-JONES: https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/MM/\
    FORCEFIELD/NONBONDED/LENNARD-JONES.html

    :parameter settings: CP2K settings.
    :type settings: |plams.Settings|_
    :parameter lj_df: A nested dictionary with atom pairs as keys
        (*e.g.* *Se Se* or *Cd OG2D2*). An overview of accepted values is provided in the
        LENNARD-JONES_ section of the CP2K documentation. The value assigned to the *rcut* key will
        default to 12.0 Ångström if not provided by the user.
    :return:
    :rtype: |pd.Series|_ (index: |pd.Index|_, values: |str|_)
    """
    lj_df['key'] = None
    key_map = map(''.join, itertools.product(*zip('LENNARD-JONES', 'lennard-jones')))
    for i, j in zip(key_map, lj_df.index):
        settings.input.force_eval.mm.forcefield.nonbonded[i] = {'name': j, 'rcut': 12.0}
        lj_df.at[j, 'key'] = i
        dict_ = lj_df.loc[j, ['epsilon', 'sigma']].to_dict()
        dict_['atoms'] = dict_['name']
        del dict_['name']
        settings.input.force_eval.mm.forcefield.nonbonded[i].update(dict_)


def set_atomic_charges(settings, charge_df):
    """ Set the FORCE_EVAL/MM/FORCEFIELD/CHARGE_ keyword(s) in CP2K job settings.
    Performs an inplace update of **charge_df** and the input.mm.forcefield key in **settings**.

    .. _CHARGE: https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/MM/FORCEFIELD/CHARGE.html

    :parameter settings: CP2K settings.
    :type settings: |plams.Settings|_
    :parameter dict charge_df: A dictionary with atom types as keys
        (*e.g.* *Se*, *Cd* or *COG2D2*) and matching atomic charge as values.
    """
    charge_df['key'] = None
    key_map = map(''.join, itertools.product(*zip('CHARGE', 'charge')))
    for i, j in zip(key_map, charge_df.index):
        charge_df.at[j, 'key'] = i
        settings.input.force_eval.mm.forcefield[i] = {
            'atom': j,
            'charge': charge_df.at[j, 'charge']
        }


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
