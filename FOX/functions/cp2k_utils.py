""" A module with miscellaneous functions related to CP2K. """

__all__ = ['update_cp2k_settings', 'set_keys']

import itertools

from scm.plams import Settings


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
        settings.input.force_eval.mm.forcefield.nonbonded[i] = {'atoms': j, 'rcut': 12.0}
        lj_df.at[j, 'key'] = i
        dict_ = lj_df.loc[j, ['epsilon', 'sigma']].to_dict()
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
    lj = settings.input.force_eval.mm.forcefield.nonbonded['lennard-jones']

    # Update the Lennard-Jones epsilon parameters
    for _, (param_, unit, i, *_) in param.loc['epsilon', :].iterrows():
        lj[int(i)].epsilon = unit.format(param_)

    # Update the Lennard-Jones sigma parameters
    for _, (param_, unit, i, *_) in param.loc['sigma', :].iterrows():
        lj[int(i)].sigma = unit.format(param_)


def set_keys(settings, param, rcut=12.0,):
    r""" Find and return the keys in **settings** matching all parameters in **param**.

    Units can be specified under the *unit* key (see the CP2K_ documentation for more details).

    :parameter settings: CP2K Job settings.
    :type settings: |plams.Settings|_
    :parameter param: A dataframe with MM parameters and parameter names as 2-level multiindex.
    :type param: |pd.DataFrame|_ (index: |pd.MultiIndex|_)
    :return: A list of all matched keys.
    :rtype: |list|_ [|str|_]

    .. _CP2K: https://manual.cp2k.org/trunk/units.html
    """
    def _get_settings(param, at):
        eps_unit = param.loc[('epsilon', at), 'unit']
        eps_value = param.loc[('epsilon', at), 'param']
        sigma_unit = param.loc[('sigma', at), 'unit']
        sigma_value = param.loc[('sigma', at), 'param']
        return Settings({'epsilon': eps_unit.format(eps_value),
                         'sigma': sigma_unit.format(sigma_value),
                         'rcut': rcut,
                         'atoms': at})

    # Create a new column in **param** with the quantity units
    param['unit'] = None
    units = {'epsilon': '[K_e] {:f}', 'sigma': '[angstrom] {:f}'}
    for key, value in units.items():
        try:
            param.loc[[key], 'unit'] = '[{}] {}'.format(param.loc[(key, 'unit'), 'param'], '{:f}')
            param.drop(index=[(key, 'unit')], inplace=True)
        except KeyError:
            param.loc[key, 'unit'] = value

    # Generate the keys for atomic charges (note: there are no charge keys)
    key_list = [-1 for _ in param.loc['charge'].index]

    # Create a list for all CP2K &LENNARD-JONES blocks
    lj_list = [_get_settings(param, at) for at in param.loc['epsilon'].index]
    settings.input.force_eval.mm.forcefield.nonbonded.update({'lennard-jones': lj_list})
    key_list += 2 * [i for i, _ in enumerate(param.loc['epsilon'].index)]
    return key_list
