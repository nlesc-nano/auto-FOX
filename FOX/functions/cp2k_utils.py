""" A module with miscellaneous functions related to CP2K. """

__all__ = ['set_subsys_kind', 'set_lennard_jones']

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
    """ Set the FORCE_EVAL/MM/FORCEFIELD/LENNARD-JONES_ keyword(s) in CP2K job settings.
    Performs an inplace update of the input.mm.forcefield key in **settings**.

    .. _LENNARD-JONES: https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/MM/\
    FORCEFIELD/NONBONDED/LENNARD-JONES.html

    :parameter settings: CP2K settings.
    :type settings: |plams.Settings|_
    :parameter dict lj_dict: A nested dictionary with atom pairs as keys
        (*e.g.* *Se Se* or *Cd OG2D2*). An overview of accepted values is provided in the
        LENNARD-JONES_ section of the CP2K documentation. The value assigned to the *rcut* key will
        default to 12.0 Ångström if not provided by the user.
    """
    key_map = map(''.join, itertools.product(*zip('LENNARD-JONES', 'lennard-jones')))
    for key, value in zip(key_map, lj_dict):
        settings.input.mm.forcefield[key] = {'name': value, 'rcut': 12.0}
        settings.input.mm.forcefield[key].update(lj_dict[value])
