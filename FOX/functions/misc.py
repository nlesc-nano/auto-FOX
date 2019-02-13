""" A module with work in progress miscellaneous functions """

import numpy as np

# from scm.plams.tools.units import Units


def get_rel_error(g_QM, g_MM, T=298.15, unit='kcal/mol'):
    """ Return the relative error defined as dF_rij = dXi = Xi_QM - Xi_MM.

    g_QM & G_MM <np.ndarray>: A m*n numpy arrays of m radial distribution functions of QM & MM
        calculations, respectively.
    T <float>: The temperature in Kelvin.
    unit <str>: The unit of the to be returned energy
    return <np.array>: The relative error dXi.
    """
    RT = 0.00198720 * T # * Units.conversion_ratio('kcal/mol', unit)
    return -RT * np.log1p((g_MM - g_QM) / g_QM)


def get_aux_error(g_QM, g_MM):
    """ Return the auxiliary error defined as dEps = Eps_QM - Eps_MM.

    g_QM & G_MM <np.ndarray>: A m*n numpy arrays of m radial distribution functions of QM & MM
        calculations, respectively.
    return <float>: The auxiliary error dEps.
    """
    return np.linalg.norm(g_QM - g_MM, axis=0).sum()


def get_increment(phi_old, a_old, a, y=2.0):
    """ Return the incremental factor Phi for the (k)th block.
    phi_old <float>: Phi_Omega produced in the (k - 1)th.

    a_old <np.ndarray>: The accepatance rates from all Omega iterations in the (k - 1)th block.
    a <float>: The target accepatance rate.
    y <float>: Regulates the correction of Phi_old, must be larger than or equal to 1.0.
    return <float>: The new incremental factor Phi.
    """
    return phi_old * y**np.sign(a - a_old.mean())
