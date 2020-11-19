"""Functions related to the Lennard-Jones parameters ...

Index
-----
.. currentmodule:: FOX.ff.shannon_radii
.. autosummary::
    RADII_DF
    SIGMA_DF

API
---
.. autodata:: RADII_DF
    :annotation: : pandas.DataFrame

    .. code:: python

        >>> print(RADII_DF)
        Å    charge coordination spin_state  crystal_radius  ionic_radius key*
        ion
        Ac        3           VI        NaN            1.26          1.12    R
        Ag        1           II        NaN            0.81          0.67  NaN
        Ag        1           IV        NaN            1.14          1.00    C
        Ag        1         IVSQ        NaN            1.16          1.02  NaN
        Ag        1            V        NaN            1.23          1.09    C
        ..      ...          ...        ...             ...           ...  ...
        Zr        4            V        NaN            0.80          0.66    C
        Zr        4           VI        NaN            0.86          0.72   R*
        Zr        4          VII        NaN            0.92          0.78    *
        Zr        4         VIII        NaN            0.98          0.84    *
        Zr        4           IX        NaN            1.03          0.89  NaN


.. autodata:: SIGMA_DF
    :annotation: : pandas.DataFrame

    .. code:: python

        >>> print(SIGMA_DF)
        Å    crystal_sigma  ionic_sigma
        ion
        Ac        2.245065     1.995613
        Ag        1.963217     1.713765
        Al        1.083927     0.834475
        Am        2.195429     1.945977
        As        1.066109     0.816657
        ..             ...          ...
        Xe        1.033443     0.783991
        Y         2.010758     1.761307
        Yb        2.046522     1.797070
        Zn        1.550164     1.300712
        Zr        1.579860     1.330409

"""

from os.path import join, dirname

import pandas as pd

__all__ = ['SIGMA_DF']


#: Absolute path to the ``FOX.data.shannon_radii`` .csv file.
_CSV: str = join(dirname(dirname(__file__)), 'data', 'shannon_radii.csv')

#: A :class:`pandas.DataFrame` with ionic and crystal radii.
#: Data taken from http://abulafia.mt.ic.ac.uk/shannon/radius.php.
#:
#: See `10.1107/S0567739476001551 <https://doi.org/10.1107/S0567739476001551>`_:
#: R. D. Shannon, Revised effective ionic radii and systematic studies of
#: interatomic distances in halides and chalcogenides, *Acta Cryst.* (1976). A32, 751-767.
RADII_DF: pd.DataFrame = pd.read_csv(_CSV, index_col=0)
RADII_DF.columns.name = 'Å'
del _CSV

#: A :class:`pandas.DataFrame` with :math:`\sigma` values based on ionic
#: and crystal radii.
#: Values are averaged with respect to all charges and coordination numbers per atom type.
#: Data taken from http://abulafia.mt.ic.ac.uk/shannon/radius.php.
#:
#: See `10.1107/S0567739476001551 <https://doi.org/10.1107/S0567739476001551>`_:
#: R. D. Shannon, Revised effective ionic radii and systematic studies of
#: interatomic distances in halides and chalcogenides, *Acta Cryst.* (1976). A32, 751-767.
SIGMA_DF: pd.DataFrame = RADII_DF[['crystal_radius', 'ionic_radius']].groupby(RADII_DF.index).mean()
SIGMA_DF.columns = pd.Index([i.replace('radius', 'sigma') for i in SIGMA_DF.columns], name='Å')
SIGMA_DF *= 2 / 2**(1/6)  # Conversion factor between R / 2 and the Lennard-Jones sigma parameter
