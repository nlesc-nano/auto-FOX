Radial Distribution Function Generator
======================================

A radial distribution function (RDF) generator has been implemented in the :mod:`FOX.functions.radial_distribution` module.
The radial distribution function, or pair correlation function, describes how the particale density in a system varies as a function of distance from a reference particle.
The herein implemented get_all_radial() function is designed for constructing RDFs between all possible (user-defined) atom-pairs.

.. autofunction:: FOX.functions.radial_distribution.get_all_radial

.. autofunction:: FOX.functions.radial_distribution.get_radial_distr

.. _np.ndarray: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html
.. _np.float64: https://docs.scipy.org/doc/numpy/user/basics.types.html
.. _pd.DataFrame: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
.. _dict: https://docs.python.org/3/library/stdtypes.html#dict
.. _str: https://docs.python.org/3/library/stdtypes.html#str
.. _list: https://docs.python.org/3/library/stdtypes.html#list
.. _int: https://docs.python.org/3/library/functions.html#int

.. |np.ndarray| replace:: *np.ndarray*
.. |np.float64| replace:: *np.float64*
.. |pd.DataFrame| replace:: *pd.DataFrame*
.. |dict| replace:: *dict*
.. |str| replace:: *str*
.. |list| replace:: *list*
.. |int| replace:: *int*
