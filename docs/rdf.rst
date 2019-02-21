Radial Distribution Function Generator
======================================

A radial distribution function (RDF) generator has been implemented in the
:mod:`FOX.functions.rdf` module. The radial distribution function, or pair
correlation function, describes how the particale density in a system varies
as a function of distance from a reference particle. The herein implemented
:func:`FOX.functions.rdf.get_all_radial` function is designed for constructing
RDFs between all possible (user-defined) atom-pairs.

A slower but more memory efficient method of RDF construction can be enabled
with the **low_mem** argument in :func:`FOX.functions.rdf.get_all_radial`.
Given *m* molecules, an equal number of distance matrices has to be created per
atom pair. When **low_mem**: *False*, all distance matrices are simultaneously
created and stored in memory (per set of atom pairs), taking advantage of
Numpy's vectorized operations. When **low_mem**: *True*, only a single distance
matrix is stored in memory at once, removing the linear scaling of memory with
respect to the number of molecules.

#.. autofunction:: FOX.functions.rdf.get_all_radial

#.. autofunction:: FOX.functions.rdf.get_radial_distr

#.. autofunction:: FOX.functions.rdf.get_radial_distr_lowmem

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
