Multi-XYZ reader
================

A reader of multi-xyz files has been implemented in the
:mod:`FOX.functions.read_xyz` module. The .xyz fileformat is designed
for storing the atomic symbols and cartesian coordinates of one or more
molecules. The herein implemented read_multi_xyz() function allows for
the fast, and memory-effiecient, retrieval of the various molecular
geometries stored in an .xyz file.

.. autofunction:: FOX.functions.read_xyz.read_multi_xyz

.. _plams.Settings: https://www.scm.com/doc/plams/components/settings.html
.. _np.ndarray: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html
.. _np.float64: https://docs.scipy.org/doc/numpy/user/basics.types.html
.. _dict: https://docs.python.org/3/library/stdtypes.html#dict
.. _str: https://docs.python.org/3/library/stdtypes.html#str
.. _list: https://docs.python.org/3/library/stdtypes.html#list
.. _int: https://docs.python.org/3/library/functions.html#int

.. |plams.Settings| replace:: *plams.Settings*
.. |np.ndarray| replace:: *np.ndarray*
.. |np.float64| replace:: *np.float64*
.. |dict| replace:: *dict*
.. |str| replace:: *str*
.. |list| replace:: *list*
.. |int| replace:: *int*
