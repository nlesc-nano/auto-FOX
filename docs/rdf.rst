Radial Distribution Function
============================

A radial distribution function (RDF) generator has been implemented in the
:mod:`FOX.classes.multi_molecule` and :mod:`FOX.functions.rdf` modules.
The radial distribution function, or pair correlation function, describes how
the particale density in a system varies as a function of distance from a
reference particle. The herein implemented function is designed for
constructing RDFs between all possible (user-defined) atom-pairs.

.. math::

    g(r) =
    \frac{V}{N_a*N_b} \sum_{i=1}^{N_a} \sum_{j=1}^{N_b} \left< *placeholder* \right>


Given a trajectory, ``mol``, stored as a *MultiMolecule* object, the RDF can be
calculated with the following
command: ``rdf = mol.init_rdf(atom_subset=None, low_mem=False)``.
The resulting ``rdf`` is a Pandas_ dataframe, an object which is effectively a
hybrid between a dictionary and a Numpy_ array.

A slower, but more memory efficient, method of RDF construction can be enabled
with ``low_mem=True``, causing the script to only store the distance matrix
of a single molecule in memory at once. If ``low_mem=False``, all distance
matrices are stored in memory simultaneously, speeding up the calculation
but also introducing an additional linear scaling of memory with respect to
the number of molecules.

Below is an example RDF of a CdSe quantum dot pacified with formate ligands.
The RDF is printed for all possible combinations of cadmium, selenium and
oxygen (Cd_Cd, Cd_Se, Cd_O, Se_Se, Se_O and O_O).

::

    from FOX.classes.multi_mol import MultiMolecule
    from FOX.examples.example_xyz import get_example_xyz

    example_xyz_file = get_example_xyz()
    mol = MultiMolecule(filename=example_xyz_file)
    rmsd = mol.init_rdf(atom_subset=('Cd', 'Se', 'O'))
    rmsd.plot()


.. plot::

    from FOX.classes.multi_mol import MultiMolecule
    from FOX.examples.example_xyz import get_example_xyz
    atoms = ('Cd', 'Se', 'O')
    xyz_file = get_example_xyz()
    mol = MultiMolecule(filename=xyz_file)
    rmsd = mol.init_rdf(atom_subset=atoms)
    rmsd.plot()

API
---

.. automethod:: FOX.classes.multi_mol.MultiMolecule.init_rdf
    :noindex:

.. _Numpy: https://www.numpy.org/
.. _Pandas: https://pandas.pydata.org/
.. _plams.Settings: https://www.scm.com/doc/plams/components/settings.html
.. _plams.Molecule: https://www.scm.com/doc/plams/components/molecule.html#id1
.. _np.ndarray: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html
.. _np.float64: https://docs.scipy.org/doc/numpy/user/basics.types.html#array-types-and-conversions-between-types
.. _np.int64: https://docs.scipy.org/doc/numpy/user/basics.types.html#array-types-and-conversions-between-types
.. _pd.DataFrame: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
.. _dict: https://docs.python.org/3/library/stdtypes.html#dict
.. _list: https://docs.python.org/3/library/stdtypes.html#list
.. _tuple: https://docs.python.org/3/library/stdtypes.html#tuple
.. _str: https://docs.python.org/3/library/stdtypes.html#str
.. _int: https://docs.python.org/3/library/functions.html#int
.. _None: https://docs.python.org/3.7/library/constants.html#None

.. |plams.Molecule| replace:: *plams.Molecule*
.. |plams.Settings| replace:: *plams.Settings*
.. |np.ndarray| replace:: *np.ndarray*
.. |np.float64| replace:: *np.float64*
.. |np.int64| replace:: *np.int64*
.. |pd.DataFrame| replace:: *pd.DataFrame*
.. |dict| replace:: *dict*
.. |list| replace:: *list*
.. |tuple| replace:: *tuple*
.. |str| replace:: *str*
.. |int| replace:: *int*
.. |None| replace:: *None*
