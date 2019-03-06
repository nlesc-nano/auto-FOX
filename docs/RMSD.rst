Root Mean Squared Displacement & Fluctuation
============================================

Root Mean Squared Displacement
------------------------------

The root mean squared displacement (RMSD) represents the average displacement
of a set or subset of atoms as a function of time or, equivalently,
moleculair indices in a MD trajectory.

.. math::

    \rho^{\mathrm{RMSD}}(t) =
    \sqrt{
        \frac{1}{N} \sum_{i=1}^{N}\left(
            \mathbf{r}_{i}(t) - \mathbf{r}_{i}^{\mathrm{ref}}\right
        )^2
    }

Given a trajectory, ``mol``, stored as a *MultiMolecule* object, the RMSD
can be calculated with the following command:
``rmsd = mol.init_rmsd(atom_subset=None)``.
The resulting ``rmsd`` is a Pandas_ dataframe, an object which is effectively a
hybrid between a dictionary and a Numpy_ array.

Below is an example RMSD of a CdSe quantum dot pacified with formate ligands.
The RMSD is printed for cadmium, selenium and oxygen atoms.

::

    from FOX.classes.multi_mol import MultiMolecule
    from FOX.examples.example_xyz import get_example_xyz

    example_xyz_file = get_example_xyz()
    mol = MultiMolecule(filename=example_xyz_file)
    rmsd = mol.init_rmsd(atom_subset=('Cd', 'Se', 'O'))
    rmsd.plot()


.. plot::

    from FOX.classes.multi_mol import MultiMolecule
    from FOX.examples.example_xyz import get_example_xyz
    atoms = ('Cd', 'Se', 'O')
    xyz_file = get_example_xyz()
    mol = MultiMolecule(filename=xyz_file)
    rmsd = mol.init_rmsd(atom_subset=atoms)
    rmsd.plot()

Root Mean Squared Fluctuation
-----------------------------

The root mean squared fluctuation (RMSD) represents the time-averaged
displacement, with respect to the time-averaged position, as a function
of atomic indices.

.. math::

    \rho^{\mathrm{RMSF}}_i =
    \sqrt{
        \left\langle
        \left(\mathbf{r}_i - \langle \mathbf{r}_i \rangle \right)^2
        \right\rangle
    }

Given a trajectory, ``mol``, stored as a *MultiMolecule* object, the RMSF
can be calculated with the following command:
``rmsf = mol.init_rmsf(atom_subset=None)``.
The resulting ``rmsf`` is a Pandas_ dataframe, an object which is effectively a
hybrid between a dictionary and a Numpy_ array.

Below is an example RMSF of a CdSe quantum dot pacified with formate ligands.
The RMSF is printed for cadmium, selenium and oxygen atoms.

::

    from FOX.classes.multi_mol import MultiMolecule
    from FOX.examples.example_xyz import get_example_xyz

    example_xyz_file = get_example_xyz()
    mol = MultiMolecule(filename=example_xyz_file)
    rmsd = mol.init_rmsf(atom_subset=('Cd', 'Se', 'O'))
    rmsd.plot()


.. plot::

    from FOX.classes.multi_mol import MultiMolecule
    from FOX.examples.example_xyz import get_example_xyz
    atoms = ('Cd', 'Se', 'O')
    xyz_file = get_example_xyz()
    mol = MultiMolecule(filename=xyz_file)
    rmsd = mol.init_rmsf(atom_subset=atoms)
    rmsd.plot()

The atom_subset argument
------------------------

In the above two examples ``atom_subset=None`` was used an optional keyword,
one which allows one to customize for which atoms the RMSD & RMSF should be
calculated and how the results are distributed over the various columns.

There are a total of four different approaches to the ``atom_subset`` argument:

1.  ``atom_subset=None``: Examine all atoms and store the results in a single \
column.

2.  ``atom_subset=int``: Examine a single atom, based on its index, and store \
the results in a single column.

3.  ``atom_subset=str`` or ``atom_subset=list(int)``: Examine multiple atoms, \
based on their atom type or indices, and store the results in a single column.

4.  ``atom_subset=tuple(str)`` or ``atom_subset=tuple(list(int))``: Examine \
multiple atoms, based on their atom types or indices, and store the results \
in multiple columns. A column is created for each string or nested list \
in ``atoms``.

It should be noted that lists and/or tuples can be interchanged for any other \
iterable container (*e.g.* a Numpy_ array), as long as the iterables elements \
can be accessed by their index.

API
---

.. automethod:: FOX.classes.multi_mol.MultiMolecule.init_rmsd
    :noindex:

.. automethod:: FOX.classes.multi_mol.MultiMolecule.init_rmsf
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
