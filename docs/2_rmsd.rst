.. _RMSD:

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

Given a trajectory, ``mol``, stored as a :class:`.MultiMolecule` instance,
the RMSD can be calculated with the :meth:`.MultiMolecule.init_rmsd`
method using the following command:

.. code:: python

    >>> rmsd = mol.init_rmsd(atom_subset=None)

The resulting ``rmsd`` is a Pandas_ dataframe, an object which is effectively a
hybrid between a dictionary and a NumPy_ array.

Below is an example RMSD of a CdSe quantum dot pacified with formate ligands.
The RMSD is printed for cadmium, selenium and oxygen atoms.

.. code:: python

    >>> from FOX import MultiMolecule, example_xyz

    >>> mol = MultiMolecule.from_xyz(example_xyz)
    >>> rmsd = mol.init_rmsd(atom_subset=('Cd', 'Se', 'O'))
    >>> rmsd.plot(title='RMSD')


.. plot::

    from FOX import MultiMolecule, example_xyz

    mol = MultiMolecule.from_xyz(example_xyz)
    rmsd = mol.init_rmsd(atom_subset=('Cd', 'Se', 'O'))
    rmsd.plot(title='RMSD')


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

Given a trajectory, ``mol``, stored as a :class:`.MultiMolecule` instance,
the RMSF can be calculated with the :meth:`.MultiMolecule.init_rmsf`
method using the following command:

.. code:: python

    >>> rmsd = mol.init_rmsf(atom_subset=None)

The resulting ``rmsf`` is a Pandas_ dataframe, an object which is effectively a
hybrid between a dictionary and a Numpy_ array.

Below is an example RMSF of a CdSe quantum dot pacified with formate ligands.
The RMSF is printed for cadmium, selenium and oxygen atoms.

.. code:: python

    >>> from FOX import MultiMolecule, example_xyz

    >>> mol = MultiMolecule.from_xyz(example_xyz)
    >>> rmsd = mol.init_rmsf(atom_subset=('Cd', 'Se', 'O'))
    >>> rmsd.plot(title='RMSF')


.. plot::

    from FOX import MultiMolecule, example_xyz

    mol = MultiMolecule.from_xyz(example_xyz)
    rmsd = mol.init_rmsf(atom_subset=('Cd', 'Se', 'O'))
    rmsd.plot(title='RMSF')


Discerning shell structures
---------------------------

See the :meth:`.MultiMolecule.init_shell_search` method.

.. code:: python

    >>> from FOX import MultiMolecule, example_xyz
    >>> import matplotlib.pyplot as plt

    >>> mol = MultiMolecule.from_xyz(example_xyz)
    >>> rmsf, rmsf_idx, rdf = mol.init_shell_search(atom_subset=('Cd', 'Se'))

    >>> fig, (ax, ax2) = plt.subplots(ncols=2)
    >>> rmsf.plot(ax=ax, title='Modified RMSF')
    >>> rdf.plot(ax=ax2, title='Modified RDF')
    >>> plt.show()


.. plot::

    from FOX import MultiMolecule, example_xyz
    import matplotlib.pyplot as plt

    mol = MultiMolecule.from_xyz(example_xyz)
    rmsf, rmsf_idx, rdf = mol.init_shell_search(atom_subset=('Cd', 'Se'))

    fig, (ax, ax2) = plt.subplots(ncols=2)
    rmsf.plot(ax=ax, title='Modified RMSF')
    rdf.plot(ax=ax2, title='Modified RDF')
    plt.show()

The results above can be utilized for discerning shell structures in, *e.g.*,
nanocrystals or dissolved solutes, the RDF minima representing transitions
between different shells.

* There are clear minima for *Se* at ~ 2.0, 5.2, 7.0 & 8.5 Angstrom
* There are clear minima for *Cd* at ~ 4.0, 6.0 & 8.2 Angstrom

With the :meth:`.MultiMolecule.get_at_idx` method it is process the results of
:meth:`.MultiMolecule.init_shell_search`, allowing you to create slices of
atomic indices based on aforementioned distance ranges.

.. code:: python

    >>> dist_dict = {}
    >>> dist_dict['Se'] = [2.0, 5.2, 7.0, 8.5]
    >>> dist_dict['Cd'] = [4.0, 6.0, 8.2]
    >>> idx_dict = mol.get_at_idx(rmsf, rmsf_idx, dist_dict)

    >>> print(idx_dict)
    {'Se_1': [27],
     'Se_2': [10, 11, 14, 22, 23, 26, 28, 31, 32, 40, 43, 44],
     'Se_3': [7, 13, 15, 39, 41, 47],
     'Se_4': [1, 3, 4, 6, 8, 9, 12, 16, 17, 19, 21, 24, 30, 33, 35, 37, 38, 42, 45, 46, 48, 50, 51, 53],
     'Se_5': [0, 2, 5, 18, 20, 25, 29, 34, 36, 49, 52, 54],
     'Cd_1': [25, 26, 30, 46],
     'Cd_2': [10, 13, 14, 22, 29, 31, 41, 42, 45, 47, 50, 51],
     'Cd_3': [3, 7, 8, 9, 11, 12, 15, 16, 17, 18, 21, 23, 24, 27, 34, 35, 38, 40, 43, 49, 52, 54, 58, 59, 60, 62, 63, 66],
     'Cd_4': [0, 1, 2, 4, 5, 6, 19, 20, 28, 32, 33, 36, 37, 39, 44, 48, 53, 55, 56, 57, 61, 64, 65, 67]
     }

It is even possible to use this dictionary with atom names & indices for
renaming atoms in a :class:`.MultiMolecule` instance:

.. code:: python

    >>> print(list(mol.atoms))
    ['Cd', 'Se', 'C', 'H', 'O']

    >>> del mol.atoms['Cd']
    >>> del mol.atoms['Se']
    >>> mol.atoms.update(idx_dict)
    >>> print(list(mol.atoms))
    ['C', 'H', 'O', 'Se_1', 'Se_2', 'Se_3', 'Se_4', 'Se_5', 'Cd_1', 'Cd_2', 'Cd_3']


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

.. automethod:: FOX.classes.multi_mol.MultiMolecule.init_shell_search
    :noindex:

.. automethod:: FOX.classes.multi_mol.MultiMolecule.get_at_idx
    :noindex:


.. _NumPy: https://www.numpy.org/
.. _Pandas: https://pandas.pydata.org/
