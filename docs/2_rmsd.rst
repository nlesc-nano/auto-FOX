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

Given a trajectory, ``mol``, stored as a :class:`FOX.MultiMolecule` instance,
the RMSD can be calculated with the :meth:`FOX.MultiMolecule.init_rmsd`
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

Given a trajectory, ``mol``, stored as a :class:`FOX.MultiMolecule` instance,
the RMSF can be calculated with the :meth:`FOX.MultiMolecule.init_rmsf`
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
.. automethod:: FOX.MultiMolecule.init_rmsd
    :noindex:

.. automethod:: FOX.MultiMolecule.init_rmsf
    :noindex:

.. automethod:: FOX.MultiMolecule.init_shell_search
    :noindex:

.. automethod:: FOX.MultiMolecule.get_at_idx
    :noindex:


.. _NumPy: https://www.numpy.org/
.. _Pandas: https://pandas.pydata.org/
