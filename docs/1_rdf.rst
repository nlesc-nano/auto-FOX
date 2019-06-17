.. _RDF:

Radial & Angular Distribution Function
======================================

Radial and angular distribution function (RDF & ADF) generators have been
implemented in the :class:`.MultiMolecule` class.
The radial distribution function, or pair correlation function, describes how
the particale density in a system varies as a function of distance from a
reference particle. The herein implemented function is designed for
constructing RDFs between all possible (user-defined) atom-pairs.

.. math::

    g(r) =
    \frac{V}{N_a*N_b} \sum_{i=1}^{N_a} \sum_{j=1}^{N_b} \left< *placeholder* \right>


Given a trajectory, ``mol``, stored as a :class:`.MultiMolecule` instance, the RDF can
be calculated with the following
command: ``rdf = mol.init_rdf(atom_subset=None, low_mem=False)``.
The resulting ``rdf`` is a Pandas_ dataframe, an object which is effectively a
hybrid between a dictionary and a NumPy_ array.

A slower, but more memory efficient, method of RDF construction can be enabled
with ``low_mem=True``, causing the script to only store the distance matrix
of a single molecule in memory at once. If ``low_mem=False``, all distance
matrices are stored in memory simultaneously, speeding up the calculation
but also introducing an additional linear scaling of memory with respect to
the number of molecules.
Note: Due to larger size of angle matrices it is recommended to use
``low_mem=False`` when generating ADFs.

Below is an example RDF of a CdSe quantum dot pacified with formate ligands.
The RDF is printed for all possible combinations of cadmium, selenium and
oxygen (Cd_Cd, Cd_Se, Cd_O, Se_Se, Se_O and O_O).

.. code:: python

    >>> from FOX import (MultiMolecule, get_example_xyz)

    >>> example_xyz_file = get_example_xyz()
    >>> mol = MultiMolecule.from_xyz(example_xyz_file)

    >>> rdf = mol.init_rdf(atom_subset=('Cd', 'Se', 'O'))
    >>> adf = mol.init_adf(atom_subset=('Cd', 'Se'))
    >>> rdf.plot(title='RDF')
    >>> adf.plot(title='ADF')


.. plot::

    from FOX import (MultiMolecule, get_example_xyz)
    mol = MultiMolecule.from_xyz(get_example_xyz())
    rdf = mol.init_rdf(atom_subset=('Cd', 'Se', 'O'))
    adf = mol.init_adf(atom_subset=('Cd', 'Se'))
    rdf.plot(title='RDF')
    adf.plot(title='ADF')


API
---

.. automethod:: FOX.classes.multi_mol.MultiMolecule.init_rdf
    :noindex:

.. automethod:: FOX.classes.multi_mol.MultiMolecule.init_adf
    :noindex:


.. _NumPy: https://www.numpy.org/
.. _Pandas: https://pandas.pydata.org/
