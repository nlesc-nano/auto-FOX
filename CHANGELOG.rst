###########
Change Log
###########

All notable changes to this project will be documented in this file.
This project adheres to `Semantic Versioning <http://semver.org/>`_.


0.4.3
*****

Added
-----
* Generation of angular distribution functions, :meth:`.MultiMolecule.init_adf`,
  is now conducted in parallel if DASK_ is installed.
* A distance cutoff can now be specified in :meth:`.MultiMolecule.init_adf`.

Changed
-------
* Changed :class:`.PSF` into a dataclass_.

.. _dataclass: https://docs.python.org/3/library/dataclasses.html
.. _DASK: https://dask.org/


0.4.2
*****

Added
-----
* Minimum and maximum allowed values can now be specified for
  all ARMC paramaters.
* Added a commandline interface for generating and exporting
  plots & .csv files.
* Added a function for translating strings to callable objects.

Changed
-------
* Split the armc.job.settings block into .job.md_settings
  & .job.preopt_setting.
* Removed the unused FrozenSettings class.
* Further generalized the param section; a path of keys now has
  to be specified for each block.
* Removed a couple of unused functions.
* Cleaned up the ARMC input parsing; now utilizes `schema <https://pypi.org/project/schema/>`_.
* Updated many docstrings with examples.


0.4.1
*****

Added
-----
* Potential energy surfaces, over the course of last ARMC super-iteration,
  are now stored in .hdf5 format.
* Added increased control over the non-bonded inter-atomic potential.

Changed
-------
* Molecular dynamics (MD) jobs are now preceded by a geometry
  optimization.
* MD simulations can now be skipped of the geometry optimization
  RMSD is too large.
* Docstrings changed to NumPy style.
* Cleaned up the AMRC code.
* Comments in .xyz files are now parsed.


0.4.0
*****

Added
-----
* Added an entry point for accessing :meth:`.ARMC.init_armc`.
* Expanded io-related capabilities, including the option to
  read KF PDB, PSF, PDB and PRM files.

Changed
-------
* Formatting of docstrings in accordance to PEP257_.
* Implementation of type hints.
  Support for python versions prior to 3.7 has been dropped.
* :class:`.ARMC` was moved to its own seperate modules.

.. _PEP257: https://www.python.org/dev/peps/pep-0257/


0.3.2
*****

Added
-----
* Simplified the FOX.ARMC input and updated its documentation.
  (see https://github.com/nlesc-nano/auto-FOX/issues/33)
* Added 2 new methods to the FOX.MultiMolecule class for calculating
  average and time-averaged atomic velocities.
* Added 2 modules for handling atomic charges and .prm files.


0.3.1
*****

Added
-----
* Added new tests for the FOX.MultiMolecule class.
  (see https://github.com/nlesc-nano/auto-FOX/issues/18)

Changed
-------
* Minor style changes to the documentation and the .xyz reader.
* The FOX.MultiMolecule has been changed into a np.ndarray subclass.
  (see https://github.com/nlesc-nano/auto-FOX/issues/30)


0.3.0
*****

Added
-----

* Wrapped up implementation of the Monte Carlo forcefield optimizer.
  (see https://github.com/nlesc-nano/auto-FOX/issues/17)


0.2.3
*****

Added
-----

* Introduced two new methods to the FOX.MultiMolecule class for identifying
  shell structures in, *e.g.*, nanocrystals or dissolved solutes.
  (see https://github.com/nlesc-nano/auto-FOX/issues/29)


0.2.2
*****

Added
-----

* Introduced an angular distribution generator in the MultiMolecule class.

Changed
-------

* Fixed a renormalization bug in the 0.2.1 improved get_rdf() function.


0.2.1
*****

Added
-----

* Introduced new FOX.MutliMolecule methods for slicing MD trajectories.
* Added the MonteCarlo API to the documentation.
* WiP: Split the MonteCarlo class into 2 classes: MonteCarlo & ARMC (subclass).

Changed
-------

* Minor update to copy/deepcopy-related methods.
* Improved the get_rdf() function.


0.2.0
*****

Added
-----

* Added a root mean squared displacement generator (RMSD).
* Added a root mean squared fluctuation generator (RMSF).
* Introduced the FOX.MultiMolecule class for handling and storing all atoms,
  bonds and coordinates.


0.1.0
*****

Added
-----

* Added a reader for multi-xyz files.
* Added a radial distribution functions generator (RDF).


[Unreleased]
************

Added
-----

* Empty Python project directory structure.
