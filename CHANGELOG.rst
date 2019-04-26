###########
Change Log
###########

All notable changes to this project will be documented in this file.
This project adheres to `Semantic Versioning <http://semver.org/>`_.

0.3.1
*****

Added
-----
* Added new tests

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

* Introduced an angular distribution generator in the MultiMolecule class

Changed
-------

* Fixed a renormalization bug in the 0.2.1 improved get_rdf() function


0.2.1
*****

Added
-----

* Introduced new FOX.MutliMolecule methods for slicing MD trajectories
* Added the MonteCarlo API to the documentation
* WiP: Split the MonteCarlo class into 2 classes: MonteCarlo & ARMC (subclass)

Changed
-------

* Minor update to copy/deepcopy-related methods
* Improved the get_rdf() function


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

* Empty Python project directory structure
