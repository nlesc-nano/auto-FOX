###########
Change Log
###########

All notable changes to this project will be documented in this file.
This project adheres to `Semantic Versioning <http://semver.org/>`_.


1.0.0
*****
* *Placeholder*.


0.10.2
******
* Fix the atom-pairs not being sorted if specific parameters are guessed.
* Add a workflow for computing Debye scattering.
* Fix various PES-averaged ARMC issues.
* Fix improper dihedral-indices being incorrectly sorted.
* Make the .xyz parser more robust.
* Export warnings to the logger when running ARMC.
* Fix a Boost issues with the PLAMS master branch.


0.10.1
******
* Avoid `schema` 0.7.5.


0.10.0
******
* Add a new ``MultiMolecule`` method for constructing supercells.
* Add a timestep parameter to ``MultiMolecule.init_power_spectrum``.
* Fixed an issue wherein ``init_power_spectrum`` could raise when the atom subset is specified.
* Raise a ``ValueError`` if an atom type with multiple charges is found.
* Fixed an issue wherein net charges weren't properly conserved in PES-averaged ARMC.
* Allow users to pass custom error functions.
* Make rdkit an optional dependency.
* Fixed an issue wherein auto-FOX would not check the status of jobs.
* Add the new atom_pairs keyword to init_rdf and init_adf.
* Round the net charge to the nearest integer.


0.9.1
*****
* Added the new ``segment_dict`` parameter to the ``PSFContainer.generate_x()`` functions.
* Added the new ``PSFContainer.sort_values`` method.
* Store metadata and net charges for each individual system in PES-averaged ARMC and ARMCPT.
* Fixed an issue wherein frozen atomic charges were ignored when not explicitly specified.
* Fixed in issue wherein PSFContainer.sort_values did not not update the residue ID.
* Fixed an issue wherein guessed parameters could overwrite ones that were explicitly specified.
* Fixed an issue wherein frozen parameters weren't properly sorted.
* Fixed an issue wherein the mean pair-density was computed using lattice vectors in non-periodic calculations.


0.9.0
*****
* Fixed an issue wherein `PSFContainer.to_atom_alias_dict` would assign incorrect indices to atom aliases.
* Fixed an issue wherein treating an entire ARMC parameter block as frozen could raise.
* Added the `FOX.MultiMolecule.lattice` property for storing the lattice vectors of periodic systems.
* Added `FOX.io.lattice_from_cell`, a function for reading lattice vectors from CP2K .cell files.
* Added support for periodic ADF and RDF calculations.
* Added the new `job.lattice` keyword to the ARMC workflow for specifying lattice vectors of the reference system.
* Lattice vectors are now preserved when interconverting between PLAMS, ASE and Auto-FOX.
* Read the pressure from the CP2K .out file when calculating the bulk modulus, rather than calculating it from scratch.


0.8.12
******
* Added a recipe for calculating the similarity between 2 MD trajectories.
* Added improved support for custom atom types.
* Made the ligand optional in the ``FOX.recipes.generate_psf()`` recipe.
* Fixed an issue where non-absolute distances were used when calculating the pressure.
* Fixed an issue where non-charge parameters were updated incorrectly.
* Fixed an issue where parameter guessing could fail if no ``.prm`` file was provided.


0.8.11
******
* Fixed an issue where the .hdf5 status would not be properly cleared (if necessary).
* Grab the cell parameters from previous jobs if applicable.
* More thouroughly check the ``qmflows.Result`` status before accessing it.
* Automatically determine appropiate chunk sizes when calculating the RDF.
* Added a new sub-module dedicated to the calculation of properties: ``FOX.properties``.
* Fixed an issue where parsing the ``unit`` would fail when parameter guessing.
* Fixed an issue where writing ``nan`` to ``armc.xyz.hdf5`` would fail.


0.8.10
******
* Reorganized the dataframes in ``FOX.armc.ParamMapping``.
* Added the ``pes_validation`` keyword. Used for constructing a set
  of PES descriptors for the purpose of validation.
* Dissallow the specification of duplicate .yaml keys.
* Allow users to provide unique settings for each molecule in PES-averaged ARMC.
* Added tests using CP2K, python 3.9, pre-releases and the minimum supported version
  of dependencies.
* Removed the plotting and .csv-related entry points.
* Added the ``param.validation.enforce_constraints`` keyword; used for checking
  whether or not initially supplied parameters satisfy all user-provided constraints.
* Added the ``unit`` field to the ``param_metadata`` dataset.
* Added the new ``pes.block.ref`` keyword; used for constructing PES descriptors
  from ``qmflows.Result`` objects rather than the ``FOX.MultiMolecule`` instances
  constructed therefrom.


0.8.9
*****
* Fixed an issue where frozen parameters were not respected when performing
  contrained parameter updates.
* Fixed an issue where the ARMCPT parameters weren't properly swapped.
* Added two new ARMC options: ``param.validation.allow_non_existent``
  and ``param.validation.charge_tolerance``.
* Update the type hints within Auto-FOX.
* Allow the ARMC(PT) input to, once again, be dumped to a .yaml file via
  the ``armc2yaml`` command.
* Dump more information into the ARMC logger.
* Re-enable the ARMC restart option.


0.8.8
*****
* Added recipes for calculating time-resolved angular/radial distribution functions.
* Various documentation-related updates.
* The order in wich atom-pair/-triplet based parameters are provided is now irrelevant.
  For example ``Cd Se`` and ``Se Cd`` are now treated as equivalent, as well as
  ``O C H`` and ``H C O`` (but not ``O H C``).
* Fixed an issue where guessed parameters were not properly parsed.
* Relaxed the PLAMS version requirement.
* Log all local variables whenever an exception is encountered.
* Move the ARMC test files to their own repo.
* Export parameter metadata to the .hdf5 file.


0.8.7
*****
* Moved from ``PRMContainer.__dict__`` to a ``PRMContainer.__slots__`` based class structure.
* Cleaned up the ``PRMContainer`` code; updated annotations, *etc.*.
* Removed ``assertionlib.AbstractDataClass`` as base class from ``PRMContainer``.
* Do not read or write comments to and from a .prm file.
* Upped the minimum Sphinx version to ``2.1``.
* Removed ``sphinx-autodoc-typehints``.


0.8.6
*****
* Import ``AbstractFileContainer`` from Nano-Utils.
* Removed ``TypeMapping`` in favor of `TypedDict`.
* Remove travis in favor of GitHub Actions.


0.8.5
*****
* Moved a number of functions to the `Nano-Utils <https://github.com/nlesc-nano/Nano-Utils>`_ Package.


0.8.4
*****
* Updated the ARMC documentation.


0.8.3
*****
* Updated the ARMC tests.
* Renamed ``FOX.test_utils`` to ``FOX.testing_utils``.
* Added flake8 and pydocstyle to the tests.


0.8.2
*****
* Fixed and generalized the frocefield parameter guessing procedures
  (https://github.com/nlesc-nano/auto-FOX/issues/100 and https://github.com/nlesc-nano/auto-FOX/pull/112).
* Log the optimum ARMC cycle in ``get_best()`` and ``overlay_descriptor()``
  (https://github.com/nlesc-nano/auto-FOX/pull/111).
* Fixed an issue where certain ADF atom-subset-permutations were ignored
  (https://github.com/nlesc-nano/auto-FOX/pull/110).
* Aux error: Ensure that the summation over ``qm`` occurs row-wise
  (https://github.com/nlesc-nano/auto-FOX/pull/108).


0.8.1
*****
* WiP: Introduction of the ``ARMCPT`` class.


0.8.0
*****
* Move all ARMC related modules to the new ``FOX.armc`` module.
* Switched from ``plams.Job`` to ``qmflows.Package`` runners.
* Introduced the ``PhiUpdater`` class for handling and updating the ``phi`` parameter.
* Introduced the ``ParamMapping`` class for handling and updating the forcefield parameters.
* Introduced the ``PackageManager`` class for handling the and managing the ``qmflows.Package``
  instances, including the running of jobs.
* Store the Auto-FOX ``__version__`` in the .hdf5 file.
* Changed the .yaml input to closer resemble the actual class structure.
* Overhauled the .yaml input parsing.
* Bumped the minimum Python version to 3.7.
* Marked Auto-FOX as a typed package.
* Added ``qmflows`` and ``noodles`` as new dependencies.
* Added ``typing_extensions`` as a new dependency for Python < 3.8.


0.7.4
*****
* Increased the assertionlib version requirement to >= v2.1.
* Generalized ``PRMContainer.overlay_cp2k_settings()`` to work for all
  forcefield parameters (potentially) specified in CP2K settings.
* Added the ``PRMContainer.overlay_mapping()`` function for overlaying
  an arbitrary (nested) Mapping with the ``PSFContainer``.
* Added the ``TypedMapping`` class, a baseclass for creating typed ``Mappings``,
  *i.e.* Mappings with a set number of specific keys.
* Exchange a number of ``Settings.__contains__()`` operations for ``Settings.get()``.


0.7.3
*****
* Updated the default CP2K Settings template.
* Employ more rigorous index-sorting when creating the .psf bonds,
  angles, dihedrals and impropers sections.
* Fixed a bug where values reported by ``degree_of_separation()`` were
  incorrectly ordered when ``dtype != bool``.
* Added the ``shift_cutoff`` keyword for the calculation of all forcefield non-bonded potential energies.
  Sets the value of the potentials to zero at the specified ``distance_upper_bound``.
* Fixed an issue where sigma-values produced by .prm files were not properly parsed.
* Fixed an issue where multiple potentials for a single set of dihedrals were not properly parsed.
* Further miscellaneous improvements and fixes to the ``FOX.ff`` modules.


0.7.2
*****
* All forcefield related energies are now returned in their entirety,
  rather than averaging them with respect to the number of MD iterations.
* Add recipes for the analyses of forcefield energies.
* Increased the flexibility of the ``recipes.plot_descriptor()`` function.
* https://github.com/nlesc-nano/auto-FOX/pull/77 & https://github.com/nlesc-nano/auto-FOX/pull/78:
  Combine NumPy vectorization with ``for``-loops during the calculation of inter-/intra-ligand
  non-bonded interactions if array sizes start to exceed 100 million elements.
* https://github.com/nlesc-nano/auto-FOX/pull/78:
  Truncated distance matrices can now be used for the calculation of inter-ligand
  non-bonded interactions.


0.7.1
*****
* Renamed the ``csv`` module to ``csv_utils``.
* Fixed the previously broken ``MultiMolecule.delete_atoms()`` method.
* Ensure that ``MultiMolecule._get_atom_subset()`` can handle all Iterables.
* When assigning new bonds (``MultiMolecule.bonds``) all bond orders will default
  to ``1`` if not explicitly specified.
* Cleaned up the ``LJDataFrame()`` class.
* Implemented multiple bugfixes related to the calculation of intra-moleculair
  non-bonded interactions.
* Lennard-Jones and Electrostatic scaling factors can now be applied for the
  calculation of 1,4 non-bonded interactions, similiar to the CP2K EI_SCALE14_
  and VDW_SCALE14_ keywords.
* Ensure that the ``IMPROPER`` / ``IMPROPERS`` .prm block is always written as
  ``IMPROPER``.
  While both of them are in principle valid block-names, CP2K will only accept ``IMPROPER``.
* Introduced code style changes to the ``recipes.psf`` module.

.. _EI_SCALE14: https://manual.cp2k.org/cp2k-2_3-branch/CP2K_INPUT/FORCE_EVAL/MM/FORCEFIELD.html#list_EI_SCALE14
.. _VDW_SCALE14: https://manual.cp2k.org/cp2k-2_3-branch/CP2K_INPUT/FORCE_EVAL/MM/FORCEFIELD.html#list_VDW_SCALE14


0.7.0
*****
* Multiple updates to the ``FOX.ff`` modules:
* Fixed a missing ``+1`` addition in the calculation of the dihedral potential.
* Wildcard atoms (``"X"``) are now properly parsed.
* 1,4-nonbonded interactions (intra-moleculair) are now calculated.
* 1,3-nonbonded interactions (intra-moleculair), aka the Urey-Bradley terms, are now calculated.
* Non-bonded interactions between explicitly specified atom-pairs are now calculated.
* Fixed a number of issues introduced in https://github.com/nlesc-nano/auto-FOX/pull/74.


0.6.21
******
* Fixed an issue where a ``MultiMolecule()`` couldn't be converted into a ``Molecule()``.
* Upped the version requirement from the ``assertionlib`` package to >= 2.


0.6.20
******
* Cleaned up how PES descriptors are generated & stored in the ``ARMC()`` class.
* Atom names specified in .PSF files are now accessible by ``MultiMolecule()`` instances
  during the ARMC procedure.
* Generalized ``dekekulize()`` to work for all non-integer bond orders; not just ``1.5``.


0.6.19
******
* Cleaned up the ``PRMContainer()`` class.
* Cleaned up the main __init__.py file.
* https://github.com/nlesc-nano/auto-FOX/commit/b583af768b047c70565d9ed3fabfc091c94debf0:
  Increased the flexibility of ``MultiMolecule.get_pair_dict()``.


0.6.18
******
* Added the ``MultiMolecule.add_atoms()`` method.
* Added a new recipe (``FOX.recipes.ligands``) for generating radial distribution functions
  using the center of mass of ligands (`doc <https://auto-fox.readthedocs.io/en/latest/7_recipes.html#fox-recipes-ligands>`_).


0.6.17
******
* The total error (not just the error change) is now printed in the ARMC log.
* Added a new example to the param recipes for slicing DataFrames.
* Added a new workflow for creating .psf files for quantum dots with multiple different ligands.
* https://github.com/nlesc-nano/auto-FOX/commit/28abcb10726069ca8d6eda4cd747630f5d8a0442 :
  Ensure that ARMC jobs without .psf file do not crash.
* https://github.com/nlesc-nano/auto-FOX/commit/7a9f313be3f4deef2449394dae0b5b3bea013288 :
  Added the ``mol_subset`` keyword to ``MultiMolecule.init_rdf()``.
* https://github.com/nlesc-nano/auto-FOX/commit/a5ab4bfc3f21e5795cf5c80e81aae7abdb8bf030 &
  https://github.com/nlesc-nano/auto-FOX/commit/ed5acd504963c4511a2d75c23d970636e51e60f6 :
  Fixed a number of issues regarding AMRC input parsing.
* https://github.com/nlesc-nano/auto-FOX/commit/c5b38c6dddac70523b73e1019a203345bfe4b1c7 :
  Fixed an issue where ``assign_constraints()`` failed to parse ``"=="`` characters.


0.6.16
******
* There is no v0.6.16.


0.6.15
******
* Added recipes for generating .psf files in ``FOX.recipes``.
* https://github.com/nlesc-nano/auto-FOX/pull/65 : Fixed a bug where ARMC parameter constraints
  were not properly parsed.
* https://github.com/nlesc-nano/auto-FOX/pull/66 : Added new ARMC tests.


0.6.14
******
* Fixed an issue where valid .xyz files were not properly read during the ARMC procedure.
* Added a precaution against reading faulty .xyz files.
* Fixed an issue where some of datasets in the armc.xyz.hdf5 file were of incorrect shape.
* Change the datatype from the armc.xyz.hdf5's datasets from ``np.float64`` to ``np.float16``
  in order to reduce disk space.
* Added a precaution against reading faulty .xyz files.
* https://github.com/nlesc-nano/auto-FOX/pull/60 : .hdf5 files are now forcibly closed (if necessary)
  upon restarting an ARMC procedure.
* https://github.com/nlesc-nano/auto-FOX/pull/61 : Updated the recipe examples;
  ``plot_descriptor()`` no longer crashes when encountering a ``DataFrame()`` with a single column.
* https://github.com/nlesc-nano/auto-FOX/pull/62 & https://github.com/nlesc-nano/auto-FOX/pull/63 :
  Ensure that the ARMC restarting starts from the last iteration whose error is not ``np.nan``.


0.6.13
******
* Introduced a new logger; see https://github.com/nlesc-nano/auto-FOX/issues/33.
* Change the fillvalue of all float-based .hdf5 Datasets to np.nan.
* Atoms and bonds are now, again, properly stored in the .xyz.hdf5 file.


0.6.12
******
* The ARMC input parser no longer expects ``ARMC.param`` and the .psf file(s) to form identical sets.
* All atomic charges in the ARMC .psf files are now set to 0.0.
  Charges are handled, exclusively, by the cp2k input file.
* Fixed an issue where atom-types were not properly updated in the .psf file.
* Fixed an issue where the ARMC .xyz.hdf5 file was not properly updated.
* Ensure that ``None`` object encountered during the ARMC procedure are always converted
  into ``np.nan``.
  Contrary to NumPy or Pandas, h5py will *not* automatically convert ``None`` to ``np.nan``
  when assigning items to a Dataset.
* Raise a ``RuntimeError`` if a job hard-crashes in the first ARMC iteration.
* Always create a shallow copy of (to-be mutated) input parameters when
  calculating (forcefield-based) interactions.
* Fixed the atom-pair hashing in ``get_bonded()``.
* Prevent double counting non-bonded interactions when i == j in ``get_intra_non_bonded()``.
* Potentials are now (properly) averaged over all molecules within an MD trajectory in ``get_intra_non_bonded()``.
* Import scipy's ``fftconvolve()`` with a try/except approach; importing has a tendancy of raising RecursionErrors.
* Log the super- & sub-iteration upon ``ARMC()`` restarts.


0.6.11
******
* .psf files can now be directly supplied in the ARMC .yaml input.

From https://github.com/nlesc-nano/auto-FOX/issues/52:

* Added the option to estimate non-bonded parameters using either UFF or the RDF.
* ``ARMC()`` instances can now be converted into ``ARMC.from_yaml()``-compatible .yaml files.
  See the ``armc2yaml`` entry point.


0.6.10
******
* Added the option to provide multiple .rtf files for state-averaged ARMC runs.


0.6.9
*****
* ``FOX.get_example_xyz()`` has been deprecated in favor of ``FOX.example_xyz``.
* Moved the ``psf_to_atom_dict()`` function to ``PSFContainer.to_atom_dict()``.

From https://github.com/nlesc-nano/auto-FOX/issues/52:

* Repos of script to analyze AMRC data.
* Simultaneous fitting of different trajectories with different atom types;
  ensure that the PES descriptor generators can have different arguments for each trajectory.
* Restart procedure for ARMC.


0.6.8
*****
* Added a workflow for calculating covalent intra-ligand interactions using
  harmonic- + cosine-based potentials: ``FOX.get_bonded()``.
  Complementary to the in 0.6.4 introduced ``FOX.get_non_bonded()``.
* Added a workflow for calculating non-covalent intera-ligand interactions
  using electrostatic + Lennard-Jones potentials: ``FOX.get_intra_non_bonded()``.
  Complementary to the in 0.6.4 introduced ``FOX.get_non_bonded()``.
* Added a number of useful workflows as stand-alone scripts.
* Added the ``FOX.ff`` directory for all forcefield related modules.
* Slimmed down the number of exposed functions and classess.
* Changed ``PSFContainer._SHAPE_DICT`` and ``._HEADER_DICT`` to instances of ``MappingProxyType()``.
* Fixed a bug where some ``PSFContainer()`` dihedral angles where ordered incorectly.


0.6.7
*****
* ``FOX.estimate_lj()`` can now estimate sigma based on either the base or
  the inflection point of the first RDF peak.


0.6.6
*****
* Made Auto-FOX compatible with Python 3.6.
* Added tests for Python 3.6 and 3.8.
* Permanently moved a number of modules from (nano-)CAT to Auto-FOX.
* Added the ``MutliMolecule.loc`` property; allowing for the slicing of
  MultiMolecule (directly) using atomic symbols.
  Usage examples: ``mol.loc['Cd']`` and ``mol.loc['Cd', 'Se', 'O']``.
  The Equivalent to ``mol[mol.atoms['Cd']]``.


0.6.5
*****
* Cleaned up the angular distribution code & atom subset code.
* Added a module for constructing UFF Lennard-Jones parameters.
* Added the option to specify constant parameter values in the ARMC .yaml input.


0.6.4
*****
* Updated the ``read_prm`` module.
* Added a workflow for calculating inter-ligand and core/ligand interactions
  using electrostatic + Lennard-Jones potentials: ``FOX.get_non_bonded()``.


0.6.3
*****
* Added a function, ``FOX.estimate_lj()``, for estimating Lennard-Jones
  parameters using radial dsitribution functions.


0.6.2
*****
* Added the option to read ligand parameters from .rtf files produced by MATCH_.
  Serves as an alternative for cgenff's .str files.
* Fixed a missing key for MD pre-optimizations: https://github.com/nlesc-nano/auto-FOX/commit/08b9e3224965a359de8471b9976d2343db96f9de.

.. _MATCH: http://brooks.chem.lsa.umich.edu/index.php?page=match&subdir=articles/resources/software


0.6.1
*****
* Added an additionl memory consumption level to `MultiMolecule.init_rdf()`.
* Ensure that the 'constraints' column is always present in the ARMC parameter DataFrame.
* ``_xyz_to_hdf5()`` no longer crashes when ``mol_list=None``.
* Switched the `AssertionLib` package from GitHub to PyPi.


0.6.0
*****
* Many minor (consistancy) changes and codestyle improvements.
* Ported a number of classes from (nano-)CAT to Auto-FOX (``FrozenSettings`` & ``PSFContainer``).
* Reduced te number of parameters for the ``ARMC()`` and ``MonteCarlo()`` classes.
* Added the ``run_armc()`` method for handling all `JobManager` related ARMC tasks.
* Added the AssertionLib package as dependancy.
* Moved ``FOX.classes.molecule_utils`` to ``FOX.functions.molecule_utils`` in favor of a function-based approach.
* Improved the speed of `read_multi_xyz()` by roughly 10%.
* Generalized the ARMC constraints system.
* Fixed the PLAMS branch: see https://github.com/nlesc-nano/auto-FOX/commit/8a1d13b8d5e2f2a2b635ade965a1eb58488ecd2a and
  https://github.com/nlesc-nano/auto-FOX/commit/2916c937689f7d9a9439ba7cd1cce4d2add989cf.


0.5.0
*****

Added
-----

* Added the option for state-averaged ARMC parameter optimizations;
  *i.e.* simultaneously optimizing a single parameter set based on the
  auxiliary error of multiple MD trajectories.
* [reprlib](https://docs.python.org/3/library/reprlib.html) is now used
  in ``MultiMolecule.__str__()``.

Changed
-------

* Updated all module-level docstrings.
  Now includes an autosummarry_ and autodoc_ description of the module.

.. _autosummarry: https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html
.. _autodoc: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html


0.4.4
*****

Added
-----

* Added new methods for constructing the velocity autocorrelation function
  (VACF), :meth:`.MultiMolecule.get_vacf`, and VACF-derived power spectra,
  :meth:`.MultiMolecule.init_power_spectrum`.


0.4.3
*****

Added
-----
* Generation of angular distribution functions,
  :meth:`.MultiMolecule.init_adf`, is now conducted in parallel
  if DASK_ is installed.
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
