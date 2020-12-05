.. _Monte Carlo Parameters:

Monte Carlo Parameters
======================

Index
~~~~~
.. table::
    :width: 100%
    :widths: 30 70

    =========================================== =========================================================================================================
    param_                                      Description
    =========================================== =========================================================================================================
    :attr:`param.type`                          The type of parameter mapping.
    :attr:`param.move_range`                    The parameter move range.
    :attr:`param.func`                          The callable for performing the Monte Carlo moves.
    :attr:`param.kwargs`                        A dictionary with keyword arguments for :attr:`param.func`.
    :attr:`param.validation.allow_non_existent` Whether to allow parameters, that are explicitly specified, for absent atoms.
    :attr:`param.validation.charge_tolerance`   Check whether the net charge of the system is integer within a given tolerance.
    :attr:`param.block.param`                   The name of the forcefield parameter.
    :attr:`param.block.unit`                    The unit in which the forcefield parameters are expressed.
    :attr:`param.block.constraints`             A string or list of strings with parameter constraints.
    :attr:`param.block.guess`                   Estimate all non-specified forcefield parameters.
    :attr:`param.block.frozen`                  A sub-block with to-be frozen parameters.
    =========================================== =========================================================================================================

.. table::
    :width: 100%
    :widths: 30 70

    =========================================== =========================================================================================================
    psf_                                        Description
    =========================================== =========================================================================================================
    :attr:`psf.str_file`                        The path+filename to one or more stream file.
    :attr:`psf.rtf_file`                        The path+filename to one or more MATCH-produced rtf file.
    :attr:`psf.psf_file`                        The path+filename to one or more psf files.
    :attr:`psf.ligand_atoms`                    All atoms within a ligand.
    =========================================== =========================================================================================================

.. table::
    :width: 100%
    :widths: 30 70

    =========================================== =========================================================================================================
    pes_                                        Description
    =========================================== =========================================================================================================
    :attr:`pes.block.func`                      The callable for performing the Monte Carlo moves.
    :attr:`pes.block.kwargs`                    A dictionary with keyword arguments for :attr:`pes.block.func`.
    =========================================== =========================================================================================================

.. table::
    :width: 100%
    :widths: 30 70

    =========================================== =========================================================================================================
    pes_validation_                             Description
    =========================================== =========================================================================================================
    :attr:`pes_validation.block.func`           The callable for performing the Monte Carlo validation.
    :attr:`pes_validation.block.kwargs`         A dictionary with keyword arguments for :attr:`pes_validation.block.func`.
    =========================================== =========================================================================================================

.. table::
    :width: 100%
    :widths: 30 70

    =========================================== =========================================================================================================
    job_                                        Description
    =========================================== =========================================================================================================
    :attr:`job.type`                            The type of package manager.
    :attr:`job.molecule`                        One or more .xyz files with reference (QM) potential energy surfaces.
    :attr:`job.block.type`                      An instance of a QMFlows :class:`~qmflows.packages.packages.Package`.
    :attr:`job.block.settings`                  The job settings as used by :class:`job.block.type`.
    :attr:`job.block.template`                  A settings template for updating :class:`job.block.settings`.
    =========================================== =========================================================================================================

.. table::
    :width: 100%
    :widths: 30 70

    =========================================== =========================================================================================================
    monte_carlo_                                Description
    =========================================== =========================================================================================================
    :attr:`monte_carlo.type`                    The type of Monte Carlo procedure.
    :attr:`monte_carlo.iter_len`                The total number of ARMC iterations :math:`\kappa \omega`.
    :attr:`monte_carlo.sub_iter_len`            The length of each ARMC subiteration :math:`\omega`.
    :attr:`monte_carlo.logfile`                 The name of the ARMC logfile.
    :attr:`monte_carlo.hdf5_file`               The name of the ARMC .hdf5 file.
    :attr:`monte_carlo.path`                    The path to the ARMC working directory.
    :attr:`monte_carlo.folder`                  The name of the ARMC working directory.
    :attr:`monte_carlo.keep_files`              Whether to keep *all* raw output files or not.
    =========================================== =========================================================================================================

.. table::
    :width: 100%
    :widths: 30 70

    =========================================== =========================================================================================================
    phi_                                        Description
    =========================================== =========================================================================================================
    :attr:`phi.type`                            The type of phi updater.
    :attr:`phi.gamma`                           The constant :math:`\gamma`.
    :attr:`phi.a_target`                        The target acceptance rate :math:`\alpha_{t}`.
    :attr:`phi.phi`                             The initial value of the variable :math:`\phi`.
    :attr:`phi.func`                            The callable for updating :attr:`phi<phi.phi>`.
    :attr:`phi.kwargs`                          A dictionary with keyword arguments for :attr:`phi.func`.
    =========================================== =========================================================================================================


param
~~~~~
All forcefield-parameter related options.

This settings block accepts an arbitrary number of sub-blocks.

.. admonition:: Examples

    .. code:: yaml

        param:
            type: FOX.armc.ParamMapping
            move_range:
                start: 0.005
                stop: 0.1
                step: 0.005
                ratio: null
            func: numpy.multiply
            kwargs: {}
            validation:
                allow_non_existent: False
                charge_tolerance: 0.01

            charge:
                param: charge
                constraints:
                    - '0.5 < Cd < 1.5'
                    - '-0.5 > Se > -1.5'
                Cd: 0.9768
                Se: -0.9768
                O_1: -0.47041
                frozen:
                    C_1: 0.4524
            lennard_jones:
                -   unit: kjmol
                    param: epsilon
                    Cd Cd: 0.3101
                    Se Se: 0.4266
                    Cd Se: 1.5225
                    frozen:
                        guess: uff
                -   unit: nm
                    param: sigma
                    Cd Cd: 0.1234
                    Se Se: 0.4852
                    Cd Se: 0.2940
                    frozen:
                        guess: uff

|

    .. attribute:: param.type

        :Parameter:     * **Type** - :class:`str` or :class:`FOX.armc.ParamMappingABC<FOX.armc.param_mapping.ParamMappingABC>` subclass
                        * **Default Value** - ``"FOX.armc.ParamMapping"``

        The type of parameter mapping.

        Used for storing and moving user-specified forcefield values.

        .. admonition:: See Also

            :class:`FOX.armc.ParamMapping<FOX.armc.param_mapping.ParamMapping>`
                A **ParamMappingABC** subclass.


    .. attribute:: param.move_range

        :Parameter:     * **Type** - array-like or :class:`dict`
                        * **Default Value** - ``{"start": 0.005, "stop": 0.1, "step": 0.005, "ratio": None}``

        The parameter move range.

        This value accepts one of the following two types of inputs:

        1. A list of allowed moves (*e.g.* :code:`[0.9, 0.95, 1.05, 1.0]`).
        2. A dictionary with the ``"start"``, ``"stop"`` and ``"step"`` keys.
            For example, the list in 1. can be reproduced with
            ``{"start": 0.05, "stop": 0.1, "step": 0.05, "ratio": None}``.

        When running the ARMC parallel procedure (:attr:`monte_carlo.type = FOX.armc.ARMCPT<monte_carlo.type>`)
        option 1. should be supplied as a nested list (*e.g.* :code:`[[0.9, 0.95, 1.05, 1.0], [0.8, 0.9, 1.1, 1.2]]`)
        and option 2. requires the additional :code:`"ratio"` keyword (*e.g.* :code:`[1, 2]`).


    .. attribute:: param.func

        :Parameter:     * **Type** - :class:`str` or :class:`~collections.abc.Callable`
                        * **Default Value** - ``"numpy.multiply"``

        The callable for performing the Monte Carlo moves.

        The passed callable should be able to take two NumPy arrays as a arguments and return

        .. admonition:: See Also

            :func:`numpy.multiply`
                Multiply arguments element-wise.


    .. attribute:: param.kwargs

        :Parameter:     * **Type** - :class:`dict`
                        * **Default Value** - ``{}``

        A dictionary with keyword arguments for :attr:`param.func`.


    .. attribute:: param.validation.allow_non_existent

        :Parameter:     * **Type** - :class:`bool`
                        * **Default Value** - :data:`False`

        Whether to allow parameters, that are explicitly specified, for absent atoms.

        This check is performed once, before the start of the ARMC procedure.


    .. attribute:: param.validation.charge_tolerance

        :Parameter:     * **Type** - :class:`float`
                        * **Default Value** - ``0.01``

        Check whether the net charge of the system is integer within a given tolerance.

        This check is performed once, before the start of the ARMC procedure.
        Setting this parameter to ``inf`` disables the check.


    .. attribute:: param.block.param

        :Parameter:     * **Type** - :class:`str`

        The name of the forcefield parameter.

        .. important::

            Note that this option has no default value;
            one *must* be provided by the user.


    .. attribute:: param.block.unit

        :Parameter:     * **Type** - :class:`str`

        The unit in which the forcefield parameters are expressed.

        See the `CP2K manual <https://manual.cp2k.org/trunk/units.html>`_ for a
        comprehensive list of all available units.


    .. attribute:: param.block.constraints

        :Parameter:     * **Type** - :class:`str` or :class:`list` [:class:`str`]

        A string or list of strings with parameter constraints.
        Accepted types of constraints are minima/maxima (*e.g.* ``2 > Cd > 0``)
        and fixed parameter ratios (*e.g.* ``Cd == -1 * Se``).
        The special ``$LIGAND`` alias can be used for representing all
        atoms within a single ligand. For example, ``$LIGAND`` is equivalent to
        ``2 * O + C + H`` in the case of formate.


    .. attribute:: param.block.guess

        :Parameter:     * **Type** - :class:`dict`

        Estimate all non-specified forcefield parameters.

        If specified, expects a dictionary with the ``"mode"`` key,
        *e.g.* :code:`{"mode": "uff"}` or :code:`{"mode": "rdf"}`.


    .. attribute:: param.block.frozen

        :Parameter:     * **Type** - :class:`dict`

        A sub-block with to-be frozen parameters.

        Parameters specified herein will be treated as constants rather than variables.
        Accepts forcefield parameters (*e.g.* :code:`"Cd Cd" = 1.0`) and,
        optionally, the :attr:`guess<param.block.guess>` key.


psf
~~~
Settings related to the construction of protein structure files (.psf).

Note that the :attr:`psf.str_file`, :attr:`psf.rtf_file` and
:attr:`psf.psf_file` options are all mutually exclusive;
only one should be specified.
Furthermore, this block is completelly optional.

.. admonition:: Examples

    .. code:: yaml

        psf:
            rtf_file: ligand.rtf
            ligand_atoms: [C, O, H]

|

    .. attribute:: psf.str_file

        :Parameter:     * **Type** - :class:`str` or :class:`list` [:class:`str`]
                        * **Default Value** - :data:`None`

        The path+filename to one or more stream files.

        Used for assigning atom types and charges to ligands.


    .. attribute:: psf.rtf_file

        :Parameter:     * **Type** - :class:`str` or :class:`list` [:class:`str`]
                        * **Default Value** - :data:`None`

        The path+filename to one or more MATCH-produced rtf files.

        Used for assigning atom types and charges to ligands.


    .. attribute:: psf.psf_file

        :Parameter:     * **Type** - :class:`str` or :class:`list` [:class:`str`]
                        * **Default Value** - :data:`None`

        The path+filename to one or more psf files.

        Used for assigning atom types and charges to ligands.


    .. attribute:: psf.ligand_atoms

        :Parameter:     * **Type** - :class:`str` or :class:`list` [:class:`str`]
                        * **Default Value** - :data:`None`

        A list with all atoms within the organic ligands.

        Used for defining residues.


pes
~~~
Settings to the construction of potentialy energy surface (PES) descriptors.

This settings block accepts an arbitrary number of sub-blocks,
each containg the :attr:`func<pes.block.func>` and, optionally,
:attr:`kwargs<pes.block.kwargs>` keys.

.. admonition:: Examples

    .. code:: yaml

        pes:
            rdf:
                func: FOX.MultiMolecule.init_rdf
                kwargs:
                    atom_subset: [Cd, Se, O]
            adf:
                func: FOX.MultiMolecule.init_adf
                kwargs:
                    atom_subset: [Cd, Se]

|

    .. attribute:: pes.block.func

        :Parameter:     * **Type** - :class:`str` or :class:`~collections.abc.Callable`

        A callable for constructing a PES descriptor.

        The callable should take a :class:`~FOX.classes.multi_mol.MultiMolecule` instance
        as sole (positional) argument and return an array-like object.

        .. important::

            Note that this option has no default value;
            one *must* be provided by the user.

        .. admonition:: See Also

            :meth:`FOX.MultiMolecule.init_rdf<FOX.classes.multi_mol.MultiMolecule.init_rdf>`
                Initialize the calculation of radial distribution functions (RDFs).

            :meth:`FOX.MultiMolecule.init_adf<FOX.classes.multi_mol.MultiMolecule.init_adf>`
                Initialize the calculation of angular distribution functions (ADFs).


    .. attribute:: pes.block.kwargs

        :Parameter:     * **Type** - :class:`dict`
                        * **Default Value** - ``{}``

        A dictionary with keyword arguments for :attr:`func<pes.block.func>`.


pes_validation
~~~~~~~~~~~~~~
Settings to the construction of potentialy energy surface (PES) validators.

Functions identically w.r.t. to the pes_ block, the exception being that PES descriptors
calculated herein are do *not* affect the error;
they are only calculated for the purpose of validation.

This settings block accepts an arbitrary number of sub-blocks,
each containg the :attr:`func<pes_validation.block.func>` and, optionally,
:attr:`kwargs<pes_validation.block.kwargs>` keys.

.. admonition:: Examples

    .. code:: yaml

        pes_validation:
            adf:
                func: FOX.MultiMolecule.init_adf
                kwargs:
                    atom_subset: [Cd, Se]
                    mol_subset: !!python/object:builtins.slice  # i.e. slice(None, None, 10)
                    - null
                    - null
                    - 10

|

    .. attribute:: pes_validation.block.func

        :Parameter:     * **Type** - :class:`str` or :class:`~collections.abc.Callable`

        A callable for constructing a PES validators.

        The callable should take a :class:`~FOX.classes.multi_mol.MultiMolecule` instance
        as sole (positional) argument and return an array-like object.

        The structure of this block is identintical to its counterpart in :attr:`pes.block.func`.

        .. important::

            Note that this option has no default value;
            one *must* be provided by the user.

        .. admonition:: See Also

            :meth:`FOX.MultiMolecule.init_rdf<FOX.classes.multi_mol.MultiMolecule.init_rdf>`
                Initialize the calculation of radial distribution functions (RDFs).

            :meth:`FOX.MultiMolecule.init_adf<FOX.classes.multi_mol.MultiMolecule.init_adf>`
                Initialize the calculation of angular distribution functions (ADFs).


    .. attribute:: pes_validation.block.kwargs

        :Parameter:     * **Type** - :class:`dict`
                        * **Default Value** - ``{}``

        A dictionary with keyword arguments for :attr:`func<pes_validation.block.func>`.

        The structure of this block is identintical to its counterpart in :attr:`pes.block.kwargs`.


job
~~~
Settings related to the running of the various molecular mechanics jobs.

In addition to having two constant keys (:attr:`~job.type` and :attr:`~job.molecule`)
this block accepts an arbitrary number of sub-blocks representing quantum and/or classical
mechanical jobs.
In the example above there are two of such sub-blocks: ``geometry_opt`` and ``md``.
The first step consists of a geometry optimization while the second one runs the
actual molecular dynamics calculation.
Note that these jobs are executed in the order as provided by the user-input.

.. admonition:: Examples

    .. code:: yaml

        job:
            type: FOX.armc.PackageManager
            molecule: .../mol.xyz

            geometry_opt:
                type: qmflows.cp2k_mm
                settings:
                    prm: .../ligand.prm
                    cell_parameters: [50, 50, 50]
                template: qmflows.templates.geometry.specific.cp2k_mm
            md:
                type: qmflows.cp2k_mm
                settings:
                    prm: .../ligand.prm
                    cell_parameters: [50, 50, 50]
                template: qmflows.templates.md.specific.cp2k_mm

|

    .. attribute:: job.type

        :Parameter:     * **Type** - :class:`str` or :class:`FOX.armc.PackageManagerABC<FOX.armc.package_manager.PackageManagerABC>` subclass
                        * **Default Value** - ``"FOX.armc.PackageManager"``

        The type of Auto-FOX package manager.

        Used for managing and running the actual jobs.

        .. admonition:: See Also

            :class:`FOX.armc.PackageManager<FOX.armc.package_manager.PackageManager>`
                A **PackageManagerABC** subclass.


    .. attribute:: job.molecule

        :Parameter:     * **Type** - :class:`str` or :class:`list` [:class:`str`]

        One or more .xyz files with reference (QM) potential energy surfaces.


        .. important::

            Note that this option has no default value;
            one *must* be provided by the user.


    .. attribute:: job.block.type

        :Parameter:     * **Type** - :class:`str` or :class:`qmflows.packages.Package<qmflows.packages.packages.Package>` instance
                        * **Default Value** - ``"qmflows.cp2k_mm"``

        An instance of a QMFlows Package.

        .. admonition:: See Also

            :class:`qmflows.cp2k_mm<qmflows.package.cp2k_mm.cp2m_mm>`
                An instance of :class:`~qmflows.packages.cp2k_mm.CP2KMM`.


    .. attribute:: job.block.settings

        :Parameter:     * **Type** - :class:`dict` or :class:`list` [:class:`dict`]
                        * **Default Value** - ``{}``

        The job settings as used by :class:`type<job.block.type>`.

        In the case of PES-averaged ARMC one can supply a list of dictionaries,
        each one representing the settings for its counterpart in :attr:`job.molecule`.

        If a :attr:`template<job.block.template>` is specified then this block
        may or may not be redundant, depending on its completeness.


    .. attribute:: job.block.template

        :Parameter:     * **Type** - :class:`dict` or :class:`str`
                        * **Default Value** - ``{}``

        A Settings template for updating :class:`settings<job.block.settings>`.

        The template can be provided either as a dictionary or, alternativelly,
        an import path pointing to a pre-existing dictionary.
        For example, :code:`"qmflows.templates.md.specific.cp2k_mm"` is equivalent to
        :code:`import qmflows; template = qmflows.templates.md.specific.cp2k_mm`.

        .. admonition:: See Also

            :class:`qmflows.templates.md<qmflows.templates.templates.md>`
                Templates for molecular dynamics (MD) calculations.

            :class:`qmflows.templates.geometry<qmflows.templates.templates.geometry>`
                Templates for geometry optimization calculations.


monte_carlo
~~~~~~~~~~~
Settings related to the Monte Carlo procedure itself.

.. admonition:: Examples

    .. code:: yaml

        monte_carlo:
            type: FOX.armc.ARMC
            iter_len: 50000
            sub_iter_len: 10
            logfile: armc.log
            hdf5_file: armc.hdf5
            path: .
            folder: MM_MD_workdir
            keep_files: False

|

    .. attribute:: monte_carlo.type

        :Parameter:     * **Type** - :class:`str` or :class:`FOX.armc.MonteCarloABC<FOX.armc.monte_carlo.MonteCarloABC>` subclass
                        * **Default Value** - ``"FOX.armc.ARMC"``

        The type of Monte Carlo procedure.

        .. admonition:: See Also

            :class:`FOX.armc.ARMC<FOX.armc.armc.ARMC>`
                The Addaptive Rate Monte Carlo class.

            :class:`FOX.armc.ARMCPT<FOX.armc.armc_pt.ARMCPT>`
                An :class:`~FOX.armc.armc.ARMC` subclass implementing a parallel tempering procedure.


    .. attribute:: monte_carlo.iter_len

        :Parameter:     * **Type** - :class:`int`
                        * **Default Value** - ``50000``

        The total number of ARMC iterations :math:`\kappa \omega`.


    .. attribute:: monte_carlo.sub_iter_len

        :Parameter:     * **Type** - :class:`int`
                        * **Default Value** - ``100``

        The length of each ARMC subiteration :math:`\omega`.


    .. attribute:: monte_carlo.logfile

        :Parameter:     * **Type** - :class:`str`
                        * **Default Value** - ``"armc.log"``

        The name of the ARMC logfile.


    .. attribute:: monte_carlo.hdf5_file

        :Parameter:     * **Type** - :class:`str`
                        * **Default Value** - ``"armc.hdf5"``

        The name of the ARMC .hdf5 file.


    .. attribute:: monte_carlo.path

        :Parameter:     * **Type** - :class:`str`
                        * **Default Value** - ``"."``

        The path to the ARMC working directory.


    .. attribute:: monte_carlo.folder

        :Parameter:     * **Type** - :class:`str`
                        * **Default Value** - ``"MM_MD_workdir"``

        The name of the ARMC working directory.


    .. attribute:: monte_carlo.keep_files

        :Parameter:     * **Type** - :class:`bool`
                        * **Default Value** - ``"False"``

        Whether to keep *all* raw output files or not.


phi
~~~
Settings related to the ARMC :math:`\phi` parameter.

.. admonition:: Examples

    .. code:: yaml

        phi:
            type: FOX.armc.PhiUpdater
            gamma: 2.0
            a_target: 0.25
            phi: 1.0
            func: numpy.add
            kwargs: {}

|

    .. attribute:: phi.type

        :Parameter:     * **Type** - :class:`str` or :class:`FOX.armc.PhiUpdaterABC<FOX.armc.phi.PhiUpdaterABC>` subclass
                        * **Default Value** - ``"FOX.armc.PhiUpdater"``

        The type of phi updater.

        The phi updater is used for storing, keeping track of and updating :math:`\phi`.

        .. admonition:: See Also

            :class:`FOX.armc.PhiUpdater<FOX.armc.phi.PhiUpdater>`
                A class for applying and updating :math:`\phi`.


    .. attribute:: phi.gamma

        :Parameter:     * **Type** - :class:`float` or :class:`list` [:class:`float`]
                        * **Default Value** - ``2.0``

        The constant :math:`\gamma`.

        See :eq:`4`.
        Note that a list must be supplied when running the ARMC parallel
        tempering procedure (:attr:`monte_carlo.type = FOX.armc.ARMCPT<monte_carlo.type>`)


    .. attribute:: phi.a_target

        :Parameter:     * **Type** - :class:`float` or :class:`list` [:class:`float`]
                        * **Default Value** - ``0.25``

        The target acceptance rate :math:`\alpha_{t}`.

        See :eq:`4`.
        Note that a list must be supplied when running the ARMC parallel
        tempering procedure (:attr:`monte_carlo.type = FOX.armc.ARMCPT<monte_carlo.type>`)


    .. attribute:: phi.phi

        :Parameter:     * **Type** - :class:`float` or :class:`list` [:class:`float`]
                        * **Default Value** - ``1.0``

        The initial value of the variable :attr:`phi<phi.phi>`.

        See :eq:`3` and :eq:`4`.
        Note that a list must be supplied when running the ARMC parallel
        tempering procedure (:attr:`monte_carlo.type = FOX.armc.ARMCPT<monte_carlo.type>`)


    .. attribute:: phi.func

        :Parameter:     * **Type** - :class:`str` or :class:`~collections.abc.Callable`
                        * **Default Value** - ``"numpy.add"``

        The callable for updating phi.

        The passed callable should be able to take two floats as arguments and
        return a new float.

        .. admonition:: See Also

            :func:`numpy.add`
                Add arguments element-wise.


    .. attribute:: phi.kwargs

        :Parameter:     * **Type** - :class:`dict`
                        * **Default Value** - ``{}``

        A dictionary with further keyword arguments for :attr:`phi.func`.
