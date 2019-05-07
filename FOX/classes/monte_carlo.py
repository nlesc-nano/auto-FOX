""" A module for performing Monte Carlo-based forcefield parameter optimizations. """

__all__ = ['MonteCarlo', 'ARMC']

import os
import shutil
from os.path import join

import numpy as np

from scm.plams.core.settings import Settings
from scm.plams.core.functions import (init, finish, add_to_class, config)
from scm.plams.interfaces.thirdparty.cp2k import (Cp2kJob, Cp2kResults)

from .multi_mol import MultiMolecule
from ..io.hdf5_utils import (create_hdf5, to_hdf5)
from ..functions.utils import (get_template, write_psf, _get_move_range)
from ..functions.cp2k_utils import (update_cp2k_settings, set_subsys_kind)
from ..functions.charge_utils import update_charge
from ..functions.armc_sanitization import init_armc_sanitization


@add_to_class(Cp2kResults)
def get_xyz_path(self):
    """ Return the path + filename to an .xyz file. """
    for file in self.files:
        if '-pos' in file and '.xyz' in file:
            return self[file]
    raise FileNotFoundError('No .xyz files found in ' + self.job.path)


class MonteCarlo():
    """ The base :class:`.MonteCarlo` class.

    :parameter ref: A list with :math:`n` PES descriptors as derived from , *e.g.*,
        an *Ab Initio* MD simulation.
    :type ref: :math:`n` |list|_ [|float|_ or |np.ndarray|_ [|np.float64|_]]
    :parameter param: An array with the initial to be optimized forcefield parameters.
    :type param: |np.ndarray|_ [|np.float64|_]
    :parameter molecule: A molecule.
    :type molecule: |plams.Molecule|_ or |FOX.MultiMolecule|_

    :Atributes:     * **param** (|pd.DataFrame|_ – See the **param** parameter.

                    * **pes** (|plams.Settings|_) – See :meth:`MonteCarlo.reconfigure_pes_atr`.

                    * **job** (|plams.Settings|_) – See :meth:`MonteCarlo.reconfigure_job_atr`.

                    * **hdf5_file** (|str|_) – The hdf5 path+filename.

                    * **move** (|plams.Settings|_) – See :meth:`MonteCarlo.reconfigure_move_atr`.
    """
    def __init__(self, molecule, param, **kwarg):
        # Set the inital forcefield parameters
        self.param = param

        # Settings for generating PES descriptors and assigns of reference PES descriptors
        self.pes = Settings()
        self.pes.rdf.ref = None
        self.pes.rdf.func = MultiMolecule.init_rdf
        self.pes.rdf.kwarg = {'atom_subset': None}

        # Settings for running the actual MD calculations
        self.job = Settings()
        self.job.molecule = molecule
        self.job.psf = {}
        self.job.func = Cp2kJob
        self.job.settings = get_template('md_cp2k.yaml')
        self.job.name = self.job.func.__name__.lower()
        self.job.path = os.getcwd()
        self.job.folder = 'MM_MD_workdir'
        self.job.keep_files = False

        # HDF5 settings
        self.hdf5_file = join(self.job.path, 'ARMC.hdf5')

        # Settings for generating Monte Carlo moves
        self.move = Settings()
        self.move.func = np.multiply
        self.move.kwarg = {}
        self.move.charge_constraints = {}
        self.move.range = self.get_move_range()

    def __repr__(self):
        return str(Settings(vars(self)))

    def move_param(self):
        """ Update a random parameter in **self.param** by a random value from **self.move.range**.
        Performs in inplace update of the *param* column in **self.param**.
        By default the move is applied in a multiplicative manner
        (see :meth:`MonteCarlo.reconfigure_move_atr`).

        :return: A tuple with the (new) values in the *param* column of **self.param**.
        :rtype: |tuple|_ [|float|_]
        """
        # Unpack arguments
        param = self.param
        psf = self.job.psf['atoms']

        # Perform a move
        i = param.loc[:, 'param'].sample()
        j = np.random.choice(self.move.range, 1)
        param.loc[i.index, 'param'] = self.move.func(i, j, **self.move.kwarg)
        i = param.loc[i.index, 'param']

        # Constrain the atomic charges
        if 'charge' in i:
            for (_, at), charge in i.iteritems():
                pass
            exclude = [i for i in psf if i in param.loc['charge'].index]
            update_charge(at, charge, psf, self.move.charge_constraints, exclude)
            idx_set = set(psf['atom type'].values)
            for at in idx_set:
                if ('charge', at) in param.index:
                    condition = psf['atom type'] == at, 'charge'
                    param.loc[('charge', at), 'param'] = psf.loc[condition].iloc[0]
            write_psf(**self.job.psf)

        # Update job settings and return a tuple with new parameters
        update_cp2k_settings(self.job.settings, self.param)
        return tuple(self.param['param'].values)

    def run_md(self):
        """ Run a molecular dynamics (MD) job, updating the cartesian coordinates of
        **self.job.mol** and returning a new :class:`.MultiMolecule` instance.

        * The MD job is constructed according to the provided settings in **self.job**.

        :return: A :class:`.MultiMolecule` instance constructed from the MD trajectory &
            the path to the PLAMS results directory.
        :rtype: |FOX.MultiMolecule|_ and |str|_
        """
        # Run an MD calculation
        job_type = self.job.func
        job = job_type(name=self.job.name, molecule=self.job.molecule, settings=self.job.settings)
        results = job.run()
        results.wait()

        # Construct and return a MultiMolecule object
        mol = MultiMolecule.from_xyz(results.get_xyz_path())
        if mol.shape[0] == self.job.settings.input.motion.md.steps:
            self.job.molecule = mol.as_Molecule(-1)[0]
        return mol, job.path

    def get_pes_descriptors(self, history_dict, key):
        """ Check if a **key** is already present in **history_dict**.
        If *True*, return the matching list of PES descriptors;
        If *False*, construct and return a new list of PES descriptors.

        * The PES descriptors are constructed by the provided settings in **self.pes**.

        :parameter history_dict: A dictionary with results from previous iteractions.
        :type history_dict: |dict|_ (keys: |tuple|_, values: |list|_)
        :parameter key: A key in **history_dict**.
        :type key: |tuple|_
        :return: A previous value from **history_dict** or a new value from an MD calculation.
        :rtype: |dict|_ (keys: |str|_, values: |np.ndarray|_ [|np.float64|_])
        """
        if key in history_dict:
            return False

        # Generate PES descriptors
        mol, path = self.run_md()
        ret = {key: value.func(mol, **value.kwarg) for key, value in self.pes.items()}

        # Delete the output directory and return
        if not self.job.keep_files:
            shutil.rmtree(path)
        return ret

    @staticmethod
    def get_move_range(start=0.005, stop=0.1, step=0.005):
        """ Generate an with array of all allowed moves.
        The move range spans a range of 1.0 +- **stop** and moves are thus intended to
        applied in a multiplicative manner (see :meth:`MonteCarlo.move_param`).


        Example:

        .. code: python

            >>> move_range = ARMC.get_move_range(start=0.005, stop=0.1, step=0.005)
            >>> print(move_range)
            [0.9   0.905 0.91  0.915 0.92  0.925 0.93  0.935 0.94  0.945
             0.95  0.955 0.96  0.965 0.97  0.975 0.98  0.985 0.99  0.995
             1.005 1.01  1.015 1.02  1.025 1.03  1.035 1.04  1.045 1.05
             1.055 1.06  1.065 1.07  1.075 1.08  1.085 1.09  1.095 1.1  ]

        :parameter float start: Start of the interval. The interval includes this value.
        :parameter float stop: End of the interval. The interval includes this value.
        :parameter float step: Spacing between values.
        :return: An array with allowed moves.
        :rtype: |np.ndarray|_ [|np.int64|_]
        """
        return _get_move_range(start, stop, step)

    def reconfigure_move_atr(self, move_range=None, func=np.multiply, kwarg={}):
        """ Reconfigure the attributes in **self.move**, the latter containg all settings related
        to generating Monte Carlo moves.
        See :meth:`MonteCarlo.get_move_range` for more extensive **move_range** options.

        :parameter move_range: An array of allowed moves.
        :type move_range: |None|_ or |np.ndarray|_ [|np.float64|_]
        :parameter type func: A type object of a function used for performing moves.
            The function in **func** is applied to floats.
        :parameter dict kwarg: Keyword arguments used in **func**.
        """
        self.move.range = move_range or self.get_move_range()
        self.move.func = func
        self.move.kwarg = kwarg

    def reconfigure_job_atr(self, molecule=None, func=Cp2kJob, settings=None,
                            name=None, path=None, keep_files=False):
        """ Reconfigure the attributes in **self.job**, the latter containing all settings related
        to the PLAMS Job class and its subclasses.

        :parameter molecule: A PLAMS molecule.
        :type molecule: |None|_ or |plams.Molecule|_
        :parameter type func: A type object of PLAMS Job subclass.
            Used for running the MD calculations.
        :parameter settings: The job settings used by **func**.
        :type settings: |plams.Settings|_
        :parameter str name: The name of the directories holding the MD results produced by
            **func**.
        :parameter str path: The path where **name** will be stored.
        :parameter bool keep_files: Whether or not all files produced should during the MD
            calculations should be kept or deleted. WARNING: The size of a single MD trajectory can
            easily be in the dozens or even hundreds of megabytes; the MC parameter optimization
            will require thousands of such trajectories.
        """
        if molecule is not None:
            self.job.molecule = molecule
        self.job.func = func
        self.job.settings = settings or get_template('md_cp2k.yaml')
        self.job.name = name or self.job.func.__name__
        self.job.path = path or os.getcwd()
        self.job.keep_files = keep_files


class ARMC(MonteCarlo):
    """ The Addaptive Rate Monte Carlo class (:class:`.ARMC`), a subclass of the base
    :class:`.MonteCarlo` class.

    :Atributes:     * **armc** (|plams.Settings|_) – See :meth:`ARMC.reconfigure_armc_atr`

                    * **phi** (|plams.Settings|_) – See :meth:`ARMC.reconfigure_phi_atr`
    """
    def __init__(self, molecule, param, **kwarg):
        MonteCarlo.__init__(self, molecule, param, **kwarg)

        # Settings specific to addaptive rate Monte Carlo (ARMC)
        self.armc = Settings()
        self.armc.iter_len = 50000
        self.armc.sub_iter_len = 100
        self.armc.gamma = 2.0
        self.armc.a_target = 0.25

        # Settings specific to handling the phi parameter in ARMC
        self.phi = Settings()
        self.phi.phi = 1.0
        self.phi.func = np.add
        self.phi.kwarg = {}

        # Set user-specified keywords
        for key, value in kwarg.items():
            if not hasattr(self, key):
                continue

            try:
                getattr(self, key).update(value)
            except AttributeError:
                setattr(self, key, value)

    def __str__(self):
        ret = Settings(vars(self))

        # The self.pes block
        for key, value in ret.pes.items():
            value.ref = str(value.ref.__class__)
            value.kwarg = str(value.kwarg.as_dict())
            value.func = self.get_func_name(value.func)

        # The self.job block
        ret.job.molecule = str(self.job.molecule.__class__)
        ret.job.settings = str(self.job.settings.__class__)
        ret.job.psf = str(self.job.psf.__class__)
        ret.job.func = self.get_class_name(ret.job.func)

        # The self.move block
        ret.move.kwarg = str(ret.move.kwarg.as_dict())
        ret.move.func = self.get_func_name(ret.move.func)
        ret.move.range = np.array2string(ret.move.range, precision=3,
                                         floatmode='fixed', threshold=20)

        # The self.phi block
        ret.phi.kwarg = str(ret.phi.kwarg.as_dict())
        ret.phi.func = self.get_func_name(ret.phi.func)

        # The self.move block
        for value in ret.move.charge_constraints.values():
            value.func = self.get_func_name(value.func)

        # The self.param block
        param = ret.param['param'].to_dict()
        ret.param = {}
        for (key1, key2), value in param.items():
            try:
                ret.param[key1].update({key2: value})
            except KeyError:
                ret.param[key1] = {key2: value}

        return str(ret)

    @staticmethod
    def get_func_name(item):
        try:
            item_class, item_name = item.__qualname__.split('.')
            item_module = item.__module__.split('.')[0]
        except AttributeError:
            item_name = item.__name__
            item_class = item.__class__.__name__
            item_module = item.__class__.__module__.split('.')[0]
        return '{}.{}.{}'.format(item_module, item_class, item_name)

    @staticmethod
    def get_class_name(item):
        item_class = item.__qualname__
        item_module = item.__module__.split('.')[0]
        if item_module == 'scm':
            item_module == item.__module__.split('.')[1]
        return '{}.{}'.format(item_module, item_class)

    @staticmethod
    def from_yaml(yml_file):
        """ Create a :class:`.ARMC` instance from a .yaml file.

        :parameter str yml_file: The path+filename of a .yaml file containing all :class:`ARMC`
            settings.
        :return: A :class:`ARMC` instance.
        :rtype: |FOX.ARMC|_
        """
        return ARMC.from_dict(get_template(yml_file))

    @classmethod
    def from_dict(cls, dict_):
        """ Create a :class:`.ARMC` instance from a dictionary.

        :parameter dict dict_: A dictionary containing all :class:`ARMC` settings.
        :return: A :class:`ARMC` instance.
        :rtype: |FOX.ARMC|_
        """
        molecule, param, dict_ = init_armc_sanitization(dict_)
        set_subsys_kind(dict_.job.settings, molecule.properties.psf)
        molecule = molecule.as_Molecule(-1)[0]
        return cls(molecule, param, **dict_)

    def init_armc(self):
        """ Initialize the Addaptive Rate Monte Carlo procedure.

        :return: A new set of parameters.
        :rtype: |pd.DataFrame|_ (index: |pd.MultiIndex|_, values: |np.float64|_)
        """
        # Unpack attributes
        super_iter = self.armc.iter_len // self.armc.sub_iter_len

        # Construct the HDF5 file
        create_hdf5(self.hdf5_file, self)

        # Initialize
        init(path=self.job.path, folder=self.job.folder)
        config.default_jobmanager.settings.hashing = None
        if self.job.logfile:
            config.default_jobmanager.logfile = self.job.logfile
        write_psf(**self.job.psf)

        # Initialize the first MD calculation
        history_dict = {}
        key_new = tuple(self.param['param'].values)
        pes_new = self.get_pes_descriptors(history_dict, key_new)
        history_dict[key_new] = self.get_aux_error(pes_new)
        self.param['param_old'] = self.param['param']

        # Start the main loop
        for kappa in range(super_iter):
            key_new = self.do_inner(kappa, history_dict, key_new)
        finish()
        return self.param

    def do_inner(self, kappa, history_dict, key_new):
        r""" A method that handles the inner loop of the :meth:`ARMC.init_armc` method.

        :parameter int kappa: The super-iteration, :math:`\kappa`, in :meth:`ARMC.init_armc`.
        :parameter history_dict: A dictionary with parameters as keys and a list of PES descriptors
            as values.
        :type history_dict: |dict|_ (keys: |tuple|_, values: |dict|_ [|pd.DataFrame|_])
        :parameter key_new: A tuple with the latest set of forcefield parameters.
        :type key_new: |tuple|_ [|int|_]
        :return: The latest set of parameters and the acceptance rate, :math:`\alpha`, over the
            course of the inner loop.
        :rtype: |tuple|_ [|int|_] and |np.ndarray|_ [|bool|_]
        """
        acceptance = np.zeros(self.armc.sub_iter_len, dtype=bool)
        hdf5_kwarg = {'param': self.param, 'acceptance': False}

        for omega in range(self.armc.sub_iter_len):
            # Step 1: Perform a random move
            key_old = key_new
            key_new = self.move_param()

            # Step 2: Check if the move has been performed already; calculate PES descriptors if not
            pes_new = self.get_pes_descriptors(history_dict, key_new)
            hdf5_kwarg.update(pes_new)

            # Step 3: Evaluate the auxiliary error
            if not pes_new:
                aux_new = history_dict[key_new]
                pes_new = {key: np.nan for key in self.pes}
            else:
                aux_new = self.get_aux_error(pes_new)
            aux_old = history_dict[key_old]
            accept = bool(sum(aux_new - aux_old))

            # Step 4: Update the auxiliary error history, apply phi & update job settings
            acceptance[omega] = accept
            history_dict[key_new] = aux_new
            if accept:
                history_dict[key_new] = self.apply_phi(aux_new)
                self.param['param_old'] = self.param['param']
                hdf5_kwarg['aux_error_mod'] = np.append(self.param['param'].values, self.phi.phi)
            else:
                history_dict[key_old] = self.apply_phi(aux_old)
                key_new = key_old
                self.param['param'] = self.param['param_old']
                hdf5_kwarg['aux_error_mod'] = np.append(self.param['param_old'].values,
                                                        self.phi.phi)

            # Step 5: Export the results to HDF5
            hdf5_kwarg['param'] = self.param['param']
            hdf5_kwarg['acceptance'] = accept
            hdf5_kwarg['aux_error'] = aux_new
            to_hdf5(self.hdf5_file, hdf5_kwarg, kappa, omega, self.phi.phi)

        self.update_phi(acceptance)
        return key_new

    def get_aux_error(self, pes_dict):
        r""" Return the auxiliary error, :math:`\Delta \varepsilon_{QM-MM}`, of the PES descriptors
        in **values** with respect to **self.ref**.


        The default is equivalent to:

        .. math::

            \Delta \varepsilon_{QM-MM} =
            \sum_{n} \sqrt{
                \sum_{r_{ij}=0}^{r_{max}} (\Delta g_{n} (r_{ij}))^2
            }

        :parameter pes_dict: A dictionary of *n* PES descriptors.
        :type pes_dict: *n* |dict|_ (keys: |str|_, values: |np.ndarray|_ [|np.float64|_])
        :return: An array with *n* auxilary errors
        :rtype: *n* |np.ndarray|_ [|np.float64|_]
        """
        def norm_sum(a, b):
            return np.linalg.norm(a - b, axis=0).sum()
        return np.array([norm_sum(pes_dict[i], self.pes[i].ref) for i in pes_dict])

    def apply_phi(self, aux_error):
        r""" Apply :math:`\phi` to all auxiliary errors, :math:`\Delta \varepsilon_{QM-MM}`,
        in **aux_error**.

        * The values are updated according to the provided settings in **self.armc**.


        The default is equivalent to:

        .. math::

            \Delta \varepsilon_{QM-MM} = \Delta \varepsilon_{QM-MM} + \phi

        :parameter aux_error: An array with auxiliary errors
        :type aux_error: |np.ndarray|_ [|np.float64|_]
        :return: **aux_error** with updated values.
        :rtype: |np.ndarray|_ [|np.float64|_]
        """
        return self.phi.func(aux_error, self.phi.phi, **self.phi.kwarg)

    def update_phi(self, acceptance):
        r""" Update :math:`\phi` based on the target accepatance rate, :math:`\alpha_{t}`, and the
        acceptance rate, **acceptance**, in the current super-iteration.

        * The values are updated according to the provided settings in **self.armc**.


        The default is equivalent to:

        .. math::

            \phi_{\kappa \omega} =
            \phi_{ ( \kappa - 1 ) \omega} * \gamma^{
                \text{sgn} ( \alpha_{t} - \overline{\alpha}_{ ( \kappa - 1 ) })
            }

        :parameter acceptance: An array denoting the accepted moves within a sub-iteration.
        :type acceptance: |np.ndarray|_ [|bool|_]
        """
        sign = np.sign(self.armc.a_target - np.mean(acceptance))
        self.phi.phi *= self.armc.gamma**sign

    def reconfigure_armc_atr(self, iter_len=50000, sub_iter_len=100, gamma=2.0, a_target=0.25):
        r""" Reconfigure the attributes in **self.armc**, the latter containing all settings
        specific to addaptive rate Monte Carlo (except :math:`\phi`,
        see :meth:`.reconfigure_phi_atr`).

        :parameter int iter_len: The total number of iterations :math:`\kappa \omega`.
        :parameter int sub_iter_len: The length of each sub-iteration :math:`\omega`.
        :parameter float gamma: The parameter :math:`\gamma`.
        :parameter float a_target: The target acceptance rate :math:`\alpha_{t}`.
        """
        self.armc.iter_len = iter_len
        self.armc.sub_iter_len = sub_iter_len
        self.armc.gamma = gamma
        self.armc.a_target = a_target

    def reconfigure_phi_atr(self, phi=1.0, func=np.add, kwarg={}):
        r""" Reconfigure the attributes in **self.phi**, the latter containing all settings specific
        to the phi parameter in addaptive rate Monte Carlo.

        :parameter float phi: The parameter :math:`\phi`.
        :parameter type func: A type object of a function used for performing moves.
            The function in **func** is applied to scalars, arrays and/or dataframes.
        :parameter dict kwarg: Keyword arguments used in **func**.
        """
        self.phi.phi = phi
        self.phi.func = func
        self.phi.kwarg = kwarg
