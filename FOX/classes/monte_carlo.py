""" A module fo running MD simulations. """

__all__ = ['MonteCarlo', 'ARMC']

import os
import shutil
from os.path import (join, dirname, isfile, isdir)

import yaml
import numpy as np

from scm.plams import Molecule
from scm.plams.core.results import Results
from scm.plams.core.settings import Settings
from scm.plams.core.functions import (init, finish, add_to_class)
from scm.plams.interfaces.thirdparty.cp2k import Cp2kJob

try:
    import h5py
    H5PY_ERROR = False
except ImportError:
    __all__ = []
    H5PY_ERROR = "Use of the FOX.{} class requires the 'h5py' package.\
                  \n\t'h5py' can be installed via anaconda with the following command:\
                  \n\tconda install --name FOX -y -c conda-forge h5py"

from .multi_mol import MultiMolecule
from ..functions.utils import get_shape


@add_to_class(Results)
def get_xyz_path(self):
    """ Return the path + filename to an .xyz file. """
    for file in self.files:
        if 'pos.xyz' in file:
            return self[file]
    raise FileNotFoundError('No .xyz files found in ' + self.job.path)


class MonteCarlo():
    """ The base MonteCarlo class.

    :parameter ref: A list with *n* PES descriptors as derived from , *e.g.*,
        an *Ab Initio* MD simulation.
    :type ref: *n* |list|_ [|float|_ or |np.ndarray|_ [|np.float64|_]]
    :parameter param: An array with the initial to be optimized forcefield parameters.
    :type param: |np.ndarray|_ [|np.float64|_]
    :parameter molecule: A molecule.
    :type molecule: |plams.Molecule|_ or |FOX.MultiMolecule|_

    :Atributes:     * **ref** (|list|_) – See the **ref** parameter.

                    * **param** (|np.ndarray|_ [|np.float64|_]) – See the **param** parameter.

                    * **pes** (|plams.Settings|_) – See :meth:`MonteCarlo.reconfigure_pes_atr`

                    * **move** (|plams.Settings|_) – See :meth:`MonteCarlo.reconfigure_move_atr`

                    * **job** (|plams.Settings|_) – See :meth:`MonteCarlo.reconfigure_job_atr`
    """
    def __init__(self, param, molecule, **kwarg):
        # Double check of h5py is actually installed
        try:
            h5py.__name__
        except NameError:
            raise ModuleNotFoundError("No module named 'h5py'")

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
        self.job.func = Cp2kJob
        self.job.settings = self.get_settings()
        self.job.name = self._get_name()
        self.job.path = os.getcwd()
        self.job.keep_files = False

        # Settings for generating Monte Carlo moves
        self.move = Settings()
        self.move.range = self.reconfigure_move_range()
        self.move.func = np.add
        self.move.kwarg = {}

        # Set user-specified keywords
        for key in kwarg:
            if key in self.param:
                self.param[key] = kwarg[key]
            elif key in self.move:
                self.move[key] = kwarg[key]
            elif key in self.job:
                self.job[key] = kwarg[key]

        self._sanitize()

    def __str__(self):
        return str(Settings(vars(self)))

    def _get_name(self):
        """ Return a jobname derived from **self.job.func**. """
        return str(self.job.func).rsplit("'", 1)[0].rsplit('.', 1)[-1]

    def _sanitize(self):
        """ Sanitize and validate all arguments of __init__(). """
        assert isinstance(self.job.molecule, (Molecule, MultiMolecule))
        if isinstance(self.job.molecule, MultiMolecule):
            self.job.molecule = self.job.molecule.as_Molecule(-1)[0]

        assert isinstance(self.job.name, str)
        assert isdir(self.job.path)

    def move(self):
        """ Update a random parameter in **self.param** by a random value from **self.move.range**.
        """
        i = np.random.choice(len(self.param), 1)
        j = np.random.choice(self.move.range, 1)
        self.param[i] = self.move.func(self.param[i], j, **self.move.kwarg)

    def run_md(self):
        """ Run an MD job, updating the cartesian coordinates of **self.job.mol** and returning
        a new MultiMolecule object.

        * The MD job is constructed according to the provided settings in **self.job**.

        :return: A MultiMolecule object constructed from the MD trajectory & the path to the PLAMS
            results directory.
        :rtype: |FOX.MultiMolecule|_ and |str|_
        """
        # Run an MD calculation
        job_type = self.job.func
        job = job_type(**self.job)
        results = job.run()
        results.wait()

        # Construct and return a MultiMolecule object
        mol =  MultiMolecule(filename=results.get_xyz_path())
        self.job.mol.from_array(mol[-1])
        return mol, job.path

    def run_first_md(self):
        """ Run the first MD job (before starting the actual main for-loop) and construct a
        matching list of PES descriptors.

        * The MD job is constructed according to the provided settings in **self.job**.

        * The PES descriptors are constructed by the provided settings in **self.pes**.

        :return: A dictionary, containing a list with one or more PES descriptors, and
            the first (and only) key in aforementioned dictionary.
        :rtype: |dict|_ (keys: |tuple|_, values: |list|_) and |tuple|_
        """
        # Run MD
        key = tuple(self.param)
        mol, path = self.run_md()

        # Create PES descriptors
        values = {i: self.pes[i].func(mol, **self.pes[i].kwarg) for i in self.pes}
        history_dict = {key: self.apply_phi(values)}

        # Delete the output directory and return
        if self.job.keep_files:
            shutil.rmtree(path)
        return history_dict, key

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
            return history_dict[key]
        else:
            mol, path = self.run_md()
            ret = {i: self.pes[i].func(mol, **self.pes[i].kwarg) for i in self.pes}

            # Delete the output directory and return
            if not self.job.keep_files:
                shutil.rmtree(path)
            return ret

    def get_settings(self, path=None):
        """ Grab the template with default CP2K_ MM-MD settings.
        If **path** is not *None*, read a settings template (.yaml file) from a user-specified path.
        Performs an inplace update of **self.job.settings**.

        :parameter path: An (optional) user-specified to a .yaml file with job settings
        :type path: |None|_ or |str|_
        """
        if isinstance(path, str):
            file_name = path
            assert isfile(path)
        elif isinstance(path, dict):
            self.job.settings = Settings(path)
            return
        else:
            file_name = join(join(dirname(dirname(__file__)), 'data'), 'md_cp2k.yaml')
        with open(file_name, 'r') as file:
            self.job.settings = yaml.load(file, Loader=yaml.FullLoader)

    def reconfigure_move_range(self, start=0.005, stop=0.1, step=0.005):
        """ Generate an with array of all allowed moves, the moves spanning both the positive and
        negative range.
        Performs an inplace update of **self.move.range**.

        :parameter float start: Start of the interval. The interval includes this value.
        :parameter float stop: End of the interval. The interval includes this value.
        :parameter float step: Spacing between values.
        """
        rng_range = np.arange(start, start + step, step)
        self.move_range = np.concatenate((-rng_range, rng_range))

    def reconfigure_move_atr(self, move_range=None, func=np.add, kwarg={}):
        """ Reconfigure the attributes in **self.move**., the latter containg all settings related
        to generating Monte Carlo moves.
        See :meth:`MonteCarlo.set_move_range` for more extensive **set_move_range** options.

        :parameter move_range: An array of allowed moves.
        :type move_range: |None|_ or |np.ndarray|_ [|np.float64|_]
        :parameter type func: A type object of a function used for performing moves.
            The function in **func** is applied to floats.
        :parameter dict kwarg: Keyword arguments used in **func**.
        """
        self.move.range = move_range or self.set_move_range()
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
        self.job.settings = settings or self.get_settings(settings)
        self.job.name = name or self._get_name()
        self.job.path = path or os.getcwd()
        self.job.keep_files = keep_files


class ARMC(MonteCarlo):
    """ The addaptive rate Monte Carlo (ARMC) class, a subclass of the FOX.MonteCarlo_.

    :Atributes:     * **armc** (|plams.Settings|_) – See :meth:`ARMC.reconfigure_armc_atr`

                    * **phi** (|plams.Settings|_) – See :meth:`ARMC.reconfigure_phi_atr`
    """
    def __init__(self, param, molecule, **kwarg):
        MonteCarlo.__init__(self, param, molecule, **kwarg)

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
        for key in kwarg:
            if key in self.armc:
                self.armc[key] = kwarg[key]

    def init_armc(self):
        """ Initialize the Addaptive Rate Monte Carlo procedure.

        :return: A new set of parameters.
        :rtype: |np.ndarray|_ [|np.float64|_]
        """
        # Unpack attributes
        acceptance = np.zeros(self.armc.sub_iter_len, dtype=bool)
        sub_iter = self.armc.sub_iter_len
        super_iter = self.armc.iter_len // sub_iter

        # Create a directory for storing pes descriptors
        self.create_mc_dirs()

        # Initialize
        init(path=self.job.path, folder='MM_MD_workdir')
        key_new, history_dict = self.run_first_md()
        for i in range(super_iter):
            for j in range(sub_iter):
                # Step 1: Perform a random move
                key_old = key_new
                self.move()
                key_new = tuple(self.param)
                self.param_to_hdf5(self.param, i, j)

                # Step 2: Check if the move has been performed already
                value_new = self.get_pes_descriptors(history_dict, key_new)
                self.pes_descriptors_to_hdf5(value_new, i, j)

                # Step 3: Evaluate the auxilary error
                value_old = history_dict[key_old]
                accept = bool(sum(self.get_aux_error(value_old) - self.get_aux_error(value_new)))
                self.acceptance_to_hdf5(accept, i, j)

                # Step 4: Update the PES descriptor history
                if accept:
                    acceptance[j] = True
                    history_dict[key_new] = self.apply_phi(value_new)
                else:
                    history_dict[key_old] = self.apply_phi(value_old)
            self.update_phi(acceptance)
            acceptance[:] = False
        finish()

        return self.param

    def acceptance_to_hdf5(self, acceptance, i, j):
        """ Add

        :parameter bool acceptance: Whether or not the parameters in the latest Monte Carlo
            iteration were accepted.
        :parameter int i: The iteration in the outer loop of :meth:`ARMC.init_armc`.
        :parameter int j: The subiteration in the inner loop of :meth:`ARMC.init_armc`.
        """
        k = j + i * j
        self._to_hdf5(acceptance, 'acceptance', k)

    def pes_descriptors_to_hdf5(self, descriptor_dict, i, j):
        """

        :parameter descriptor_dict: The latest set of PES-descriptors.
        :type descriptor_dict: |dict|_ (keys: |str|_, values: |np.ndarray|_ [|np.float64|_])
        :parameter int i: The iteration in the outer loop of :meth:`ARMC.init_armc`.
        :parameter int j: The subiteration in the inner loop of :meth:`ARMC.init_armc`.
        """
        k = j + i * j
        self._to_hdf5(descriptor_dict, 'pes_descriptors', k)

    def param_to_hdf5(self, param, i, j):
        """

        :parameter param: The latest set of ARMC-optimized parameters.
        :type param: |np.ndarray|_ [|np.float64|_]
        :parameter int i: The iteration in the outer loop of :meth:`ARMC.init_armc`.
        :parameter int j: The subiteration in the inner loop of :meth:`ARMC.init_armc`.
        """
        k = j + i * j
        self._to_hdf5(param, 'param', k)

    def _to_hdf5(self, item, name, k):
        """
        :parameter object item: An item which is to be added to an hdf5 file.
        :parameter str name: The filename (excluding path) of the hdf5 file.
        :parameter int k: The index in the hdf5 file where item should be placed.
        """
        filename = join(self.job.path, name)
        with h5py.File(filename, 'r+') as f:
            if isinstance(item, dict):
                for key in item:
                    f[key][k] = item[key]
            else:
                f[name][k] = item

    def get_aux_error(self, pes_dict):
        """ Return the auxiliary error of the PES descriptors in **values** with respect to
        **self.ref**.

        :parameter pes_dict: A dictionary of *n* PES descriptors.
        :type pes_dict: *n* |dict|_ (keys: |str|_, values: |np.ndarray|_ [|np.float64|_])
        :return: An array with *n* auxilary errors
        :rtype: *n* |np.ndarray|_ [|np.float64|_]
        """
        def get_norm(a, b):
            return np.linalg.norm(a - b, axis=0).sum()
        return np.array([get_norm(pes_dict[j], self.pes[j].ref) for i, j in enumerate(pes_dict)])

    def apply_phi(self, values):
        """ Apply **self.phi.phi** to all PES descriptors in **values**.

        * The values are updated according to the provided settings in **self.armc**.

        :parameter values: A list of *n* PES descriptors.
        :type values: *n* |list|_ [|float|_ or |np.ndarray|_ [|np.float64|_]]
        """
        phi = self.phi.phi
        func = self.phi.func
        kwarg = self.phi.kwarg
        for i in values:
            func(i, phi, **kwarg, out=i)

    def update_phi(self, acceptance):
        """ Update **self.phi** based on **self.armc.a_target** and **acceptance**.

        * The values are updated according to the provided settings in **self.armc**.

        :parameter acceptance: An array denoting the accepted moves within a sub-iteration.
        :type acceptance: |np.ndarray|_ [|bool|_]
        """
        sign = np.sign(self.armc.a_target - np.mean(acceptance))
        self.phi.phi *= self.armc.gamma**sign

    def create_hdf5(self):
        """ Create hdf5 files for storing ARMC results. """
        path = self.job.path

        # Create a dictionary with the shape and dtype of all to-be stored data
        shape_dict = Settings()
        shape_dict.param.shape = self.armc.iter_len, len(self.param)
        shape_dict.param.dtype = float
        shape_dict.acceptance.shape = self.armc.iter_len
        shape_dict.acceptance.dtype = bool

        # Create *n* hdf5 files with a single dataset
        kwarg = {'chunks': True, 'compression': 'gzip'}
        for key in shape_dict:
            filename = join(path, key) + '.hdf5'
            with h5py.File(filename, 'a') as f:
                shape = shape_dict[key].shape
                dtype = shape_dict[key].dtype
                f.create_dataset(name=key, data=np.zeros(shape, dtype=dtype),
                                 maxshape=shape, **kwarg)

        # Create a dictionary with the shape for all PES descriptors
        shape_dict = {}
        for i, j in zip(self.pes, self.ref):
            shape_dict[i].shape = (self.armc.iter_len, ) + get_shape(j)

        # Create a single hdf5 files with a *n* dataset
        filename = join(path, 'pes_descriptors') + '.hdf5'
        with h5py.File(filename, 'a') as f:
            for key in shape_dict:
                shape = shape_dict[key].shape
                f.create_dataset(name=key, data=np.zeros(shape, dtype=float),
                                 maxshape=shape, **kwarg)

    def reconfigure_armc_atr(self, iter_len=50000, sub_iter_len=100, gamma=2.0, a_target=0.25):
        """ Reconfigure the attributes in **self.armc**, the latter containing all settings
        specific to addaptive rate Monte Carlo (except phi).

        :parameter int iter_len: The total number of iterations (including sub-iterations).
        :parameter int sub_iter_len: The length of each sub-iteration.
        :parameter float gamma: The parameter gamma.
        :parameter float a_target: The target acceptance rate.
        """
        self.armc.iter_len = iter_len
        self.armc.sub_iter_len = sub_iter_len
        self.armc.gamma = gamma
        self.armc.a_target = a_target

    def reconfigure_phi_atr(self, phi=1.0, func=np.add, kwarg={}):
        """ Reconfigure the attributes in **self.phi**, the latter containing all settings specific
        to the phi parameter in addaptive rate Monte Carlo.

        :parameter float phi: The parameter phi.
        :parameter type func: A type object of a function used for performing moves.
            The function in **func** is applied to scalars, arrays and/or dataframes.
        :parameter dict kwarg: Keyword arguments used in **func**.
        """
        self.phi.phi = phi
        self.phi.func = func
        self.phi.kwarg = kwarg


# Raise an error when trying to call the MonteCarlo or ARMC class without 'h5py' installed
if H5PY_ERROR:
    class MonteCarlo(MonteCarlo):
        def __init__(self, *arg, **kwarg):
            name = str(self.__class__).rsplit("'", 1)[0].rsplit('.', 1)[1]
            raise ModuleNotFoundError(H5PY_ERROR.format(name))

    class ARMC(ARMC):
        def __init__(self, *arg, **kwarg):
            name = str(self.__class__).rsplit("'", 1)[0].rsplit('.', 1)[1]
            raise ModuleNotFoundError(H5PY_ERROR.format(name))
