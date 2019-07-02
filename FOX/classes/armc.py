"""A module for performing Addaptive Rate Monte Carlo (ARMC) forcefield parameter optimizations."""

from __future__ import annotations

from os.path import (isfile, split)

from typing import (Tuple, Dict, Any, Optional)
import numpy as np

from scm.plams import Settings
from scm.plams.core.functions import (init, finish, config)

from .monte_carlo import MonteCarlo
from ..io.hdf5_utils import (create_hdf5, to_hdf5, create_xyz_hdf5)
from ..functions.utils import (get_template, get_class_name, get_func_name)
from ..armc_functions.sanitization import init_armc_sanitization

__all__ = ['ARMC']


class ARMC(MonteCarlo):
    r"""The Addaptive Rate Monte Carlo class (:class:`.ARMC`).

    A subclass of :class:`.MonteCarlo`.

    Attributes
    ----------
    armc : |plams.Settings|_
        A PLAMS Settings instance with ARMC-specific settings.
        Contains the following keys:

        * ``"gamma"`` (|float|_): The constant :math:`\gamma`.
        * ``"a_target"`` (|float|_): The target acceptance rate :math:`\alpha_{t}`.
        * ``"iter_len"`` (|int|_): The total number of ARMC iterations :math:`\kappa \omega`.
        * ``"sub_iter_len"`` (|int|_): The length of each ARMC subiteration :math:`\omega`.

    phi : |plams.Settings|_
        A PLAM Settings instance with :math:`\phi`-specific settings.
        Contains the following keys:

        * ``"phi"`` (|float|_): The variable :math:`\phi`.
        * ``"arg"`` (|list|_): A list of arguments for :attr:`.ARMC.phi` [``"func"``].
        * ``"func"`` (|type|_): The callable used for applying :math:`\phi` to the auxiliary error.
        * ``"kwarg"`` (|dict|_): A dictionary with keyword arguments
          for :attr:`.ARMC.phi` [``"func"``].

    """

    def __init__(self, **kwarg: dict) -> None:
        """Initialize a :class:`ARMC` instance."""
        MonteCarlo.__init__(self, **kwarg)

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

    def __str__(self) -> str:
        """Return a string constructed from this instance."""
        ret = Settings(vars(self))

        # The self.pes block
        for value in ret.pes.values():
            value.ref = str(value.ref.__class__)
            value.kwarg = str(value.kwarg.as_dict())
            value.func = get_func_name(value.func) + '()'

        # The self.job block
        ret.job.molecule = str(self.job.molecule.__class__)
        ret.job.preopt_settings = str(self.job.preopt_settings.__class__)
        ret.job.md_settings = str(self.job.md_settings.__class__)
        ret.job.psf = str(self.job.psf.__class__)
        ret.job.func = get_class_name(ret.job.func) + '()'

        # The self.move block
        ret.move.kwarg = str(ret.move.kwarg.as_dict())
        ret.move.func = get_func_name(ret.move.func) + '()'
        ret.move.range = np.array2string(ret.move.range, precision=3,
                                         floatmode='fixed', threshold=20)

        # The self.phi block
        ret.phi.kwarg = str(ret.phi.kwarg.as_dict())
        ret.phi.func = get_func_name(ret.phi.func) + '()'

        # The self.move block
        if not ret.move.charge_constraints:
            ret.move.charge_constraints = '{}'
        else:
            for value in ret.move.charge_constraints.values():
                value.func = get_func_name(value.func) + '()'

        # The self.param block
        param = ret.param['param'].to_dict()
        unit = ret.param['unit'].to_dict()
        ret.param = {}
        for (key1, key2), value in param.items():
            if unit[(key1, key2)] is not None:
                value = str(value) + ' \t' + unit[(key1, key2)].lstrip('[').rstrip('] {:f}')
            try:
                ret.param[key1].update({key2: value})
            except KeyError:
                ret.param[key1] = {key2: value}

        return _str(ret)

    @staticmethod
    def from_yaml(filename: str) -> ARMC:
        """Create a :class:`.ARMC` instance from a .yaml file.

        Parameters
        ----------
        filename : str
            The path+filename of a .yaml file containing all :class:`ARMC` settings.

        Returns
        -------
        |FOX.ARMC|_:
            A new :class:`ARMC` instance.

        """
        if isfile(filename):
            path, filename = split(filename)
            return ARMC.from_dict(get_template(filename, path=path))
        else:
            return ARMC.from_dict(get_template(filename))

    @classmethod
    def from_dict(cls, armc_dict: Settings) -> ARMC:
        """Create a :class:`.ARMC` instance from a dictionary.

        Parameters
        ----------
        armc_dict : dict
            A dictionary containing all :class:`ARMC` settings.

        Returns
        -------
        |FOX.ARMC|_:
            A new :class:`ARMC` instance.

        """
        s = init_armc_sanitization(armc_dict)
        s.job.molecule = s.job.molecule.as_Molecule(0)[0]
        return cls(**s)

    def _create_history_dict(self) -> Tuple[Dict[Tuple[float], np.ndarray], Tuple[float]]:
        """Create a the ``history_dict`` variable and its first key.

        The to-be returned key servers as the starting argument for :meth:`.do_inner`,
        the latter method relying on both producing and requiring a key as argument.

        Returns
        -------
        |dict|_ [|tuple|_ [|float|_], |np.ndarray|_ [|np.float64|_]] and |tuple|_ [|float|_]
            Returns two items:
            * A dictionary with parameters as keys and a list of PES descriptors as values.
            * A tuple with the latest set of forcefield parameters.

        """
        history_dict: dict = {}
        key_new = tuple(self.param['param'].values)
        pes_new, _ = self.get_pes_descriptors(history_dict, key_new)

        history_dict[key_new] = self.get_aux_error(pes_new)
        self.param['param_old'] = self.param['param']
        return history_dict

    def init_armc(self) -> None:
        """Initialize the Addaptive Rate Monte Carlo procedure."""
        # Unpack attributes
        super_iter = self.armc.iter_len // self.armc.sub_iter_len

        # Construct the HDF5 file
        create_hdf5(self.hdf5_file, self)

        # Initialize; configure PLAMS global variables
        init(path=self.job.path, folder=self.job.folder)
        config.default_jobmanager.settings.hashing = None
        if self.job.logfile:
            config.default_jobmanager.logfile = self.job.logfile
            config.log.file = 3

        # Create a .psf file if specified
        if self.job.psf[0]:  # TODO: too vague
            self.job.psf.write_psf()

        # Initialize the first MD calculation
        history_dict, key_new = self._create_history_dict()

        # Start the main loop
        for kappa in range(super_iter):
            key_new = self.do_inner(kappa, history_dict, key_new)
        finish()

    def do_inner(self, kappa: float,
                 history_dict: Dict[Tuple[float], np.ndarray],
                 key_new: Tuple[float]) -> Tuple[float]:
        r"""Run the inner loop of the :meth:`ARMC.init_armc` method.

        Parameters
        ----------
        kappa : int
            The super-iteration, :math:`\kappa`, in :meth:`ARMC.init_armc`.

        history_dict : |dict|_ [|tuple|_ [|float|_], |np.ndarray|_ [|np.float64|_]]
            A dictionary with parameters as keys and a list of PES descriptors as values.

        key_new : tuple [float]
            A tuple with the latest set of forcefield parameters.

        Returns
        -------
        |tuple|_ [|float|_] and |np.ndarray|_ [|bool|_]:
            The latest set of parameters and the acceptance rate, :math:`\alpha`, over the
            course of the inner loop.

        """
        acceptance = np.zeros(self.armc.sub_iter_len, dtype=bool)
        create_xyz_hdf5(self.hdf5_file, self.job.molecule, iter_len=self.armc.sub_iter_len)

        for omega in range(self.armc.sub_iter_len):
            # Step 1: Perform a random move
            key_old = key_new
            key_new = self.move_param()

            # Step 2: Check if the move has been performed already; calculate PES descriptors if not
            pes_new, mol = self.get_pes_descriptors(history_dict, key_new)

            # Step 3: Evaluate the auxiliary error; accept if the new parameter set lowers the error
            aux_new = self.get_aux_error(pes_new)
            aux_old = history_dict[key_old]
            accept = True if (aux_new - aux_old).sum() < 0 else False

            # Step 4: Update the auxiliary error history, apply phi & update job settings
            acceptance[omega] = accept
            history_dict[key_new] = aux_new
            if accept:
                history_dict[key_new] = self.apply_phi(aux_new)
                self.param['param_old'] = self.param['param']
            else:
                history_dict[key_old] = self.apply_phi(aux_old)
                key_new = key_old
                self.param['param'] = self.param['param_old']

            # Step 5: Export the results to HDF5
            hdf5_kwarg = self._get_hdf5_dict(mol, accept, aux_new, pes_new)
            to_hdf5(self.hdf5_file, hdf5_kwarg, kappa, omega)

        self.update_phi(acceptance)
        return key_new

    def _get_hdf5_dict(self, mol: Optional['FOX.MultiMolecule'],
                       accept: bool,
                       aux_new: np.ndarray,
                       pes_new: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Construct a dictionary with the **hdf5_kwarg** argument for :func:`.to_hdf5`.

        Parameters
        ----------
        mol : |FOX.MultiMolecule|_
            Optional: A :class:`.MultiMolecule` instance.

        accept : bool
            Whether or not the latest set of parameters was accepted.

        aux_new : |np.ndarray|_
            The latest auxiliary error.

        pes_new : |dict|_ [|str|_, |np.ndarray|_]
            A dictionary of PES descriptors.

        Returns
        -------
        |dict|_
            A dictionary with the **hdf5_kwarg** argument for :func:`.to_hdf5`.

        """
        param_key = 'param' if accept else 'param_old'
        hdf5_kwarg = {
            'param': self.param['param'],
            'xyz': mol if mol is not None else np.nan,
            'phi': self.phi.phi,
            'acceptance': accept,
            'aux_error': aux_new,
            'aux_error_mod': np.append(self.param[param_key].values, self.phi.phi)
        }
        hdf5_kwarg.update(pes_new)
        return hdf5_kwarg

    def get_aux_error(self, pes_dict: Dict[str, np.ndarray]) -> np.ndarray:
        r"""Return the auxiliary error :math:`\Delta \varepsilon_{QM-MM}`.

        The auxiliary error is constructed using the PES descriptors in **values**
        with respect to **self.ref**.

        The default function is equivalent to:

        .. math::

            \Delta \varepsilon_{QM-MM} =
            \sqrt {
                \frac{1}{N}
                \sum_{i}^{N}
                \frac{ |r_{i}^{QM} - r_{i}^{MM}|^2 }
                {r_{i}^{QM}}
            }

        Parameters
        ----------
        pes_dict : dict [str, |np.ndarray|_ [|np.float64|_]]
            A dictionary with *n* PES descriptors.

        Returns
        -------
        :math:`n` |np.ndarray|_ [|np.float64|_]:
            An array with *n* auxilary errors

        """
        def norm_mean(mm_pes: np.ndarray, key: str) -> float:
            qm_pes = self.pes[key].ref
            A, B = np.asarray(qm_pes, dtype=float), np.asarray(mm_pes, dtype=float)
            ret = np.abs(A - B)**2
            return ret.sum() / A.sum()

        return np.array([norm_mean(mm_pes, key) for key, mm_pes in pes_dict.items()])

    def apply_phi(self, aux_error: np.ndarray) -> np.ndarray:
        r"""Apply :math:`\phi` to all auxiliary errors :math:`\Delta \varepsilon_{QM-MM}`.

        * The values are updated according to the provided settings in **self.armc**.

        The default function is equivalent to:

        .. math::

            \Delta \varepsilon_{QM-MM} = \Delta \varepsilon_{QM-MM} + \phi

        Parameters
        ----------
        aux_error : |np.ndarray|_ [|np.float64|_]
            An array with auxiliary errors.

        Returns
        -------
        |np.ndarray|_ [|np.float64|_]:
            **aux_error** with updated values.

        """
        return self.phi.func(aux_error, self.phi.phi, *self.phi.arg, **self.phi.kwarg)

    def update_phi(self, acceptance: np.ndarray) -> None:
        r"""Update the variable :math:`\phi`.

        :math:`\phi` is updated based on the target accepatance rate, :math:`\alpha_{t}`, and the
        acceptance rate, **acceptance**, of the current super-iteration.

        * The values are updated according to the provided settings in **self.armc**.

        The default function is equivalent to:

        .. math::

            \phi_{\kappa \omega} =
            \phi_{ ( \kappa - 1 ) \omega} * \gamma^{
                \text{sgn} ( \alpha_{t} - \overline{\alpha}_{ ( \kappa - 1 ) })
            }

        Parameters
        ----------
        acceptance : |np.ndarray|_ [|bool|_]
            A 1D boolean array denoting the accepted moves within a sub-iteration.

        """
        sign = np.sign(self.armc.a_target - np.mean(acceptance))
        self.phi.phi *= self.armc.gamma**sign

    def restart(self, filename: str) -> None:
        r"""Restart a previously started Addaptive Rate Monte Carlo procedure.

        Restarts from the beginning of the last super-iteration :math:`\kappa`.

        Parameters
        ----------
        filename : str
            The path+name of the an ARMC hdf5 file.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError


def _str(dict_: dict,
         indent1: str = '') -> str:
    ret = ''
    indent2 = 3
    try:
        indent2 += max(len(i) for i in dict_.keys())
    except ValueError:
        pass
    for key, value in sorted(dict_.items()):
        if indent1 == '':
            ret += '\n'
        if isinstance(value, dict):
            ret += indent1 + key + ':'
            ret += '\n' + _str(value, indent1+'    ')
        elif isinstance(value, (int, float)) and value < 0:
            ret += indent1 + '{:{width}}'.format(key + ':', width=indent2-1) + str(value) + '\n'
        else:
            ret += indent1 + '{:{width}}'.format(key + ':', width=indent2) + str(value) + '\n'
    return ret
