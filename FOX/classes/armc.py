"""A module for performing Addaptive Rate Monte Carlo (ARMC) forcefield parameter optimizations."""

from __future__ import annotations

from os.path import (isfile, split)

from typing import (Tuple, Dict)
import numpy as np
import pandas as pd

from scm.plams import Settings, Molecule
from scm.plams.core.functions import (init, finish, config)

from .psf_dict import PSFDict
from .monte_carlo import MonteCarlo
from ..io.hdf5_utils import (create_hdf5, to_hdf5)
from ..functions.utils import (get_template, get_class_name, get_func_name)
from ..functions.cp2k_utils import set_subsys_kind
from ..functions.armc_sanitization import init_armc_sanitization

__all__ = ['ARMC']


class ARMC(MonteCarlo):
    """The Addaptive Rate Monte Carlo class (:class:`.ARMC`), a subclass of the base
    :class:`.MonteCarlo` class.

    :Atributes:     * **armc** (|plams.Settings|_) – See :meth:`ARMC.reconfigure_armc_atr`

                    * **phi** (|plams.Settings|_) – See :meth:`ARMC.reconfigure_phi_atr`
    """

    def __init__(self,
                 molecule: Molecule,
                 param: pd.DataFrame,
                 **kwarg: dict) -> None:
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

    def __str__(self) -> str:
        ret = Settings(vars(self))

        # The self.pes block
        for value in ret.pes.values():
            value.ref = str(value.ref.__class__)
            value.kwarg = str(value.kwarg.as_dict())
            value.func = get_func_name(value.func) + '()'

        # The self.job block
        ret.job.molecule = str(self.job.molecule.__class__)
        ret.job.settings = str(self.job.settings.__class__)
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

        :parameter str filename: The path+filename of a .yaml file containing all :class:`ARMC`
            settings.
        :return: A :class:`ARMC` instance.
        :rtype: |FOX.ARMC|_
        """
        if isfile(filename):
            path, filename = split(filename)
            return ARMC.from_dict(get_template(filename, path=path))
        else:
            return ARMC.from_dict(get_template(filename))

    @classmethod
    def from_dict(cls, dict_: Settings) -> ARMC:
        """Create a :class:`.ARMC` instance from a dictionary.

        :parameter dict dict_: A dictionary containing all :class:`ARMC` settings.
        :return: A :class:`ARMC` instance.
        :rtype: |FOX.ARMC|_
        """
        molecule, param, dict_ = init_armc_sanitization(dict_)
        set_subsys_kind(dict_.job.settings, dict_.job.psf['atoms'])
        molecule = molecule.as_Molecule(-1)[0]
        return cls(molecule, param, **dict_)

    def init_armc(self) -> None:
        """Initialize the Addaptive Rate Monte Carlo procedure."""
        # Unpack attributes
        super_iter = self.armc.iter_len // self.armc.sub_iter_len

        # Construct the HDF5 file
        create_hdf5(self.hdf5_file, self)

        # Initialize
        init(path=self.job.path, folder=self.job.folder)
        config.default_jobmanager.settings.hashing = None
        if self.job.logfile:
            config.default_jobmanager.logfile = self.job.logfile
            config.log.file = 3
        if self.job.psf[0]:
            PSFDict.write_psf(self.job.psf)

        # Initialize the first MD calculation
        history_dict: dict = {}
        key_new = tuple(self.param['param'].values)
        pes_new, mol = self.get_pes_descriptors(history_dict, key_new)
        history_dict[key_new] = self.get_aux_error(pes_new)
        self.param['param_old'] = self.param['param']

        # Start the main loop
        for kappa in range(super_iter):
            key_new = self.do_inner(kappa, history_dict, key_new)
        finish()

    def do_inner(self,
                 kappa: float,
                 history_dict: Dict[Tuple[float], np.ndarray],
                 key_new: Tuple[float]) -> Tuple[float]:
        r"""A method that handles the inner loop of the :meth:`ARMC.init_armc` method.

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
        hdf5_kwarg = {}
        acceptance = np.zeros(self.armc.sub_iter_len, dtype=bool)

        for omega in range(self.armc.sub_iter_len):
            # Step 1: Perform a random move
            key_old = key_new
            key_new = self.move_param()

            # Step 2: Check if the move has been performed already; calculate PES descriptors if not
            pes_new, mol = self.get_pes_descriptors(history_dict, key_new)
            hdf5_kwarg.update(pes_new)

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
                hdf5_kwarg['aux_error_mod'] = np.append(self.param['param'].values, self.phi.phi)
            else:
                history_dict[key_old] = self.apply_phi(aux_old)
                key_new = key_old
                self.param['param'] = self.param['param_old']
                hdf5_kwarg['aux_error_mod'] = np.append(self.param['param_old'].values,
                                                        self.phi.phi)

            # Step 5: Export the results to HDF5
            hdf5_kwarg['xyz'] = mol if mol is not None else np.nan
            hdf5_kwarg['phi'] = self.phi.phi
            hdf5_kwarg['param'] = self.param['param']
            hdf5_kwarg['acceptance'] = accept
            hdf5_kwarg['aux_error'] = aux_new
            to_hdf5(self.hdf5_file, hdf5_kwarg, kappa, omega)

        self.update_phi(acceptance)
        return key_new

    def get_aux_error(self, pes_dict: Dict[str, np.ndarray]) -> np.ndarray:
        r"""Return the auxiliary error, :math:`\Delta \varepsilon_{QM-MM}`, of the PES descriptors
        in **values** with respect to **self.ref**.


        The default is equivalent to:

        .. math::

            \Delta \varepsilon_{QM-MM} =
            \sqrt {
                \frac{1}{N}
                \sum_{i}^{N}
                \left(
                    \frac{ r_{i}^{QM} - r_{i}^{MM} }
                    {r_{i}^{QM}}
                \right )^2
            }

        :parameter pes_dict: A dictionary of *n* PES descriptors.
        :type pes_dict: *n* |dict|_ (keys: |str|_, values: |np.ndarray|_ [|np.float64|_])
        :return: An array with *n* auxilary errors
        :rtype: *n* |np.ndarray|_ [|np.float64|_]
        """
        def norm_mean(mm_pes: np.ndarray, key: str) -> float:
            qm_pes = self.pes[key].ref
            A, B = np.asarray(qm_pes), np.asarray(mm_pes)
            ret = (A - B)**2
            return ret.sum() / A.sum()

        return np.array([norm_mean(mm_pes, key) for key, mm_pes in pes_dict.items()])

    def apply_phi(self, aux_error: float) -> float:
        r"""Apply :math:`\phi` to all auxiliary errors, :math:`\Delta \varepsilon_{QM-MM}`,
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

    def update_phi(self, acceptance: np.ndarray) -> None:
        r"""Update :math:`\phi` based on the target accepatance rate, :math:`\alpha_{t}`, and the
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


def _str(dict_: dict,
         indent1: str = '') -> str:
    ret = ''
    indent2 = 3 + max(len(i) for i in dict_.keys())
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
