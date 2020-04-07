"""
FOX.armc.armc_pt
================

A module for performing Addaptive Rate Monte Carlo Parallel Tempering (ARMC-PT)
forcefield parameter optimizations.

Index
-----
.. currentmodule:: FOX.armc.armc_pt
.. autosummary::
    ARMCPT

API
---
.. autoclass:: ARMCPT
    :members:
    :private-members:
    :special-members:

"""

from typing import (
    Generic, TypeVar, Tuple, Dict, Mapping, Iterable, List, Sequence, overload, Optional,
    TYPE_CHECKING
)

import numpy as np

from .armc import ARMC
from ..type_hints import ArrayOrScalar
from ..io.hdf5_utils import create_hdf5

if TYPE_CHECKING:
    from ..classes import MultiMolecule
else:
    from ..type_alias import MultiMolecule

__all__ = ['ARMCPT']

KT = TypeVar('KT', bound=Tuple[float, ...])
VT = TypeVar('VT', bound=np.ndarray)

PesDict = Dict[str, ArrayOrScalar]
PesMapping = Mapping[str, ArrayOrScalar]

MolList = List[MultiMolecule]
MolIter = Iterable[MultiMolecule]


class ARMCPT(ARMC, Generic[KT, VT]):
    def acceptance(self) -> np.ndarray:
        """Create an empty 2D boolean array for holding the acceptance."""
        shape = (self.sub_iter_len, len(self.phi.phi))
        return np.zeros(shape, dtype=bool)

    @overload  # type: ignore
    def _parse_call(self, start: None = ..., key_new: None = ...) -> List[KT]: ...
    @overload  # noqa: E301
    def _parse_call(self, start: int = ..., key_new: Iterable[KT] = ...) -> List[KT]: ...
    def _parse_call(self, start=None, key_new=None):  # noqa: E301
        """Parse the arguments of :meth:`__call__` and prepare the first key."""
        if start is None:
            create_hdf5(self.hdf5_file, self)  # Construct the HDF5 file

            key_new = [self._get_first_key(i) for i in self.param['param'].columns]
            if np.inf in np.array([self[k] for k in key_new]):
                raise RuntimeError('One or more jobs crashed in the first ARMC iteration; '
                                   'manual inspection of the cp2k output is recomended')
            elif not self.keep_files:
                self.clear_jobs()

        elif key_new is None:
            raise TypeError("'key_new' cannot be None if 'start' is None")
        else:
            return list(key_new)
        return key_new

    def do_inner(self, kappa: int, omega: int, acceptance: np.ndarray,
                 key_old: Sequence[KT]) -> List[KT]:
        r"""Run the inner loop of the :meth:`ARMC.__call__` method.

        Parameters
        ----------
        kappa : int
            The super-iteration, :math:`\kappa`, in :meth:`ARMC.__call__`.

        omega : int
            The sub-iteration, :math:`\omega`, in :meth:`ARMC.__call__`.

        history_dict : |dict|_ [|tuple|_ [|float|_], |np.ndarray|_ [|np.float64|_]]
            A dictionary with parameters as keys and a list of PES descriptors as values.

        key_new : tuple [float]
            A tuple with the latest set of forcefield parameters.

        Returns
        -------
        |tuple|_ [|float|_]:
            The latest set of parameters.

        """
        # Step 1: Perform a random move
        _key_new = [self._do_inner1(key, i) for i, key in enumerate(key_old)]

        # Step 2: Calculate PES descriptors
        pes_new, mol_list = self._do_inner2()

        # Step 3: Evaluate the auxiliary error; accept if the new parameter set lowers the error
        error_change, aux_new = self._do_inner3(pes_new, _key_new)
        accept: np.ndarray = error_change < 0

        # Step 4: Update the auxiliary error history, apply phi & update job settings
        acceptance[omega] = accept
        key_new = self._do_inner4(accept, error_change, aux_new,
                                  _key_new, key_old, kappa, omega)

        # Step 5: Export the results to HDF5
        self._do_inner5(mol_list, accept, aux_new, pes_new, kappa, omega)

        # Step 6: Allow for swapping between parameter sets
        key_main, *key_other = key_new
        key_new_swap = self._do_inner6(key_main, key_other)
        return key_new_swap

    def _do_inner3(self, pes_new: PesMapping,
                   key_old: Sequence[KT]) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the auxiliary error; accept if the new parameter set lowers the error."""
        error_change = []
        aux_new = []

        for i, key in enumerate(key_old):
            _pes_new = {k: v for k, v in pes_new.items() if k.endswith(str(i))}
            _aux_new = self.get_aux_error(_pes_new)
            _aux_old = self[key]

            error_change.append((_aux_new - _aux_old).sum())
            aux_new.append(_aux_new)
        return np.array(aux_new), np.array(aux_new)

    def _do_inner4(self, accept: np.ndarray, error_change: np.ndarray, aux_new: np.ndarray,
                   key_new: Sequence[KT], key_old: Sequence[KT],
                   kappa: int, omega: int) -> List[KT]:
        """Update the auxiliary error history, apply phi & update job settings."""
        ret = []

        enumerator = enumerate(zip(key_new, key_old, error_change, aux_new))
        for i, (k_new, k_old, err_change, _aux_new) in enumerator:
            err_round = round(err_change, 4)
            aux_round = round(_aux_new.sum(), 4)
            epilog = f'total error change / error: {err_round} / {aux_round}\n'

            if accept:
                self.logger.info(f"Accepting move {(kappa, omega)}; {epilog}")
                self[k_new] = self.phi(_aux_new)
                self.param['param_old'][i] = self.param['param'][i]
                ret.append(k_new)
            else:
                self.logger.info(f"Rejecting move {(kappa, omega)}; {epilog}")
                self[k_new] = _aux_new
                self[k_old] = self.apply_phi(self[key_old])
                ret.append(k_old)

        return ret

    def _do_inner6(self, key_main: KT, key_other: Sequence[KT]) -> List[KT]:
        """Allow for swapping between parameter sets."""
        err_main = self[key_main]
        err_other = np.array([self[k] for k in key_other])
        swap = self.swap_parameters(err_main, err_other)

        if swap is None:
            return key_other
        pass

    @staticmethod
    def swap_parameters(err_main: np.ndarray, err_other: np.ndarray) -> Optional[Tuple[int, int]]:
        pass
