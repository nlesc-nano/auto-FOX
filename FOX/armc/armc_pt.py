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
    Tuple, Dict, Mapping, Iterable, List, Sequence, overload, Any, TYPE_CHECKING
)

import numpy as np

from .armc import ARMC
from ..type_hints import ArrayOrScalar
from ..io.hdf5_utils import create_hdf5, create_xyz_hdf5

if TYPE_CHECKING:
    from ..classes import MultiMolecule
else:
    from ..type_alias import MultiMolecule

__all__ = ['ARMCPT']

PesDict = Dict[str, ArrayOrScalar]
PesMapping = Mapping[str, ArrayOrScalar]

MolList = List[MultiMolecule]
MolIter = Iterable[MultiMolecule]

Key = Tuple[float, ...]


class ARMCPT(ARMC):

    def __init__(self, **kwargs: Any) -> None:
        r"""Initialize an :class:`ARMCPT` instance.

        Parameters
        ----------
        \**kwargs : :data:`~typing.Any`
            Further keyword arguments for the :class:`ARMC` and
            :class:`MonteCarloABC` superclasses.

        """
        super().__init__(**kwargs)
        if len(self.phi.phi) <= 1:
            raise ValueError("{self.__class__.__name__!r} requires 'phi.phi' to "
                             "contain more than 1 element")
        elif len(self._molecule) > 1:
            raise NotImplementedError

    def acceptance(self) -> np.ndarray:
        """Create an empty 2D boolean array for holding the acceptance."""
        shape = (self.sub_iter_len, len(self.phi.phi))
        return np.zeros(shape, dtype=bool)

    @overload
    def _parse_call(self) -> List[Key]: ...
    @overload
    def _parse_call(self, start: None, key_new: None) -> List[Key]: ...
    @overload
    def _parse_call(self, start: int, key_new: Iterable[Key]) -> List[Key]: ...
    def _parse_call(self, start=None, key_new=None):  # noqa: E301
        """Parse the arguments of :meth:`__call__` and prepare the first key."""
        ret = super()._parse_call(start, key_new)
        if start is key_new is None:
            self[ret] = self[ret][0]
            i = len(self.param['param'].columns)
            return i * [ret]
        else:
            return list(ret)

    @overload
    def __call__(self) -> None: ...
    @overload
    def __call__(self, start: None, key_new: None) -> None: ...
    @overload
    def __call__(self, start: int, key_new: Key) -> None: ...
    def __call__(self, start=None, key_new=None):  # noqa: E301
        """Initialize the Addaptive Rate Monte Carlo procedure."""
        key_new = self._parse_call(start, key_new)
        start_ = start if start is not None else 0

        # Start the main loop
        for kappa in range(start_, self.super_iter_len):
            acceptance = self.acceptance()
            create_xyz_hdf5(self.hdf5_file, self.molecule,
                            iter_len=self.sub_iter_len, phi=self.phi.phi)

            for omega in range(self.sub_iter_len):
                key_new = self.do_inner(kappa, omega, acceptance, key_new)
            self.phi.update(acceptance)
            self.swap_phi(acceptance)

    def do_inner(self, kappa: int, omega: int, acceptance: np.ndarray,
                 key_old: Sequence[Key]) -> List[Key]:
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
        error_change, aux_new = self._do_inner3(pes_new, key_old)
        accept: np.ndarray = error_change < 0

        # Step 4: Update the auxiliary error history, apply phi & update job settings
        acceptance[omega] = accept
        key_new = self._do_inner4(accept, error_change, aux_new,
                                  _key_new, key_old, kappa, omega)

        # Step 5: Export the results to HDF5
        self._do_inner5(mol_list, accept, aux_new.T, pes_new, kappa, omega)
        return key_new

    def _do_inner3(self, pes_new: PesMapping,
                   key_old: Iterable[Key]) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the auxiliary error; accept if the new parameter set lowers the error."""
        aux_new = self.get_aux_error(pes_new)
        aux_old = np.array([self[k] for k in key_old])
        error_change = (aux_new - aux_old).sum(axis=-1)
        return error_change, aux_new

    def _do_inner4(self, accept: Iterable[bool], error_change: Iterable[bool],
                   aux_new: Iterable[np.ndarray],
                   key_new: Iterable[Key], key_old: Iterable[Key],
                   kappa: int, omega: int) -> List[Key]:
        """Update the auxiliary error history, apply phi & update job settings."""
        ret = []
        enumerator = enumerate(zip(key_new, key_old, accept, error_change, aux_new))
        for i, (k_new, k_old, acc, err_change, _aux_new) in enumerator:
            err_round = round(err_change, 4)
            aux_round = round(_aux_new.sum(), 4)
            newline = '\n' if i == len(self.phi) - 1 else ''
            epilog = f'total error change / error: {err_round} / {aux_round}{newline}'

            if acc:
                self.logger.info(f"Accepting move {(kappa, omega)}; {epilog}")
                self[k_new] = self.apply_phi(_aux_new)
                self.param['param_old'][i] = self.param['param'][i]
                ret.append(k_new)
            else:
                self.logger.info(f"Rejecting move {(kappa, omega)}; {epilog}")
                self[k_new] = _aux_new
                self[k_old] = self.apply_phi(self[k_old])
                ret.append(k_old)

        return ret

    def swap_phi(self, acceptance: np.ndarray) -> None:
        r"""Swap the :math:`\phi` and move range of two between forcefield parameter sets.

        The two main parameters are the acceptance rate
        :math:`\boldsymbol{\alpha} \in \mathbb{R}^{(n,)}` and target acceptance rate
        :math:`\boldsymbol{\alpha^{t}} \in \mathbb{R}^{(n,)}`,
        both vector's elements being the :math:`[0,1]` interval.

        The (weighted) probability distribution :math:`\boldsymbol{p} \in \mathbb{R}^{(n,)}` is
        consequently used for identifying which two forcefield parameter sets
        are swapped.

        .. math::

            \hat{\boldsymbol{p}} = |\boldsymbol{\alpha} - \boldsymbol{\alpha^{t}}|^{-1}

            \boldsymbol{p} = \frac{\hat{\boldsymbol{p}}}{\sum^n_{i} {\hat{\boldsymbol{p}}}_{i}}

        Parameters
        ----------
        acceptance : :class:`numpy.ndarray` [:class:`bool`], shape :math:`(n, m)`
            A 2D boolean array with acceptance rates over
            the course of the last super-iteration.

        """
        _p = acceptance.mean(axis=-1) - self.phi.a_target
        if 0 in _p:
            p = np.zeros_like(_p, dtype=float)
            p[_p == 0] = 1
        else:
            p = _p**-1
        p /= p.sum()  # normalize

        idx_range = np.arange(len(acceptance))
        idx1 = np.random.choice(idx_range, p=p)
        idx2 = np.random.choice(idx_range, p=p)

        if idx1 != idx2:
            self._swap_phi(idx1, idx2)

    def _swap_phi(self, idx1: int, idx2: int) -> None:
        """Swap the array-elements **idx1** and **idx2** of four :class:`ARMCPT` attributes.

        Affects the following attributes:

        * :attr:`ARMCPT.phi.phi<FOX.armc.PhiUpdater.phi>`
        * :attr:`ARMCPT.phi.a_target<FOX.armc.PhiUpdater.a_target>`
        * :attr:`ARMCPT.phi.gamma<FOX.armc.PhiUpdater.gamma>`
        * :attr:`ARMCPT.param.move_range<FOX.armc.ParamMapping.move_range>`

        """
        i = [idx1, idx2]
        j = [idx2, idx1]

        self.phi.phi[i] = self.phi.phi[j]
        self.phi.a_target[i] = self.phi.a_target[j]
        self.phi.gamma[i] = self.phi.gamma[j]
        self.param.move_range[i] = self.param.move_range[j]
