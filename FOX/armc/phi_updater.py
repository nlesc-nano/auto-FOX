"""
FOX.armc.phi_updater
====================

A module holding classes for managing and updating :math:`\phi`.

Index
-----
.. currentmodule:: FOX.armc.phi_updater
.. autosummary::
    PhiUpdater
    MultiPhiUpdater

API
---
.. autoclass:: PhiUpdater
.. autoclass:: MultiPhiUpdater

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Any, Optional, TypeVar, Union, Iterable, overload, Generic
from logging import Logger
from functools import partial, wraps
from collections import abc

import numpy as np

from assertionlib.dataclass import AbstractDataClass

from ..type_hints import ArrayLike, ArrayOrScalar, Scalar, Array

__all__ = ['PhiUpdater', 'MultiPhiUpdater']

AST1 = TypeVar('AST1', bound=ArrayOrScalar)
AST2 = TypeVar('AST2', bound=ArrayOrScalar)
AST3 = TypeVar('AST3', bound=ArrayOrScalar)

AT1 = TypeVar('AT1', bound=Array)
AT2 = TypeVar('AT2', bound=Array)
AT3 = TypeVar('AT3', bound=Array)

AT = TypeVar('AT', bound=Array)
ST = TypeVar('ST', bound=Scalar)
T = TypeVar('T')

IterableOrScalar = Union[Scalar, Iterable[Scalar]]

_PhiFunc = Callable[..., ArrayOrScalar]
PhiFunc = Callable[[ArrayOrScalar, ArrayOrScalar], ArrayOrScalar]


class PhiUpdaterABC(AbstractDataClass, ABC, Generic[AST1, AST2, AST3]):
    r"""A class for applying and updating :math:`\phi`.

    Has two main methods:

    * :meth:`__call__` for applying :attr:`phi` to the passed value.
    * :meth:`update` for updating the value of :attr:`phi`.

    Examples
    --------
    .. code:: python

        >>> import numpy as np

        >>> value = np.ndarray(...)
        >>> phi = PhiUpdater(...)

        >>> phi(value)
        >>> phi.update(...)

    Attributes
    ----------
    phi
        The variable :math:`\phi`.
    gamma
        The constant :math:`\gamma`.
    a_target
        The target acceptance rate :math:`\alpha_{t}`.
    func : :data:`~typing.Callable`
        The callable used for applying :math:`\phi` to the auxiliary error.
        The callable should take an array-like object and float as argument
        and return a numpy array.

    """

    phi: AST1
    gamma: AST2
    a_target: AST3

    def __init__(self, phi: AST1, gamma: AST2, a_target: AST3,
                 func: _PhiFunc, **kwargs: Any) -> None:
        r"""Initialize an :class:`AbstractPhiUpdater` instance.

        Parameters
        ----------
        phi
            The variable :math:`\phi`.
            See :attr:`AbstractPhiUpdater.phi`.
        gamma
            The constant :math:`\gamma`.
            See :attr:`AbstractPhiUpdater.gamma`.
        a_target
            The target acceptance rate :math:`\alpha_{t}`.
            See :attr:`AbstractPhiUpdater.a_target`.
        func : :data:`~typing.Callable`
            The callable used for applying :math:`\phi` to the auxiliary error.
            The callable should take an array-like object and float as argument
            and return a numpy array.
            See :attr:`AbstractPhiUpdater.func`.
        \**kwargs : :data:`~typing.Any`
            Further keyword arguments **func**

        """
        super().__init__()
        self.phi = phi
        self.gamma = gamma
        self.a_target = a_target
        self.func: PhiFunc = wraps(func)(partial(func, **kwargs))

    @staticmethod
    @AbstractDataClass.inherit_annotations()
    def _eq(v1, v2):
        if isinstance(v1, partial):
            names = ("func", "args", "keywords")
            return all([getattr(v1, n) == getattr(v2, n) for n in names])
        else:
            return v1 == v2

    @overload
    def __call__(self: PhiUpdaterABC[AT, Any, Any], value: AT, dtype: type = ...) -> AT: ...
    @overload
    def __call__(self: PhiUpdaterABC[ST, Any, Any], value: AT, dtype: type = ...) -> AT: ...
    @overload
    def __call__(self: PhiUpdaterABC[AT, Any, Any], value: ST, dtype: type = ...) -> AT: ...
    @overload
    def __call__(self: PhiUpdaterABC[ST, Any, Any], value: ST, dtype: type = ...) -> ST: ...
    def __call__(self, value, dtype=float):   # noqa: E301
        """Pass **value** and :attr:`phi` to :attr:`func`.

        Parameters
        ----------
        value : array-like or scalar
            A array-like object or a scalar.

        Returns
        -------
        :class:`numpy.ndarray` or scalar
            An array or a scalar.

        """
        phi = self.phi
        ar = np.asarray(value, dtype=float)
        return self.func(ar, phi)

    @abstractmethod
    def update(self, acceptance: ArrayLike, **kwargs: Any) -> None:
        r"""An abstract method for updating :attr:`phi` based on the values of :attr:`gamma` and **acceptance**.

        Parameters
        ----------
        acceptance : array-like [:class:`bool`]
            An array-like object consisting of booleans.
        \**kwargs : :data:`~typing.Any`
            Further keyword arguments which can be customized in the methods of subclasses.

        """  # noqa
        raise NotImplementedError("Trying to call an abstract method")


class PhiUpdater(PhiUpdaterABC, Generic[AST1, AST2, AST3]):

    @PhiUpdaterABC.inherit_annotations()
    def __init__(self, phi=1.0, gamma=2.0, a_target=0.25, func=np.add, **kwargs) -> None:
        super().__init__(phi, gamma, a_target, func, **kwargs)

    def update(self, acceptance: ArrayLike, logger: Optional[Logger] = None) -> None:
        r"""Update the variable :math:`\phi`.

        :math:`\phi` is updated based on the target accepatance rate, :math:`\alpha_{t}`, and the
        acceptance rate, **acceptance**, of the current super-iteration:

        .. math::

            \phi_{\kappa \omega} =
            \phi_{ ( \kappa - 1 ) \omega} * \gamma^{
                \text{sgn} ( \alpha_{t} - \overline{\alpha}_{ ( \kappa - 1 ) })
            }

        Parameters
        ----------
        acceptance : array-like [:class:`bool`]
            An array-like object consisting of booleans.
        logger : :class:`logging.Logger`, optional
            A logger for reporting the updated value.

        """
        mean_acceptance = np.mean(acceptance)
        sign = np.sign(self.a_target - mean_acceptance)

        phi = self.phi * self.gamma**sign
        if logger is not None:
            logger.info(f"Updating phi: {self.phi} -> {phi}")
        self.phi = phi


@overload
def _to_array(value: Scalar) -> np.ndarray: ...
@overload
def _to_array(value: Iterable[Scalar]) -> np.ndarray: ...
def _to_array(value):  # noqa: E302
    if isinstance(value, abc.Iterable):
        return np.fromiter(value, dtype=float)
    else:
        return np.array(value, ndmin=1, dtype=float)


class MultiPhiUpdater(PhiUpdater, Generic[AT1, AT2, AT3]):

    _phi: AT1
    _gamma: AT2
    _a_target: AT3

    @property
    def phi(self) -> AT1:
        return self._phi

    @phi.setter
    def phi(self, value: Union[Scalar, Iterable[Scalar]]):
        self._phi = _to_array(value)

    @property
    def gamma(self) -> AT2:
        return self._gamma

    @gamma.setter
    def gamma(self, value: Union[Scalar, Iterable[Scalar]]):
        self._gamma = _to_array(value)

    @property
    def a_target(self) -> AT3:
        return self._a_target

    @a_target.setter
    def a_target(self, value: Union[Scalar, Iterable[Scalar]]):
        self._a_target = _to_array(value)

    @PhiUpdater.inherit_annotations()
    def _str_iterator(self):
        items = sorted(vars(self).items())
        return ((k.strip('_'), v) for k, v in items if k not in self._PRIVATE_ATTR)

    def __init__(self, phi: IterableOrScalar = (1.0, 1.0),
                 gamma: IterableOrScalar = (2.0, 2.0),
                 a_target: IterableOrScalar = (0.25, 0.25),
                 func: PhiFunc = np.add, **kwargs: Any) -> None:
        super().__init__(phi, gamma, a_target, func, **kwargs)
        len_set = {len(getattr(self, name)) for name in ('phi', 'gamma', 'a_target')}
        if len(len_set) != 1:
            raise ValueError("'phi', 'gamma' and 'a_target' should all be of the same length")

    __init__.__doc__ = PhiUpdater.__init__.__doc__


PhiUpdater.__doc__ = PhiUpdaterABC.__doc__
MultiPhiUpdater.__doc__ = PhiUpdaterABC.__doc__
