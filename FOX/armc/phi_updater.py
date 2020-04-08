r"""
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
from typing import Callable, Any, Optional, Union, Iterable
from logging import Logger
from functools import partial, wraps

import numpy as np

from assertionlib.dataclass import AbstractDataClass

from ..type_hints import ArrayLike, ArrayLikeOrScalar, ArrayOrScalar, Scalar
from ..functions.utils import as_nd_array

__all__ = ['PhiUpdater']

_PhiFunc = Callable[..., np.ndarray]
PhiFunc = Callable[[np.ndarray, ArrayOrScalar], np.ndarray]


class PhiUpdaterABC(AbstractDataClass, ABC):
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

    phi: np.ndarray
    gamma: np.ndarray
    a_target: np.ndarray

    def __init__(self, phi: Union[Scalar, Iterable[Scalar], ArrayLike],
                 gamma: Union[Scalar, Iterable[Scalar], ArrayLike],
                 a_target: Union[Scalar, Iterable[Scalar], ArrayLike],
                 func: _PhiFunc, **kwargs: Any) -> None:
        r"""Initialize an :class:`AbstractPhiUpdater` instance.

        Parameters
        ----------
        phi : array-like [:class:`float`]
            The variable :math:`\phi`.
            See :attr:`AbstractPhiUpdater.phi`.

        gamma : array-like [:class:`float`]
            The constant :math:`\gamma`.
            See :attr:`AbstractPhiUpdater.gamma`.

        a_target : array-like [:class:`float`]
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

        array = partial(as_nd_array, dtype=float, ndmin=1)
        self.phi = array(phi)
        self.gamma = array(gamma)
        self.a_target = array(a_target)
        self._validate_shape()

        self.func: PhiFunc = wraps(func)(partial(func, **kwargs))

    def _validate_shape(self):
        """Ensure that :attr:`phi`, :attr:`gamma` and :attr:`a_target` all have the same shape."""
        names = ('phi', 'gamma', 'a_target')
        shape_set = {getattr(self, name).shape for name in names}
        if len(shape_set) != 1:
            raise ValueError("'phi', 'gamma', 'a_target' should all be of the same shape")

    @staticmethod
    @AbstractDataClass.inherit_annotations()
    def _eq(v1, v2):
        if isinstance(v1, partial):
            names = ("func", "args", "keywords")
            return all([getattr(v1, n) == getattr(v2, n, None) for n in names])
        else:
            return np.all(v1 == v2)

    def __call__(self, value: ArrayLikeOrScalar,
                 dtype: Union[type, np.dtype] = float) -> np.ndarray:
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


class PhiUpdater(PhiUpdaterABC):

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
        mean_acceptance = np.mean(acceptance, axis=0)
        sign = np.sign(self.a_target - mean_acceptance)

        phi = self.phi * self.gamma**sign
        if logger is not None:
            logger.info(f"Updating phi: {self.phi} -> {phi}")
        self.phi = phi


PhiUpdater.__doc__ = PhiUpdaterABC.__doc__
