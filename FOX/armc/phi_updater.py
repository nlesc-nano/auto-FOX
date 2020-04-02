from abc import ABC, abstractmethod
from functools import partial, wraps
from typing import Callable, Any, Optional, TypeVar, Union
from logging import Logger

import numpy as np

from assertionlib.dataclass import AbstractDataClass

from ..type_hints import ArrayLikeOrScalar, ArrayLike

__all__ = ['PhiUpdater']

AT = TypeVar('AT', bound=np.ndarray)
PhiFunc = Callable[[Union[ArrayLikeOrScalar, AT], float], AT]


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

    phi: float
    gamma: float
    a_target: float
    # func: PhiFunc

    def __init__(self, phi: float, gamma: float, a_target: float,
                 func: PhiFunc, **kwargs: Any) -> None:
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
        self.func = wraps(func)(partial(func, **kwargs))

    def __call__(self, value: Union[AT, ArrayLikeOrScalar], dtype: type = float) -> AT:
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
    def __init__(self, phi=1.0, gamma=2.0, a_target=0.25,
                 func=np.add, **kwargs) -> None:
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

        phi = self.phi
        phi *= self.gamma**sign
        if logger is not None:
            logger.info(f"Updating phi: {self.phi} -> {phi}")
        self.phi = phi


PhiUpdater.__doc__ = PhiUpdaterABC.__doc__
