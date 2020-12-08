r"""A module holding classes for managing and updating :math:`\phi`.

Index
-----
.. currentmodule:: FOX.armc
.. autosummary::
    PhiUpdaterABC
    PhiUpdater

API
---
.. autoclass:: PhiUpdaterABC
    :members:
.. autoclass:: PhiUpdater
    :members:

"""

from abc import ABC, abstractmethod
from typing import Callable, Any, Optional, Union, Iterable, cast, Sized, Tuple, List, Dict
from logging import Logger
from functools import partial, wraps

import numpy as np

from assertionlib.dataclass import AbstractDataClass
from nanoutils import set_docstring, SupportsIndex, as_nd_array, TypedDict

from ..type_hints import ArrayLike, ArrayLikeOrScalar, Scalar, DtypeLike

__all__ = ['PhiUpdaterABC', 'PhiUpdater']

_PhiFunc = Callable[..., np.ndarray]
PhiFunc = Callable[[ArrayLikeOrScalar, np.ndarray], np.ndarray]

IterOrArrayLike = Union[Scalar, Iterable[Scalar], ArrayLike]


class _PhiDict(TypedDict):
    type: str
    phi: List[float]
    gamma: List[float]
    a_target: List[float]
    func: str
    kwargs: Dict[str, Any]


class PhiUpdaterABC(AbstractDataClass, ABC, Sized):
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
    phi : :class:`np.ndarray[np.float64] <numpy.ndarray>`
        The variable :math:`\phi`.
    gamma : :class:`np.ndarray[np.float64] <numpy.ndarray>`
        The constant :math:`\gamma`.
    a_target : :class:`np.ndarray[np.float64] <numpy.ndarray>`
        The target acceptance rate :math:`\alpha_{t}`.
    func : :class:`Callable[[array-like, ndarray], ndarray]<collections.abc.Callable>`
        The callable used for applying :math:`\phi` to the auxiliary error.
        The callable should take an array-like object and a :class:`numpy.ndarray`
        as arguments and return a new array.

    """

    @property
    def phi(self) -> np.ndarray:
        return self._phi

    @phi.setter
    def phi(self, value: IterOrArrayLike) -> None:
        self._phi = as_nd_array(value, dtype=float, ndmin=1)

    @property
    def gamma(self) -> np.ndarray:
        return self._gamma  # type: ignore

    @property
    def a_target(self) -> np.ndarray:
        return self._a_target  # type: ignore

    @property
    def func(self) -> PhiFunc:
        return self._func  # type: ignore

    _PRIVATE_ATTR = frozenset({'__name__', '__qualname__'})  # type: ignore

    def __init__(self, phi: IterOrArrayLike,
                 gamma: IterOrArrayLike,
                 a_target: IterOrArrayLike,
                 func: _PhiFunc, **kwargs: Any) -> None:
        r"""Initialize an :class:`AbstractPhiUpdater` instance.

        Parameters
        ----------
        phi : :term:`ArrayLike[np.float64] <numpy:array_like>`
            The variable :math:`\phi`.
            See :attr:`AbstractPhiUpdater.phi`.
        gamma : :term:`ArrayLike[np.float64] <numpy:array_like>`
            The constant :math:`\gamma`.
            See :attr:`AbstractPhiUpdater.gamma`.
        a_target : :term:`ArrayLike[np.float64] <numpy:array_like>`
            The target acceptance rate :math:`\alpha_{t}`.
            See :attr:`AbstractPhiUpdater.a_target`.
        func : :class:`Callable[[array-like, ndarray, **kwargs], ndarray]<collections.abc.Callable>`
            The callable used for applying :math:`\phi` to the auxiliary error.
            The callable should take an array-like object and a :class:`numpy.ndarray`
            as arguments and return a new array.
            See :attr:`AbstractPhiUpdater.func`.
        \**kwargs : :data:`~typing.Any`
            Further keyword arguments for **func**

        """
        super().__init__()
        cls = type(self)
        self.__name__: str = cls.__name__
        self.__qualname__: str = cls.__qualname__

        self.phi = cast(np.ndarray, phi)
        self._gamma = as_nd_array(gamma, dtype=float, ndmin=1, copy=True)
        self._a_target = as_nd_array(a_target, dtype=float, ndmin=1, copy=True)

        self._validate_shape()
        self._func = wraps(func)(partial(func, **kwargs))

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
            return all([np.all(getattr(v1, n) == getattr(v2, n, None)) for n in names])
        else:
            return np.all(v1 == v2)

    @AbstractDataClass.inherit_annotations()
    def _str_iterator(self):
        ret = ((k.strip('_'), v) for k, v in self._iter_attrs() if k not in self._PRIVATE_ATTR)
        return sorted(ret)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the :attr:`~numpy.ndarray.shape` of :attr:`phi`.

        Serves as a wrapper around the :attr:`~numpy.ndarray.shape` attribute of :attr:`phi`.
        Note that :attr:`phi`, :attr:`gamma` and :attr:`a_target` all have the same shape.
        """
        return self.phi.shape

    def __len__(self) -> int:
        """Implement :code:`len(self)`.

        Serves as a wrapper around the :meth:`~numpy.ndarray.__len__` method of :attr:`phi`.
        Note that :attr:`phi`, :attr:`gamma` and :attr:`a_target` are all of the same length.
        """
        return len(self.phi)

    def __call__(self, value: ArrayLikeOrScalar, *,
                 idx: Optional[SupportsIndex] = None,
                 dtype: DtypeLike = np.float64) -> np.ndarray:
        """Pass **value** and :attr:`phi` to :attr:`func`.

        Parameters
        ----------
        value : :term:`ArrayLike[np.bool_] <numpy:array_like>`
            A array-like object or a scalar.
        idx : :class:`int`, optional
            If not :data:`None`, apply :attr:`func` to **value** using :attr:`phi[idx]<phi>`.
        dtype : :class:`numpy.dtype`, optional
            The desired data-type for the output array.

        Returns
        -------
        :class:`numpy.ndarray`
            An array.

        """
        phi = self.phi
        ar = np.asarray(value, dtype=float)
        if idx is None:
            return self.func(ar, phi)
        else:
            return self.func(ar, phi[idx])

    def to_yaml_dict(self) -> _PhiDict:
        """Convert this instance into a .yaml-compatible dictionary."""
        cls = type(self)
        func = cast('partial[np.ndarray]', self.func)
        try:
            if isinstance(func.func, np.ufunc):
                module = 'numpy'
            else:
                module = func.func.__module__
            name = getattr(func.func, '__qualname__', func.func.__name__)
        except AttributeError as ex:
            raise TypeError(f"Failed to parse {cls.__name__}.func.func: {func.func!r}") from ex

        return {
            'type': f'{cls.__module__}.{cls.__name__}',
            'phi': self.phi.tolist(),
            'gamma': self.gamma.tolist(),
            'a_target': self.a_target.tolist(),
            'func': f'{module}.{name}',
            'kwargs': func.keywords,
        }

    @abstractmethod
    def update(self, acceptance: ArrayLike, **kwargs: Any) -> None:
        r"""An abstract method for updating :attr:`phi` based on the values of :attr:`gamma` and **acceptance**.

        Parameters
        ----------
        acceptance : :term:`ArrayLike[np.bool_] <numpy:array_like>`
            An array-like object consisting of booleans.
        \**kwargs : :data:`~typing.Any`
            Further keyword arguments which can be customized in the methods of subclasses.

        """  # noqa
        raise NotImplementedError("Trying to call an abstract method")


@set_docstring(PhiUpdaterABC.__doc__)
class PhiUpdater(PhiUpdaterABC):

    @PhiUpdaterABC.inherit_annotations()
    def __init__(self, phi=1.0, gamma=2.0, a_target=0.25, func=np.add, **kwargs) -> None:
        super().__init__(phi, gamma, a_target, func, **kwargs)

    def update(self, acceptance: ArrayLike, *, logger: Optional[Logger] = None) -> None:  # type: ignore[override] # noqa: E501
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
        acceptance : :term:`ArrayLike[np.bool_] <numpy:array_like>`
            An array-like object consisting of booleans.
        logger : :class:`logging.Logger`, optional
            A logger for reporting the updated value.

        """
        mean_acceptance: np.ndarray = np.mean(acceptance, axis=0)
        sign = cast(np.ndarray, np.sign(self.a_target - mean_acceptance))

        phi = self.phi * self.gamma**sign
        if phi.shape != self.shape:
            raise ValueError(f"Incorrect new 'phi' shape ({phi.shape!r}); "
                             f"expected shape: {self.shape!r}")
        elif logger is not None:
            logger.info(f"Updating phi: {self.phi} -> {phi}")
        self.phi = phi
