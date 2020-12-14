"""A module containing the abstract :class:`FromResult` class."""

from __future__ import annotations

import inspect
from abc import ABCMeta, abstractmethod
from types import MappingProxyType
from typing import Generic, TypeVar, Any, Callable, Optional, ClassVar, overload, TYPE_CHECKING

import numpy as np
from nanoutils import Literal, TypedDict, ArrayLike
from qmflows.packages import Result

__all__ = ['FromResult']

T = TypeVar("T")
T1 = TypeVar("T1")
ST = TypeVar("ST", bound='FromResult[Any, Any]')
FT = TypeVar("FT", bound=Callable[..., Any])
RT = TypeVar("RT", bound=Result)


class ReductionDict(TypedDict):
    """A dictionary mapping keywords to callbacks."""

    min: Callable[[ArrayLike], np.float64]
    max: Callable[[ArrayLike], np.float64]
    mean: Callable[[ArrayLike], np.float64]
    sum: Callable[[ArrayLike], np.float64]
    product: Callable[[ArrayLike], np.float64]
    var: Callable[[ArrayLike], np.float64]
    std: Callable[[ArrayLike], np.float64]
    ptp: Callable[[ArrayLike], np.float64]
    all: Callable[[ArrayLike], np.bool_]
    any: Callable[[ArrayLike], np.bool_]


class FromResult(Generic[FT, RT], metaclass=ABCMeta):
    """A :class:`~collections.abc.Callable` wrapper.

    Besides :meth:`__call__`, instances have access to the :meth:`from_result` method,
    which is used for applying the wrapped callable to
    a :class:`qmflows.Result <qmflows.packages.Result>` instance.

    Parameters
    ----------
    func : :class:`Callable[..., Any] <collections.abc.Callable>`
        The to-be wrapped function.
    name : :class:`str`
        The :attr:`~object.__name__` of the to-be created instance.
    module : :class:`str`
        The :attr:`~object.__module__` of the to-be created instance.
        If :data:`None`, set it to ``"__main__"``.
    doc : :class:`str`, optional
        The :attr:`~object.__doc__` of the to-be created instance.
        If :data:`None`, extract the docstring from **func**.

    """

    #: A mapping that maps :meth:`from_result` aliases to callbacks.
    REDUCTION_NAMES: ClassVar[ReductionDict] = MappingProxyType({  # type: ignore[assignment]
        'min': np.min,
        'max': np.max,
        'mean': np.mean,
        'sum': np.sum,
        'product': np.product,
        'var': np.var,
        'std': np.std,
        'ptp': np.ptp,
        'all': np.all,
        'any': np.any,
    })

    def __init__(
        self,
        func: FT,
        name: str,
        module: Optional[str] = None,
        doc: Optional[str] = None,
    ) -> None:
        """Initialize the instance."""
        if not callable(func):
            raise TypeError("`func` expected a callable object")
        self._func = func

        cls = type(self)
        self.__name__ = name
        self.__qualname__ = name
        self.__module__ = module if module is not None else '__main__'

        if doc is None:
            self.__doc__: Optional[str] = getattr(func, '__doc__', None)
        else:
            self.__doc__ = doc

        try:
            self._hash = hash((cls, func))
        except TypeError:  # `func` may not be hashable in rare cases
            self._hash = hash((cls, id(func)))

        try:
            self.__annotations__ = func.__annotations__.copy()
        except AttributeError:
            self.__annotations__ = {'args': Any, 'kwargs': Any, 'return': Any}

        self.__signature__: Optional[inspect.Signature] = None
        self.__text_signature__: Optional[str] = None
        if hasattr(func, '__signature__'):
            self.__signature__ = func.__signature__  # type: ignore[attr-defined]
        elif getattr(func, '__text_signature__', None) is not None:
            self.__text_signature__ = func.__text_signature__  # type: ignore[attr-defined]
        else:
            try:
                self.__signature__ = inspect.signature(func)
            except ValueError:
                self.__signature__ = inspect.Signature([
                    inspect.Parameter('args', kind=inspect.Parameter.VAR_POSITIONAL),
                    inspect.Parameter('kwargs', kind=inspect.Parameter.VAR_KEYWORD),
                ])

    if TYPE_CHECKING:
        __call__: FT
    else:
        @property
        def __call__(self):
            """Get the underlying function."""
            return self._func

    def __hash__(self) -> int:
        """Implement :func:`hash(self) <hash>`."""
        return self._hash

    def __eq__(self, value: object) -> bool:
        """Implement :meth:`self == value <object.__eq__>`."""
        try:
            return hash(self) == hash(value)
        except TypeError:
            return False

    def __repr__(self) -> str:
        """Implement :class:`str(self) <str>` and :func:`repr(self) <repr>`."""
        cls = type(self)
        sgn = inspect.signature(self)
        return f'<FOX.{cls.__name__} instance {self.__module__}.{self.__name__}{sgn}>'

    @overload
    def from_result(self: FromResult[Callable[..., T], RT], result: RT, reduction: None = ...) -> T: ...  # noqa: E501
    @overload
    def from_result(self: FromResult[Callable[..., T], RT], result: RT, reduction: Callable[[T], T1]) -> T1: ...  # noqa: E501
    @overload
    def from_result(self, result: RT, reduction: Literal['min', 'max', 'mean', 'sum', 'product', 'var', 'std', 'ptp']) -> np.float64: ...  # noqa: E501
    @overload
    def from_result(self, result: RT, reduction: Literal['all', 'any']) -> np.bool_: ...  # noqa: E501
    @abstractmethod  # noqa: E301
    def from_result(self, result, reduction=None):
        r"""Call **self** using argument extracted from **result**.

        Parameters
        ----------
        result : :class:`qmflows.Result <qmflows.packages.Result>`
            The Result instance that **self** should operator on.
        reduction : :class:`str` or :class:`Callable[[Any], Any] <collections.abc.Callable>`, optional
            A callback for reducing the output of **self**.
            Alternativelly, one can provide on of the string aliases from :attr:`REDUCTION_NAMES`.
        \**kwargs : :data:`~typing.Any`
            Further keyword arguments for :meth:`__call__`.

        Returns
        -------
        :data:`~typing.Any`
            The output of :meth:`__call__`.

        """  # noqa: E501
        raise NotImplementedError("Trying to call an abstract method")

    @overload
    @classmethod
    def _reduce(cls, value: T, reduction: None) -> T: ...  # noqa: E501
    @overload
    @classmethod
    def _reduce(cls, value: T, reduction: Callable[[T], T1]) -> T1: ...  # noqa: E501
    @overload
    @classmethod
    def _reduce(cls, value: T, reduction: Literal['min', 'max', 'mean', 'sum', 'product', 'var', 'std', 'ptp']) -> np.float64: ...  # noqa: E501
    @overload
    @classmethod
    def _reduce(cls, value: T, reduction: Literal['all', 'any']) -> np.bool_: ...  # noqa: E501
    @classmethod  # noqa: E301
    def _reduce(cls, value, reduction):
        """A helper function to handle the reductions in :meth:`from_result`."""
        if reduction is None:
            return value
        elif callable(reduction):
            return reduction(value)

        try:
            func = cls.REDUCTION_NAMES[reduction]
        except (TypeError, KeyError):
            if not isinstance(reduction, str):
                raise ValueError("Expected a string; observed type: "
                                 f"{reduction.__class__.__name__!r}") from None
            else:
                raise ValueError(f"Invalid value: {reduction!r}") from None
        else:
            return func(value)
