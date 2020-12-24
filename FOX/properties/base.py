"""A module containing the abstract :class:`FromResult` class."""

from __future__ import annotations

import inspect
import textwrap
from abc import ABCMeta, abstractmethod
from types import MappingProxyType, ModuleType
from typing import Generic, TypeVar, Any, Callable, Dict, Union

import scipy.special
import numpy as np
from qmflows.packages import Result

__all__ = ['FromResult', 'get_attr', 'call_method']

FT = TypeVar("FT", bound=Callable[..., Union[np.ndarray, np.generic]])
RT = TypeVar("RT", bound=Result)


def _gather_ufuncs(module: ModuleType) -> Dict[str, Callable[..., Any]]:
    """Gather a dictionary with all :class:`~numpy.ufunc.reduce`-supporting :class:`ufuncs <numpy.ufunc>` from the passed **module**."""  # noqa: E501
    iterator = (getattr(module, name) for name in getattr(module, '__all__', []))
    condition = lambda ufunc: (isinstance(ufunc, np.ufunc) and ufunc.signature is None and
                               ufunc.nin == 2 and ufunc.nout == 1)
    return {ufunc.__name__: ufunc.reduce for ufunc in iterator if condition(ufunc)}


class FromResult(Generic[FT, RT], metaclass=ABCMeta):
    """An abstract base class for wrapping :class:`~collections.abc.Callable` objects.

    Besides :meth:`__call__`, instances have access to the :meth:`from_result` method,
    which is used for applying the wrapped callable to
    a :class:`qmflows.Result <qmflows.packages.Result>` instance.

    Parameters
    ----------
    func : :class:`Callable[..., Any] <collections.abc.Callable>`
        The to-be wrapped function.
    name : :class:`str`
        The :attr:`~definition.__name__` attribute of the to-be created instance.
    module : :class:`str`
        The ``__module__`` attribute of the to-be created instance.
        If :data:`None`, set it to ``"__main__"``.
    doc : :class:`str`, optional
        The ``__doc__`` attribute of the to-be created instance.
        If :data:`None`, extract the docstring from **func**.

    """

    #: A mapping that maps :meth:`from_result` aliases to callbacks.
    REDUCTION_NAMES = MappingProxyType({
        'min': np.min,
        'max': np.max,
        'mean': np.mean,
        'sum': np.sum,
        'product': np.product,
        'var': np.var,
        'std': np.std,
        'ptp': np.ptp,
        'norm': np.linalg.norm,
        'argmin': np.argmin,
        'argmax': np.argmax,
        'all': np.all,
        'any': np.any,
        **_gather_ufuncs(np),
        **_gather_ufuncs(scipy.special),
    })

    def __init__(self, func, name, module=None, doc=None):
        """Initialize the instance."""
        if not callable(func):
            raise TypeError("`func` expected a callable object")
        self._func = func

        cls = type(self)
        self.__name__ = name
        self.__qualname__ = name
        self.__module__ = module if module is not None else '__main__'

        if doc is None:
            self.__doc__ = getattr(func, '__doc__', None)
        else:
            self.__doc__ = doc
        if cls.__call__ is not FromResult.__call__ and cls.__call__.__doc__ is not None:
            doc_append = textwrap.indent(textwrap.dedent(cls.__call__.__doc__), 4 * ' ')
            if self.__doc__ is not None:
                self.__doc__ += doc_append
            else:
                self.__doc__ = doc_append

        try:
            self._hash = hash((cls, func))
        except TypeError:  # `func` may not be hashable in rare cases
            self._hash = hash((cls, id(func)))

        self.__text_signature__ = None
        if hasattr(func, '__signature__'):
            self.__signature__ = func.__signature__
        elif getattr(func, '__text_signature__', None) is not None:
            self.__signature__ = None
            self.__text_signature__ = func.__text_signature__
        else:
            try:
                self.__signature__ = inspect.signature(func)
            except ValueError:
                self.__signature__ = inspect.Signature([
                    inspect.Parameter('args', kind=inspect.Parameter.VAR_POSITIONAL),
                    inspect.Parameter('kwargs', kind=inspect.Parameter.VAR_KEYWORD),
                ])

        self.__globals__ = MappingProxyType(getattr(func, '__globals__', {}))
        self.__closure__ = getattr(self._func, '__closure__', None)
        self.__defaults__ = getattr(self._func, '__defaults__', None)
        self.__annotations__ = MappingProxyType(getattr(
            func, '__annotations__', {'args': Any, 'kwargs': Any, 'return': Any}
        ))
        kwd = getattr(self._func, '__kwdefaults__', None)
        self.__kwdefaults__ = MappingProxyType({}) if kwd is None else MappingProxyType(kwd)

    @property
    def __code__(self):
        """Get the :attr:`~types.FunctionType.__code__>` of the underlying function.

        Note
        ----
        This property is only available if the underlying function supports it.

        """
        try:
            return self._func.__code__
        except AttributeError as ex:
            raise NotImplementedError(str(ex)) from None

    @property
    def __get__(self):
        """Get the :attr:`~types.FunctionType.__get__>` method of the underlying function.

        Note
        ----
        This property is only available if the underlying function supports it.

        """
        try:
            return self._func.__get__
        except AttributeError as ex:
            raise NotImplementedError(str(ex)) from None

    @property
    def __call__(self):
        """Get the underlying function."""
        return self._func

    def __hash__(self):
        """Implement :func:`hash(self) <hash>`."""
        return self._hash

    def __eq__(self, value):
        """Implement :meth:`self == value <object.__eq__>`."""
        try:
            return hash(self) == hash(value)
        except TypeError:
            return False

    def __repr__(self):
        """Implement :class:`str(self) <str>` and :func:`repr(self) <repr>`."""
        cls = type(self)
        sgn = inspect.signature(self)
        return f'<{cls.__name__} instance {self.__module__}.{self.__name__}{sgn}>'

    def __reduce__(self):
        """A helper method for :mod:`pickle`."""
        cls = type(self)
        args = self._func, self.__name__, self.__module__
        return cls, args, self.__doc__

    def __setstate__(self, state):
        """A helper method for :meth:`__reduce__`."""
        self.__doc__ = state

    @abstractmethod
    def from_result(self, result, reduce=None, axis=None, **kwargs):
        r"""Call **self** using argument extracted from **result**.

        Parameters
        ----------
        result : :class:`qmflows.Result <qmflows.packages.Result>`
            The Result instance that **self** should operator on.
        reduce : :class:`str` or :class:`Callable[[Any], Any] <collections.abc.Callable>`, optional
            A callback for reducing the output of **self**.
            Alternativelly, one can provide on of the string aliases from :attr:`REDUCTION_NAMES`.
        axis : :class:`int` or :class:`Sequence[int] <collections.abc.Sequence>`, optional
            The axis along which the reduction should take place.
            If :data:`None`, use all axes.
        \**kwargs : :data:`~typing.Any`
            Further keyword arguments for :meth:`__call__`.

        Returns
        -------
        :data:`~typing.Any`
            The output of :meth:`__call__`.

        """  # noqa: E501
        raise NotImplementedError("Trying to call an abstract method")

    @classmethod
    def _reduce(cls, value, reduce, axis=None):
        """A helper function to handle the reductions in :meth:`from_result`."""
        if reduce is None:
            return value
        elif callable(reduce):
            return reduce(value)

        try:
            func = cls.REDUCTION_NAMES[reduce]
        except (TypeError, KeyError):
            if not isinstance(reduce, str):
                raise TypeError("`reduce` expected a string; observed type: "
                                f"{reduce.__class__.__name__!r}") from None
            else:
                raise ValueError(f"Invalid `reduce` value: {reduce!r}") from None
        else:
            return func(value, axis=axis)

    @staticmethod
    def _pop(dct, key, callback):
        """Attempt to :meth:`~dict.pop` **key** from **dct**, fall back to **callback** otherwise."""  # noqa: E501
        if key in dct:
            return dct.pop(key)
        else:
            return callback()


class _Null:
    """A singleton used as sentinel value in :func:`get_attr`."""

    _INSTANCE = None

    def __new__(cls):
        """Construct a new instance."""
        if cls._INSTANCE is None:
            cls._INSTANCE = super().__new__(cls)
        return cls._INSTANCE

    def __repr__(self):
        """Implement :class:`str(self) <self>` and :func:`repr(self) <repr>`."""
        return '<null>'


#: A singleton used as sentinel value by :func:`get_attr`.
_NULL = _Null()


def get_attr(obj, name, default=_NULL, reduce=None, axis=None):
    """:func:`gettattr` with support for keyword argument.

    Parameters
    ----------
    obj : :class:`object`
        The object in question.
    name : :class:`str`
        The name of the to-be extracted attribute.
    default : :class:`~typing.Any`
        An object that is to-be returned if **obj** does not have the **name** attribute.
    reduce : :class:`str` or :class:`Callable[[Any], Any] <collections.abc.Callable>`, optional
        A callback for reducing the extracted attribute.
        Alternativelly, one can provide on of the string aliases
        from :attr:`FromResult.REDUCTION_NAMES`.
    axis : :class:`int` or :class:`Sequence[int] <collections.abc.Sequence>`, optional
        The axis along which the reduction should take place.
        If :data:`None`, use all axes.

    Returns
    -------
    :class:`~typing.Any`
        The extracted attribute.

    See Also
    --------
    :func:`getattr`
        Get a named attribute from an object.

    """
    if default is _NULL:
        ret = getattr(obj, name)
    ret = getattr(obj, name, default)
    return FromResult._reduce(ret, reduce, axis)


def call_method(obj, name, *args, reduce=None, axis=None, **kwargs):
    r"""Call the **name** method of **obj**.

    Parameters
    ----------
    obj : :class:`object`
        The object in question.
    name : :class:`str`
        The name of the to-be extracted method.
    \*args/\**kwargs : :class:`~typing.Any`
        Positional and/or keyword arguments for the (to-be called) extracted method.
    reduce : :class:`str` or :class:`Callable[[Any], Any] <collections.abc.Callable>`, optional
        A callback for reducing the output of the called function.
        Alternativelly, one can provide on of the string aliases
        from :attr:`FromResult.REDUCTION_NAMES`.
    axis : :class:`int` or :class:`Sequence[int] <collections.abc.Sequence>`, optional
        The axis along which the reduction should take place.
        If :data:`None`, use all axes.

    Returns
    -------
    :class:`~typing.Any`
        The output of the extracted method.

    """
    ret = getattr(obj, name)(*args, **kwargs)
    return FromResult._reduce(ret, reduce, axis)
