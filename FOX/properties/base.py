"""A module containing the abstract :class:`FromResult` class."""

from __future__ import annotations

import types
import inspect
from typing import Generic, TypeVar, Any, Callable

import numpy as np
import scipy.special

FT = TypeVar("FT", bound=Callable[..., Any])

__all__ = ['FromResult', 'get_attr', 'call_method']


def _gather_ufuncs(module):
    """Gather a dictionary with all :class:`~numpy.ufunc.reduce`-supporting :class:`ufuncs <numpy.ufunc>` from the passed **module**."""  # noqa: E501
    iterator = (getattr(module, name) for name in getattr(module, '__all__', []))
    condition = lambda ufunc: (
        isinstance(ufunc, np.ufunc)
        and ufunc.signature is None
        and ufunc.nin == 2
        and ufunc.nout == 1
    )
    return {ufunc.__name__: ufunc.reduce for ufunc in iterator if condition(ufunc)}


class FromResult(Generic[FT]):
    """A decorating class for wrapping :data:`~types.FunctionType` objects.

    Besides :meth:`__call__`, instances have access to the :meth:`from_result` method,
    which is used for applying the wrapped callable to
    a :class:`qmflows.CP2K_Result <qmflows.packages.Result>` instance.

    Parameters
    ----------
    func : :data:`types.FunctionType`
        The to-be wrapped function.
    result_func : :class:`~collections.abc.Callable`
        The function for reading the CP2K :class:`~qmflows.packages.Result` object.

    """

    __slots__ = ("__weakref__", "__call__", "from_result", "__dict__")

    #: A mapping that maps :meth:`from_result` aliases to callbacks.
    REDUCTION_NAMES = types.MappingProxyType({
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

    __call__: FT
    __module__: str
    __annotations__: dict[str, Any]
    __doc__: str
    from_result: Callable[..., Any]

    def __init__(self, func, result_func=None):
        """Initialize the instance."""
        if not isinstance(func, types.FunctionType):
            raise TypeError("`func` expected a function")

        super().__setattr__("__call__", func)
        super().__setattr__("__module__", func.__module__)
        super().__setattr__("__doc__", func.__doc__)
        super().__setattr__("__annotations__", func.__annotations__)
        if result_func is not None:
            super().__setattr__("from_result", types.MethodType(result_func, self))

    def __getattr__(self, name):
        """Implement :func:`getattr(self, name) <getattr>`."""
        try:
            return getattr(self.__call__, name)
        except AttributeError:
            pass
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __setattr__(self, name, value):
        """Implement :func:`setattr(self, name, value) <setattr>`."""
        if name == "__weakref__":
            return super().__setattr__(name, value)
        elif hasattr(self, name):
            raise AttributeError(f"{type(self).__name__!r} object attribute {name!r} is read-only")
        else:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __delattr__(self, name):
        """Implement :func:`delattr(self, name) <delattr>`."""
        if hasattr(self, name):
            raise AttributeError(f"{type(self).__name__!r} object attribute {name!r} is read-only")
        else:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __dir__(self):
        """Implement :func:`dir(self) <dir>`."""
        return sorted(set(super().__dir__() + dir(self.__call__)))

    def __hash__(self):
        """Implement :func:`hash(self) <hash>`."""
        return hash(self.__call__)

    def __eq__(self, value):
        """Implement :meth:`self == value <object.__eq__>`."""
        cls = type(self)
        if not isinstance(value, cls):
            return NotImplemented
        return self.__call__ == value.__call__

    def __repr__(self):
        """Implement :class:`str(self) <str>` and :func:`repr(self) <repr>`."""
        func = self.__call__
        sgn = inspect.signature(func)
        return f'<{type(self).__name__} instance {func.__module__}.{func.__qualname__}{sgn}>'

    def __reduce__(self):
        """A helper method for :mod:`pickle`."""
        return self.__qualname__

    def __copy__(self):
        """Implement :func:`copy.copy(self) <copy.copy>`."""
        return self

    def __deepcopy__(self, memo=None):
        """Implement :func:`copy.deepcopy(self, memo=memo) <copy.deepcopy>`."""
        return self

    def _set_result_func(self, result_func):
        """A decorator for setting :attr:`from_result`."""
        super().__setattr__("from_result", types.MethodType(result_func, self))
        return self

    @classmethod
    def _reduce(cls, value, reduce=None, axis=None):
        """A helper function to handle the reductions in :meth:`from_result`."""
        if reduce is None:
            return value
        elif callable(reduce):
            return reduce(value)

        try:
            func = cls.REDUCTION_NAMES[reduce]
        except (TypeError, KeyError):
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
    else:
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
