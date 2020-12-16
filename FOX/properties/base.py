"""A module containing the abstract :class:`FromResult` class."""

from __future__ import annotations

import inspect
import textwrap
from abc import ABCMeta, abstractmethod
from types import MappingProxyType, ModuleType
from typing import Generic, TypeVar, Any, Callable, Dict

import scipy.special
import numpy as np
from qmflows.packages import Result

__all__ = ['FromResult', 'get_attr', 'call_method']

FT = TypeVar("FT", bound=Callable[..., Any])
RT = TypeVar("RT", bound=Result)


def _gather_ufuncs(module: ModuleType) -> Dict[str, Callable[[Any], Any]]:
    """Gather a dictionary with all :class:`~numpy.ufunc.reduce`-supporting :class:`ufuncs <numpy.ufunc>` from the passed **module**."""  # noqa: E501
    iterator = (getattr(module, name) for name in getattr(module, '__all__', []))
    condition = lambda ufunc: (isinstance(ufunc, np.ufunc) and ufunc.signature is None and
                               ufunc.nin == 2 and ufunc.nout == 1)
    return {ufunc.__name__: ufunc.reduce for ufunc in iterator if condition(ufunc)}


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
        The :attr:`~definition.__name__` of the to-be created instance.
    module : :class:`str`
        The :attr:``__module__`` of the to-be created instance.
        If :data:`None`, set it to ``"__main__"``.
    doc : :class:`str`, optional
        The :attr:``__doc__`` of the to-be created instance.
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

        if hasattr(func, '__signature__'):
            self._signature = func.__signature__
        elif getattr(func, '__text_signature__', None) is not None:
            self._signature = None
        else:
            try:
                self._signature = inspect.signature(func)
            except ValueError:
                self._signature = inspect.Signature([
                    inspect.Parameter('args', kind=inspect.Parameter.VAR_POSITIONAL),
                    inspect.Parameter('kwargs', kind=inspect.Parameter.VAR_KEYWORD),
                ])

        # Can't make `__globals__` into a property or sphinx will crash:
        #
        # Extension error:
        # Handler <function record_typehints at 0x00000229C568BAF0> for event
        # 'autodoc-process-signature' threw an exception (exception: 'property' object
        # has no attribute 'get')
        self.__globals__ = MappingProxyType(getattr(func, '__globals__', {}))

    @property
    def __signature__(self):
        """Get a :class:`~inspect.Signature` representing the underlying function."""
        return self._signature

    @property
    def __annotations__(self):
        """Get the :attr:`~types.FunctionType.__annotations__>` of the underlying function as a read-only view."""  # noqa: E501
        try:
            dct = self._func.__annotations__
        except AttributeError:
            dct = {'args': Any, 'kwargs': Any, 'return': Any}
        return MappingProxyType(dct)

    @property
    def __text_signature__(self):
        """Get the :attr:`~types.FunctionType.__text_signature__>` of the underlying function."""
        return getattr(self._func, '__text_signature__', None)

    @property
    def __closure__(self):
        """Get the :attr:`~types.FunctionType.__closure__>` of the underlying function."""
        return getattr(self._func, '__closure__', None)

    @property
    def __defaults__(self):
        """Get the :attr:`~types.FunctionType.__defaults__>` of the underlying function."""
        return getattr(self._func, '__defaults__', None)

    @property
    def __kwdefaults__(self):
        """Get the :attr:`~types.FunctionType.__kwdefaults__>` of the underlying function as a read-only view."""  # noqa: E501
        return MappingProxyType(getattr(self._func, '__kwdefaults__', {}))

    @property
    def __code__(self):
        """Get the :attr:`~types.FunctionType.__code__>` of the underlying function.

        Note
        ----
        This property is only available if the underlying function supports it.

        """
        return self._func.__code__

    @property
    def __get__(self):
        """Get the :attr:`~types.FunctionType.__get__>` method of the underlying function.

        Note
        ----
        This property is only available if the underlying function supports it.

        """
        return self._func.__get__

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

    @abstractmethod
    def from_result(self, result, reduction=None, **kwargs):
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

    @classmethod
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
                raise TypeError("`reduction` expected a string; observed type: "
                                f"{reduction.__class__.__name__!r}") from None
            else:
                raise ValueError(f"Invalid `reduction` value: {reduction!r}") from None
        else:
            return func(value)

    @staticmethod
    def _pop(dct, key, callback):
        """Attempt to :meth:`~dict.pop` **key** from **dct**, fall back to **callback** otherwise."""  # noqa: E501
        if key in dct:
            return dct.pop(key)
        else:
            return callback()


class _Null:
    """A singleton used as sentinel value in :func:`get_attr`."""

    ...


def get_attr(obj, name, default=_Null, reduction=None):
    """:func:`gettattr` with support for keyword argument.

    Parameters
    ----------
    obj : :class:`object`
        The object in question.
    name : :class:`str`
        The name of the to-be extracted attribute.
    default : :class:`~typing.Any`
        An object that is to-be returned if **obj** does not have the **name** attribute.
    reduction : :class:`str` or :class:`Callable[[Any], Any] <collections.abc.Callable>`, optional
        A callback for reducing the extracted attribute.
        Alternativelly, one can provide on of the string aliases
        from :attr:`FromResult.REDUCTION_NAMES`.

    Returns
    -------
    :class:`~typing.Any`
        The extracted attribute.

    See Also
    --------
    :func:`getattr`
        Get a named attribute from an object.

    """
    if default is _Null:
        ret = getattr(obj, name)
    ret = getattr(obj, name, default)
    return FromResult._reduce(ret, reduction)


def call_method(obj, name, *args, reduction=None, **kwargs):
    r"""Call the **name** method of **obj**.

    Parameters
    ----------
    obj : :class:`object`
        The object in question.
    name : :class:`str`
        The name of the to-be extracted method.
    \*args/\**kwargs : :class:`~typing.Any`
        Positional and/or keyword arguments for the (to-be called) extracted method.
    reduction : :class:`str` or :class:`Callable[[Any], Any] <collections.abc.Callable>`, optional
        A callback for reducing the output of the called function.
        Alternativelly, one can provide on of the string aliases
        from :attr:`FromResult.REDUCTION_NAMES`.

    Returns
    -------
    :class:`~typing.Any`
        The output of the extracted method.

    """
    ret = getattr(obj, name)(*args, **kwargs)
    return FromResult._reduce(ret, reduction)
