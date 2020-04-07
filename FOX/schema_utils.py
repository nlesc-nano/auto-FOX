"""
FOX.schema_utils
================

A module with :mod:`schema`-related utility functions.

Index
-----
.. currentmodule:: FOX.schema_utils
.. autosummary::
{autosummary}

API
---
{autofunction}

"""

from __future__ import annotations

import inspect
import warnings
from types import FunctionType
from typing import (
    TypeVar, SupportsFloat, SupportsInt, Any, Callable, Union, Tuple, overload, Generic
)

from .type_hints import Literal
from .functions.utils import get_importable

__all__ = ['Default', 'Formatter', 'supports_float', 'supports_int',
           'isinstance_factory', 'issubclass_factory', 'import_factory']

T = TypeVar('T')


@overload
def supports_float(value: SupportsFloat) -> Literal[True]: ...
@overload  # noqa: E302
def supports_float(value: Any) -> bool: ...
def supports_float(value):  # noqa: E302
    """Check if a float-like object has been passed (:data:`~typing.SupportsFloat`)."""
    try:
        value.__float__()
        return True
    except Exception:
        return False


@overload
def supports_int(value: SupportsInt) -> Literal[True]: ...
@overload  # noqa: E302
def supports_int(value: Any) -> bool: ...
def supports_int(value):  # noqa: E302
    """Check if an int-like object has been passed (:data:`~typing.SupportsInt`)."""
    # floats that can be exactly represented by an integer are also fine
    try:
        value.__int__()
        return float(value).is_integer()
    except Exception:
        return False


class Default(Generic[T]):
    """A validation class akin to the likes of :class:`schemas.Use`.

    Upon executing :meth:`Default.validate` returns the stored :attr:`~Default.value`.
    If :attr:`~Default.call` is ``True`` and the value is a callable,
    then it is called before its return.

    Examples
    --------
    .. code:: python

        >>> from schema import Schema

        >>> schema1 = Schema(int, Default(True))
        >>> schema1.validate(1)
        True

        >>> schema2 = Schema(int, Default(dict))
        >>> schema2.validate(1)
        {}

        >>> schema3 = Schema(int, Default(dict, call=False))
        >>> schema3.validate(1)
        <class 'dict'>


    Attributes
    ----------
    value : :class:`~collections.abc.Callable` or :data:`~typing.Any`
        The to-be return value for when :meth:`Default.validate` is called.
        If :attr:`Default.call` is ``True`` then the value is called
        (if possible) before its return.
    call : :class:`bool`
        Whether to call :attr:`Default.value` before its return (if possible) or not.

    """

    value: Union[T, Callable[[], T]]
    call: bool

    def __init__(self, value: Union[T, Callable[[], T]], call: bool = True) -> None:
        """Initialize an instance."""
        self.value = value
        self.call = call

    def __repr__(self) -> str:
        """Implement :code:`str(self)` and :code:`repr(self)`."""
        return f'{self.__class__.__name__}({self.value!r}, call={self.call!r})'

    def validate(self, data: Any) -> Union[T, Callable[[], T]]:
        """Validate the passed **data**."""
        if self.call and callable(self.value):
            return self.value()
        else:
            return self.value


class Formatter(str):

    msg: str

    def __init__(self, msg: str):
        """Initialize an instance."""
        self.msg = msg

    def __repr__(self) -> str:
        """Implement :code:`str(self)` and :code:`repr(self)`."""
        return f'{self.__class__.__name__}(msg={self.msg!r})'

    def format(self, obj: Any) -> str:  # type: ignore
        """Return a formatted version of :attr:`Formatter.msg`."""
        name = self.msg.split("'", maxsplit=2)[1]
        name_ = name or 'value'
        try:
            return self.msg.format(name=name_, value=obj, type=obj.__class__.__name__)
        except Exception as ex:
            err = RuntimeWarning(ex)
            err.__cause__ = ex
            warnings.warn(err)
            return repr(obj)

    @property
    def __mod__(self) -> Callable[[Any], str]:  # type: ignore
        """Get :meth:`Formatter.format`."""
        return self.format


def _get_caller_module(n: int = 1) -> str:
    """Return the name of the module **n** levels up the stack."""
    try:
        frm = inspect.stack()[n]
    except IndexError:
        return f'{__package__}.{__name__}'

    mod = inspect.getmodule(frm[0])
    if mod is not None:
        return mod.__name__
    else:
        return f'{__package__}.{__name__}'


def _setattrs(func: Callable, name: str, doc: str) -> None:
    """Set the :attr:`__name__`, :attr:`__qualname__`, :attr:`__doc__` and :attr:`__module__` attributes of **func**."""  # noqa: E501
    func.__name__ = func.__qualname__ = name
    func.__doc__ = doc
    func.__module__ = _get_caller_module(n=3)


ClassOrTuple = Union[type, Tuple[type, ...]]


def isinstance_factory(class_or_tuple: ClassOrTuple) -> Callable[[Any], bool]:
    """Return a function which checks if the passed object is an instance of **class_or_tuple**.

    Serves a similar, albit more specialized, function as :class:`functools.partial`.

    """
    try:
        isinstance('bob', class_or_tuple)
    except TypeError as ex:
        raise TypeError("'class_or_tuple' expected a type or tuple of types; "  # type: ignore
                        f"observed type: {class_or_tuple.__class__.__name_!r}") from ex

    if isinstance(class_or_tuple, type):
        arg: Tuple[type, ...] = (class_or_tuple,)
    else:
        arg = class_or_tuple
    name = f"isinstance_{abs(hash(frozenset(arg)))}"
    doc = f"({', '.join(obj.__name__ for obj in arg)})"

    def func(obj: Any) -> bool:
        return isinstance(obj, class_or_tuple)

    _setattrs(func, name, f"Return :func:`isinstance(obj, {doc})<isinstance>`.")
    return func


def issubclass_factory(class_or_tuple: ClassOrTuple) -> Callable[[type], bool]:
    """Return a function which checks if the passed class is a subclass of **class_or_tuple**.

    Serves a similar, albit more specialized, function as :class:`functools.partial`.

    """
    try:
        issubclass(int, class_or_tuple)
    except TypeError as ex:
        raise TypeError("'class_or_tuple' expected a type or tuple of types; "  # type: ignore
                        f"observed type: {class_or_tuple.__class__.__name_!r}") from ex

    if isinstance(class_or_tuple, type):
        arg: Tuple[type, ...] = (class_or_tuple,)
    else:
        arg = class_or_tuple
    name = f"issubclass_{abs(hash(frozenset(arg)))}"
    doc = f"({', '.join(obj.__name__ for obj in arg)})"

    def func(cls: type) -> bool:
        return issubclass(cls, arg)

    _setattrs(func, name, f"Return :func:`issubclass(cls, {doc})<issubclass>`.")
    return func


GET_IMPORTABLE = f"{get_importable.__module__}.{get_importable.__name__}"


def import_factory(validate: Callable[[T], bool]) -> Callable[[str], T]:
    """Return a function which calls :func:`~FOX.functions.utils.get_importable` with the **validate** argument.

    Serves a similar, albit more specialized, function as :class:`functools.partial`.

    """  # noqa: E501
    if not callable(validate):
        raise TypeError("'validate' expected a callable; observed type: "
                        f"{validate.__class__.__name__!r}")

    try:
        _doc = getattr(validate, '__qualname__', validate.__name__)  # type: ignore
    except AttributeError:  # Incase something like a functools.partial object is encountered
        _doc = repr(validate)
    finally:
        name = f'get_importable_{abs(hash(validate))}'
        doc = f'{validate.__module__}.{_doc}'

    def func(string: str) -> T:
        return get_importable(string, validate=validate)

    _setattrs(func, name, f"Return :func:`get_importable(string, validate={doc})<{GET_IMPORTABLE}>`.")  # noqa: E501
    return func


def _get_directive(string: str) -> str:
    obj: Union[type, FunctionType] = globals()[string]
    if isinstance(obj, type):
        return f'.. autoclass:: {obj}'
    else:
        return f'.. autofunction:: {obj}'


__doc__ = __doc__.format(
    autosummary='\n'.join(f'    {i}' for i in __all__),
    autofunction='\n'.join(_get_directive(i) for i in __all__)
)
