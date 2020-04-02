from types import MappingProxyType
from typing import Mapping, Optional, TypeVar, Type, Dict, Callable, Any
from collections import abc

from assertionlib import assertion

from .type_hints import Literal

__all__ = ['validate_mapping']

KT = TypeVar('KT')
VT = TypeVar('VT')


class _Key:
    pass


def _val_class(mapping: Mapping) -> None:
    """Validate the :class:`type` of **mapping**."""
    assertion.isinstance(mapping, abc.Mapping)


def _val_items(mapping: Mapping, k_type: Type[KT], v_type: Type[VT]) -> None:
    """Validate the :meth:`~collections.abc.Mapping.items` method of **mapping**."""
    items = mapping.items()
    assertion.isinstance(items, abc.ItemsView)
    for k, v in mapping.items():
        assertion.isinstance(k, k_type)
        assertion.isinstance(v, v_type)


def _val_keys(mapping: Mapping, k_type: Type[KT]) -> None:
    """Validate the :meth:`~collections.abc.Mapping.keys` method of **mapping**."""
    keys = mapping.keys()
    assertion.isinstance(keys, abc.KeysView)
    for k in keys:
        assertion.isinstance(k, k_type)


def _val_values(mapping: Mapping, v_type: Type[VT]) -> None:
    """Validate the :meth:`~collections.abc.Mapping.values` method of **mapping**."""
    values = mapping.values()
    assertion.isinstance(values, abc.ValuesView)
    for v in values:
        assertion.isinstance(v, v_type)


def _val_iter(mapping: Mapping, k_type: Type[KT]) -> None:
    """Validate the :meth:`~collections.abc.Mapping.__iter__` method of **mapping**."""
    iterator = iter(mapping)
    assertion.isinstance(iterator, abc.Iterator)
    for k in iterator:
        assertion.isinstance(k, k_type)


def _val_len(mapping: Mapping) -> None:
    """Validate the :meth:`~collections.abc.Mapping.__len__` method of **mapping**."""
    length = len(mapping)
    assertion.isinstance(length, int)


def _val_contains(mapping: Mapping, key: KT) -> None:
    """Validate the :meth:`~collections.abc.Mapping.__contains__` method of **mapping**."""
    contains_true = key in mapping
    contains_False = _Key in mapping
    assertion.is_(contains_true, True)
    assertion.is_(contains_False, False)


def _val_getitem(mapping: Mapping, key: KT, v_type: Type[VT]) -> None:
    """Validate the :meth:`~collections.abc.Mapping.__getitem__` method of **mapping**."""
    v = mapping[key]
    assertion.isinstance(v, v_type)


def _val_get(mapping: Mapping, key: KT, v_type: Type[VT]) -> None:
    """Validate the :meth:`~collections.abc.Mapping.get` method of **mapping**."""
    v = mapping.get(key, default=True)
    _v = mapping.get(_Key, default=True)
    assertion.isinstance(v, v_type)
    assertion.is_(_v, True)


VALIDATION_MAP: Mapping[str, Callable[..., None]] = MappingProxyType({
    '__class__': _val_class,
    'items': _val_items,
    'keys': _val_keys,
    'values': _val_values,
    '__iter__': _val_iter,
    '__len__': _val_len,
    'get': _val_get,
    '__getitem__': _val_getitem,
    '__contains__': _val_contains
})


Name = Literal['__class__', 'items', 'keys', 'values', '__iter__',
               '__len__', 'get', '__getitem__', '__contains__']


def validate(name: Name, mapping: Mapping, *args: Any, **kwargs: Any) -> Optional[Exception]:
    """Perform the **name**-specific validation of **mapping**."""
    try:
        func = VALIDATION_MAP[name]
        func(mapping, *args, **kwargs)
    except Exception as ex:
        return ex
    else:
        return None


def validate_mapping(mapping: Mapping[KT, VT],
                     key_type: Optional[Type[KT]] = None,
                     value_type: Optional[Type[VT]] = None) -> None:
    """Validate an implementation of a :class:`~collections.abc.Mapping` protocol in **mapping**.

    Parameters
    ----------
    mapping : :class:`~collections.abc.Mapping`
        The to-be validated Mapping.

    key_type : :class:`type`, optional
        The type of the **mapping** keys.
        If not ``None`` it will be used for instance checking.

    value_type : :class:`type`, optional
        The type of the **mapping** values.
        If not ``None`` it will be used for instance checking.

    Exceptions
    ----------
    :exc:`AssertionError`
        Raised if the validation is unsuccessful.

    """
    k_type = object if key_type is None else key_type
    v_type = object if value_type is None else value_type

    cls = mapping.__class__.__name__

    err_dict: Dict[str, Optional[Exception]] = {
        f'{cls}.__class__': validate('__class__', mapping),
        f'{cls}.items': validate('items', mapping, k_type, v_type),
        f'{cls}.keys': validate('keys', mapping, k_type),
        f'{cls}.values': validate('values', mapping, v_type),
        f'{cls}.__iter__': validate('__iter__', mapping, k_type),
        f'{cls}.__len__': validate('__len__', mapping)
    }

    try:
        key = next(iter(mapping.keys()))
    except Exception as ex:
        err_dict[f'{cls}.get'] = ex
        err_dict[f'{cls}.__getitem__'] = ex
        err_dict[f'{cls}.__contains__'] = ex
    else:
        err_dict[f'{cls}.get'] = validate('get', mapping, key, v_type)
        err_dict[f'{cls}.__getitem__'] = validate('__getitem__', mapping, key, v_type)
        err_dict[f'{cls}.__contains__'] = validate('__contains__', mapping, key)

    # Gather all failed tests
    exc_dict = {k: v for k, v in err_dict.items() if v is not None}
    if not exc_dict:
        return

    # Raise an AssertionError
    _, exc_old = next(iter(exc_dict.items()))
    raise AssertionError("Failed to validate the following 'mapping' attributes:\n"
                         f"{', '.join(repr(k) for k in exc_dict)}") from exc_old
