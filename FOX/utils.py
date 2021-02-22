"""A module with miscellaneous functions.

Index
-----
.. currentmodule:: FOX.utils
.. autosummary::
    get_move_range
    array_to_index
    serialize_array
    read_str_file
    get_shape
    slice_str
    get_atom_count
    read_rtf_file
    prepend_exception
    log_traceback_locals
    slice_iter
    lattice_to_volume

API
---
.. autofunction:: array_to_index
.. autofunction:: serialize_array
.. autofunction:: read_str_file
.. autofunction:: get_shape
.. autofunction:: slice_str
.. autofunction:: get_atom_count
.. autofunction:: read_rtf_file
.. autofunction:: prepend_exception
.. autofunction:: log_traceback_locals
.. autofunction:: slice_iter
.. autofunction:: lattice_to_volume

"""

from __future__ import annotations

import operator
import inspect
import textwrap
import warnings
from pprint import pformat
from logging import Logger
from functools import wraps
from typing import (
    Iterable, Tuple, Callable, Hashable, Sequence, Optional, List, TypeVar,
    Type, Mapping, Union, Any, cast, Generator, TYPE_CHECKING
)

import numpy as np
import pandas as pd
from nanoutils import PathType

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    'get_move_range', 'array_to_index', 'serialize_array', 'read_str_file',
    'get_shape', 'slice_str', 'get_atom_count', 'read_rtf_file', 'prepend_exception',
    'log_traceback_locals', 'slice_iter', 'lattice_to_volume',
]

T = TypeVar('T')
VT = TypeVar('VT')
KT = TypeVar('KT', bound=Hashable)
FT = TypeVar('FT', bound=Callable)

ExcType = Union[Type[Exception], Tuple[Type[Exception], ...]]


def get_shape(item: Any) -> Tuple[int, ...]:
    """Try to infer the shape of an object.

    Examples
    --------
    .. code:: python

        >>> item = np.random.rand(10, 10, 10)
        >>> shape = get_shape(item)
        >>> print(shape)
        (10, 10, 10)

        >>> item = ['a', 'b', 'c', 'd', 'e']
        >>> shape = get_shape(item)
        >>> print(shape)
        (5,)

        >>> item = None
        >>> shape = get_shape(item)
        >>> print(shape)
        (1,)

    Parameters
    ----------
    item : object
        A python object.

    Returns
    -------
    |tuple|_ [|int|_]:
        The shape of **item**.

    """
    if hasattr(item, 'shape'):  # Plan A: **item** is an np.ndarray derived object
        return item.shape
    elif hasattr(item, '__len__'):  # Plan B: **item** has access to the __len__ magic method
        return (len(item), )
    return (1, )  # Plan C: **item** has access to neither A nor B


def serialize_array(array: np.ndarray, items_per_row: int = 4) -> str:
    """Serialize an array into a single string.

    Newlines are placed for every **items_per_row** rows in **array**.

    Parameters
    ----------
    array : |np.ndarray|_
        A 2D array.

    items_per_row : int
        The number of values per row before switching to a new line.

    Returns
    -------
    |str|_:
        A serialized array.

    """
    if len(array) == 0:
        return ''

    ret = ''
    k = 0
    for i in array:
        for j in i:
            ret += '{:>10d}'.format(j)
        k += 1
        if k == items_per_row:
            k = 0
            ret += '\n'

    return ret


def read_str_file(filename: PathType) -> Optional[Tuple[Sequence[str], Sequence[float]]]:
    """Read atomic charges from CHARMM-compatible stream files (.str).

    Returns a settings object with atom types and (atomic) charges.

    Parameters
    ----------
    filename : str
        the path+filename of the .str file.

    Returns
    -------
    :class:`Sequence` [:class:`str`] and :class:`Sequence` [:class:`float`]
        A settings object with atom types and (atomic) charges.

    """
    def inner_loop(f):
        ret = []
        ret_append = ret.append
        for j in f:
            if j != '\n':
                j = j.split()[2:4]
                ret_append((j[0], float(j[1])))
            else:
                return ret

    with open(filename, 'r') as f:
        for i in f:
            if 'GROUP' in i:
                return cast(Tuple[Sequence[str], Sequence[float]], zip(*inner_loop(f)))
        else:
            raise RuntimeError(f"Failed to parse {filename!r}")


def array_to_index(ar: np.ndarray) -> pd.Index:
    """Convert a NumPy array into a Pandas Index or MultiIndex.

    Examples
    --------
    .. code:: python

        >>> ar = np.arange(10)
        >>> idx = array_to_index(ar)
        >>> print(idx)
        Int64Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64')

        >>> ar = np.random.randint(0, high=100, size=(4, 2))
        >>> idx = array_to_index(ar)
        >>> print(idx)
        MultiIndex(levels=[[30, 97], [13, 45], [42, 86], [44, 63]],
                   codes=[[0, 1], [1, 0], [1, 0], [0, 1]])

    Parameters
    ----------
    ar : |np.ndarray|_
        A 1D or 2D NumPy array.

    Results
    -------
    |pd.Index|_ or |pd.MultiIndex|_:
        A Pandas Index or MultiIndex constructed from **ar**.

    Raises
    ------
    ValueError
        Raised if the dimensionality of **ar** is greater than 2.

    """
    if 'bytes' in ar.dtype.name:
        ar = ar.astype(str, copy=False)

    if ar.ndim == 1:
        return pd.Index(ar)
    elif ar.ndim == 2:
        return pd.MultiIndex.from_arrays(ar)
    raise ValueError('Could not construct a Pandas (Multi)Index from an '
                     f'{ar.ndim}-dimensional array')


def get_move_range(start: float = 0.005,
                   stop: float = 0.1,
                   step: float = 0.005,
                   ratio: Optional[Iterable[float]] = None) -> np.ndarray:
    """Generate an with array of all allowed moves.

    The move range spans a range of 1.0 +- **stop** and moves are thus intended to
    applied in a multiplicative manner (see :meth:`MonteCarlo.move_param`).

    Examples
    --------
    .. code:: python

        >>> move_range1 = get_move_range(start=0.005, stop=0.020,
        ...                              step=0.005, ratio=None)
        >>> print(move_range1)
        [0.98  0.985 0.99  0.995 1.    1.005 1.01  1.015 1.02 ]

        >>> move_range2 = get_move_range(start=0.005, stop=0.020,
        ...                              step=0.005, ratio=[1, 2, 4, 8])
        >>> print(move_range2)
        [[0.98  0.985 0.99  0.995 1.    1.005 1.01  1.015 1.02 ]
         [0.96  0.97  0.98  0.99  1.    1.01  1.02  1.03  1.04 ]
         [0.92  0.94  0.96  0.98  1.    1.02  1.04  1.06  1.08 ]
         [0.84  0.88  0.92  0.96  1.    1.04  1.08  1.12  1.16 ]]

    Parameters
    ----------
    start : :class:`float`
        Start of the interval.
        The interval includes this value.

    stop : :class:`float`
        End of the interval.
        The interval includes this value.

    step : :class:`float`
        Spacing between values.

    ratio : :class:`~collections.abc.Iterable` [:class:`float`], optional
        If an iterable of length :math:`n` is provided here then the returned array is tiled
        and scaled with the elements of **ratio** using the standard NumPy broadcasting rules.

    Returns
    -------
    :class:`numpy.ndarray` [:class:`float`]
        An array (1D if ``ratio=None``; 2D otherwise) with allowed moves.

    Raises
    ------
    :exc:`ValueError`
        Raised if one or more elements in the to-be returned array are smaller than 0.

    """
    rng_range1 = np.arange(1 + start, 1 + stop, step, dtype=float)
    rng_range2 = np.arange(1 - stop, 1 - start + step, step, dtype=float)
    ret = np.concatenate((rng_range1, rng_range2))
    ret.sort()

    if ratio is None:
        if (ret < 0).any():
            raise ValueError("The returned array cannot contain elements smaller than 0; "
                             f"smallest osberved value: {ret.min()}")
        return ret

    ratio_ar = np.fromiter(ratio, dtype=float)
    ret -= 1
    ret_ = ret[None, ...] * ratio_ar[..., None] + 1
    if (ret_ < 0).any():
        raise ValueError("The returned array cannot contain elements smaller than 0; "
                         f"smallest osberved value: {ret_.min()}")
    return ret_


@wraps(get_move_range)
def _get_move_range(*args, **kwargs):
    _warning = FutureWarning("The 'FOX.utils._get_move_range' function is deprecated; "
                             "use 'FOX.utils.get_move_range' from now on")
    warnings.warn(_warning, stacklevel=2)
    return get_move_range(*args, **kwargs)


def slice_str(str_: str, intervals: List[Optional[int]],
              strip_spaces: bool = True) -> List[str]:
    """Slice a string, **str_**, at intervals specified in **intervals**.

    Examples
    --------
    .. code:: python

        >>> my_str = '123456789'
        >>> intervals = [None, 3, 6, None]
        >>> str_list = slice_str(my_str, intervals)
        >>> print(str_list)
        ['123', '456', '789']

    Parameters
    ----------
    str_ : str
        A string.

    intverals : list [int]
        A list with :math:`n` objects suitable for slicing.

    strip_spaces : bool
        If empty spaces should be stripped or not.

    Results
    -------
    :math:`n-1` |list|_ [|str|_]:
        A list of strings as sliced from **str_**.

    """
    iter1 = intervals[:-1]
    iter2 = intervals[1:]
    if strip_spaces:
        return [str_[i:j].strip() for i, j in zip(iter1, iter2)]
    return [str_[i:j] for i, j in zip(iter1, iter2)]


def get_atom_count(iterable: Iterable[Sequence[KT]],
                   count: Mapping[KT, int]) -> List[Optional[int]]:
    """Count the occurences of each atom/atom-pair (from **iterable**) in as defined by **count**.

    Parameters
    ----------
    iterable : :class:`~collections.abc.Iterable` [:class:`~collections.abc.Sequence` [:class:`str`]], shape :math:`(n, m)`
        An iterable consisting of atom-lists.

    count : :class:`~collections.abc.Mapping` [:class:`str`, :class:`int`]
        A dict which maps atoms to atom counts.

    Returns
    -------
    :class:`list` [:class:`int`, optional], shape :math:`(n,)`
        A list of atom(-pair) counts.
        A particular value is replace with ``None`` if a :exc:`KeyError` is encountered.

    """  # noqa: E501
    def _get_atom_count(tup: Sequence[KT]) -> Optional[int]:
        try:
            if len(tup) == 2 and tup[0] == tup[1]:
                int1 = count[tup[0]]
                return (int1**2 - int1) // 2
            elif len(tup) == 2:
                return np.product([count[i] for i in tup])
            elif len(tup) == 1:
                return count[tup[0]]
            return None
        except KeyError:
            return None

    return [_get_atom_count(tup) for tup in iterable]


def read_rtf_file(filename: PathType) -> Optional[Tuple[Sequence[str], Sequence[float]]]:
    """Return a 2-tuple with all atom types and charges."""
    def _parse_item(item: str) -> Tuple[str, float]:
        item_list = item.split()
        return item_list[j], float(item_list[k])

    i, j, k = len('ATOM'), 2, 3
    with open(filename, 'r') as f:
        try:
            ret = [_parse_item(item) for item in f if item[:i] == 'ATOM']
        except Exception as ex:
            raise RuntimeError(f"Failed to parse {filename!r}") from ex

    if ret:
        return cast(Tuple[Sequence[str], Sequence[float]], tuple(zip(*ret)))
    else:
        return None


def prepend_exception(msg: str, exception: ExcType = Exception) -> Callable[[FT], FT]:
    """Prepend all :exc:`KeyError` messages raised by **func**.

    Examples
    --------
    .. code:: python

        >>> from FOX.utils import prepend_exception

        >>> @prepend_exception('custom message: ', exception=TypeError)
        ... def func():
        ...     raise TypeError('test')

        >>> func()
        Traceback (most recent call last):
            ...
        TypeError: "custom message: test"


    Parameters
    ----------
    msg : :class:`str`
        The to-be prepended message.

    exception : :class:`type` [:class:`Exception`]
        An exception type or tuple of exception types.
        All herein specified exception will have their exception messages prepended.

    """
    exc_tup = exception if isinstance(exception, tuple) else (exception,)
    if not all(isinstance(ex, type) and issubclass(ex, Exception) for ex in exc_tup):
        raise TypeError("'exception' expected an Exception type or tuple of Exception types")

    def _decorator(func: FT) -> FT:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exc_tup as ex:
                cls = type(ex)
                raise cls(f"{msg}{ex}").with_traceback(ex.__traceback__)
        return cast(FT, wrapper)

    return _decorator


def log_traceback_locals(logger: Logger, level: int = -1,
                         str_func: Callable[[object], str] = pformat) -> None:
    """Log all local variables at the specified traceback level.

    Parameters
    ----------
    logger : :class:`~logging.Logger`
        A logger for writing the local variables.
    level : :class:`int`
        The traceback level.
    str_func : :data:`Callable[[object], str]<typing.Callable>`
        The callable for creating the variables string representation.

    """
    try:
        local_dct = inspect.trace()[level].frame.f_locals
    except IndexError:
        i = operator.index(level)
        raise RuntimeError(f"No traceback was found at level {i}") from None

    for name, _value in local_dct.items():
        prefix = f"    {name}: {_value.__class__.__name__} = "
        n = len(prefix)
        value_str = textwrap.indent(str_func(_value), n * ' ')[n:].split('\n')
        value_str[0] = prefix + value_str[0]
        for v in value_str:
            logger.debug(v)


def slice_iter(
    shape: Sequence[int],
    itemsize: int = 1,
    nbytes_max: int = 1024**3,
) -> Generator[slice, None, None]:
    """Return a generator of :class:`slice` objects.

    Parameters
    ----------
    shape : :class:`Sequence[int] <collections.abc.Sequence>`
        The maximum shape of the relevant array.
    itemsize : :class:`int`
        The element size in bytes.
    nbytes_max : :class:`int`
        The maximum size (in bytes) of each of the arrays' chunks.

    Yields
    ------
    :class:`slice`
        Slice instances.

    """
    if nbytes_max <= 0:
        raise ValueError("`nbytes_max` must be larger than 0")
    elif len(shape) == 0:
        yield slice(None)
        return

    size = np.product(shape, dtype=np.int64) * itemsize
    n = shape[0]
    n_step = max(1, np.ceil(n / (size / nbytes_max)).astype(np.int64))

    start = 0
    stop = n_step
    while n > start:
        yield slice(start, stop)
        start += n_step
        stop += n_step


def lattice_to_volume(a: npt.ArrayLike) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Calculate the volume contained within a set of lattice vectors."""
    ar = np.asarray(a, dtype=np.float64)
    if ar.ndim < 2:
        raise ValueError(f"Expected a >= 2D array; observed dimensionality: {ar.ndim}")
    elif ar.shape[-2] != 3:
        raise ValueError(f"Invalid shape: {ar.shape}")

    # Calculate and return the triple product
    cross = np.cross(ar[..., 1, :], ar[..., 2, :])
    return np.einsum("...j,...j->...", ar[..., 0, :], cross)
