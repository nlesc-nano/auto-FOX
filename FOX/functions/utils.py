"""A module with miscellaneous functions."""

import reprlib
import warnings
import importlib
from os import PathLike
from os.path import join, isfile
from functools import wraps
from collections import abc
from pkg_resources import resource_filename
from typing import (
    Iterable, Tuple, Callable, Hashable, Sequence, Optional, List, Any, TypeVar, Dict,
    Type, Mapping, Union, MutableMapping, TYPE_CHECKING, AnyStr, Container
)

import yaml
import numpy as np
import pandas as pd

from scm.plams import Settings, add_to_class

from ..type_hints import Scalar, SupportsArray, DtypeLike

if TYPE_CHECKING:
    from ..classes import MultiMolecule
else:
    MultiMolecule = 'FOX.classes.multi_Mol.MultiMolecule'

__all__ = ['get_template', 'template_to_df', 'get_example_xyz', 'as_nd_array']

KT = TypeVar('KT', bound=Hashable)
VT = TypeVar('VT')
T = TypeVar('T')


def append_docstring(item: Callable[..., T]) -> Callable[..., T]:
    r"""A decorator for appending the docstring of a Callable one provided by another Callable.

    Examples
    --------
    .. code:: python

        >>> def func1():
        ...     '''func1 docstring.'''
        ...     pass

        >>> @append_docstring(func1)
        >>> def func2():
        ...     '''func2 docstring.'''
        ...     pass

        >>> help(func2)
        'func2 docstring func1 docstring'

    Parameters
    ----------
    item : |Callable|_
        A Callable object with a docstring.

    Returns
    -------
    |Callable|_
        A decorated callable.

    """
    def decorator(func):
        try:
            func.__doc__ += item.__doc__
        except TypeError:
            func.__doc__ = item.__doc__
        return func
    return decorator


def assert_error(error_msg: Optional[str] = None) -> Callable:
    """Take an error message, if evaluating ``True`` then cause a function or class to raise a ModuleNotFoundError upon being called.

    Indended for use as a decorater:

    Examples
    --------
    .. code:: python

        >>> @assert_error('An error was raised by {}')
        >>> def my_custom_func():
        ...     print(True)

        >>> my_custom_func()
        ModuleNotFoundError: An error was raised by my_custom_func

    Parameters
    ----------
    error_msg : str, optional
        A to-be printed error message.
        If available, a single set of curly brackets will be replaced
        with the function or class name.

    Returns
    -------
    |Callable|_
        A decorated callable.

    """  # noqa
    def _function_error(f_type: Callable,
                        error_msg: str) -> Callable:
        """Process functions fed into :func:`assert_error`."""
        if not error_msg:
            return f_type

        @wraps(f_type)
        def wrapper(*arg, **kwarg):
            raise ModuleNotFoundError(error_msg.format(f_type.__name__))
        return wrapper

    def _class_error(f_type: Callable,
                     error_msg: str) -> Callable:
        """Process classes fed into :func:`assert_error`."""
        if error_msg:
            @add_to_class(f_type)
            def __init__(self, *arg, **kwarg):
                raise ModuleNotFoundError(error_msg.format(f_type.__name__))
        return f_type

    type_dict = {'function': _function_error, 'type': _class_error}

    def decorator(func):
        return type_dict[func.__class__.__name__](func, error_msg)
    return decorator


def get_template(name: str,
                 path: Optional[str] = None,
                 as_settings: bool = True) -> dict:
    """Grab a .yaml template and turn it into a Settings object.

    Parameters
    ----------
    name : str
        The name of the template file.

    path : str
        The path where **name** is located.
        Will default to the :mod:`FOX.data` directory if *None*.

    as_settings : bool
        If ``False``, return a dictionary rather than a settings object.

    Returns
    -------
    |plams.Settings|_ or |dict|_:
        A settings object or dictionary as constructed from the template file.

    """
    if path is None:
        if not isfile(name):
            path = resource_filename('FOX', join('data', name))
    else:
        path = join(path, name)

    with open(path, 'r') as f:
        if as_settings:
            return Settings(yaml.load(f, Loader=yaml.FullLoader))
        return yaml.load(f, Loader=yaml.FullLoader)


def template_to_df(name: str, path: Optional[str] = None) -> pd.DataFrame:
    """Grab a .yaml template and turn it into a pandas dataframe.

    Parameters
    ----------
    name : str
        The name of the template file.

    path : str
        The path where **name** is located.
        Will default to the :mod:`FOX.data` directory if ``None*``.

    Returns
    -------
    |pd.DataFrame|_:
        A dataframe as constructed from the template file.

    """
    template_dict = get_template(name, path=path, as_settings=False)
    try:
        return pd.DataFrame(template_dict).T
    except ValueError:
        idx = list(template_dict.keys())
        values = list(template_dict.values())
        return pd.DataFrame(values, index=idx, columns=['charge'])


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


def read_str_file(filename: Union[AnyStr, PathLike]) -> Optional[Tuple[Sequence, Sequence]]:
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
                return zip(*inner_loop(f))
        else:
            raise RuntimeError(f"Failed to parse {filename!r}")


def get_shape(item: Iterable) -> Tuple[int]:
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


def dict_to_pandas(input_dict: dict, name: Hashable = 0,
                   object_type: str = 'pd.DataFrame') -> pd.DataFrame:
    """Turn a nested dictionary into a pandas series or dataframe.

    Keys are un-nested and used for generating multiindices (see meth:`flatten_dict`).

    Examples
    --------
    .. code:: python

        >>> name = 'param'
        >>> print(input_dict)
        epsilon:
                Br Br:  1.0501
                Cs Br:  0.4447
                keys:   ['input', 'force_eval', 'mm', 'forcefield', 'nonbonded', 'lennard-jones']
        sigma:
              Br Br:    0.42
              Cs Br:    0.38
              keys:     ['input', 'force_eval', 'mm', 'forcefield', 'nonbonded', 'lennard-jones']
              unit:     nm

        >>> df = dict_to_pandas(input_dict, name=name)
        >>> print(df)
                                                                   param
        epsilon Br Br                                             1.0501
                Cs Br                                             0.4447
                keys   [input, force_eval, mm, forcefield, nonbonded,...
        sigma   Br Br                                               0.42
                Cs Br                                               0.38
                keys   [input, force_eval, mm, forcefield, nonbonded,...
                unit                                                  nm

    Parameters
    ----------
    input_dict : dict
        A nested dictionary.

    name : |Hashable|_
        The name of the to be returned series or, alternatively, the dataframe column.

    object_type : str
        The object type of the to be returned item.
        Accepted values are ``"Series"`` or ``"DataFrame"``.

    Returns
    -------
    |pd.Series|_ or |pd.DataFrame|_ [|pd.MultiIndex|_]:
        A pandas series or dataframe created fron **input_dict**.

    """
    # Construct a MultiIndex
    if not isinstance(input_dict, Settings):
        flat_dict = Settings(input_dict).flatten(flatten_list=False)
    else:
        flat_dict = input_dict.flatten(flatten_list=False)
    idx = pd.MultiIndex.from_tuples(flat_dict.keys())

    # Construct a DataFrame or Series
    pd_type = object_type.split('.')[-1].lower()
    ret = pd.Series(list(flat_dict.values()), index=idx, name=name)
    if pd_type == 'dataframe':
        ret = ret.to_frame()
    elif pd_type != 'series':
        raise ValueError("{} is not an accepted value for the keyword argument 'object_type'. "
                         "Accepted values are 'DataFrame' and 'Series'".format(str(object_type)))

    # Sort and return
    ret.sort_index(inplace=True)
    return ret


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
    raise ValueError('Could not construct a Pandas (Multi)Index from an \
                     {:d}-dimensional array'.format(ar.dim))


def get_example_xyz(name: Union[str, PathLike] = 'Cd68Se55_26COO_MD_trajec.xyz') -> str:
    """Return the path + name of the example multi-xyz file."""
    err = "'FOX.get_example_xyz()' has been deprecated in favour of 'FOX.example_xyz'"
    warnings.warn(err, FutureWarning)
    return resource_filename('FOX', join('data', name))


def _get_move_range(start: float = 0.005,
                    stop: float = 0.1,
                    step: float = 0.005,
                    ratio: Optional[Iterable[float]] = None) -> np.ndarray:
    """Generate an with array of all allowed moves.

    The move range spans a range of 1.0 +- **stop** and moves are thus intended to
    applied in a multiplicative manner (see :meth:`MonteCarlo.move_param`).

    Examples
    --------
    .. code:: python

        >>> move_range1 = _get_move_range(start=0.005, stop=0.020,
        ...                               step=0.005, ratio=None)
        >>> print(move_range1)
        [0.98  0.985 0.99  0.995 1.    1.005 1.01  1.015 1.02 ]

        >>> move_range2 = _get_move_range(start=0.005, stop=0.020,
        ...                               step=0.005, ratio=[1, 2, 4, 8])
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


def get_func_name(item: Callable) -> str:
    """Return the module + class + name of a function.

    Examples
    --------
    .. code:: python

        >>> import numpy as np
        >>> import FOX

        >>> func1 = FOX.MultiMolecule.init_rdf
        >>> get_func_name(func1)
        'FOX.MultiMolecule.init_rdf'

        >>> func2 = np.add
        >>> get_func_name(func2)
        'numpy.ufunc.add'

    Parameters
    ----------
    item : |Callable|_
        A function or method.

    Returns
    -------
    |str|_:
        The module + class + name of a function.

    """
    try:
        item_class, item_name = item.__qualname__.split('.')
        item_module = item.__module__.split('.')[0]
    except AttributeError:
        item_name = item.__name__
        item_class = item.__class__.__name__
        item_module = item.__class__.__module__.split('.')[0]
    return f'{item_module}.{item_class}.{item_name}'


def get_class_name(item: Callable) -> str:
    """Return the module + name of a class.

    Examples
    --------
    .. code:: python

        >>> import FOX

        >>> class1 = FOX.MultiMolecule
        >>> get_func_name(class1)
        'FOX.MultiMolecule'

        >>> class2 = float
        >>> get_func_name(class2)
        'builtins.float'

    Parameters
    ----------
    item : |Callable|_
        A class.

    Results
    -------
    |str|_:
        The module + name of a class.

    """
    item_class = item.__qualname__
    item_module = item.__module__.split('.')[0]
    if item_module == 'scm':
        item_module == item.__module__.split('.')[1]
    return f'{item_module}.{item_class}'


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


def get_nested_element(iterable: Iterable) -> Any:
    """Grab a (nested) non-iterable element in **iterable**.

    Recursivelly calls ``iter`` followed by ``next`` until maximum recursion depth is reached.

    Parameters
    ----------
    iterable : |Iterable|_
        An iterable.

    Returns
    -------
    object:
        A (nested) non-iterable element extracted from **iterable**.

    """
    ret = iterable
    while True:
        try:
            ret = next(iter(ret))
        except TypeError:
            return ret


def get_importable(string: str, validate: Optional[Callable[[T], bool]] = None) -> T:
    """Import an importable object.

    Parameters
    ----------
    string : str
        A string representing an importable object.
        Note that the string *must* contain the object's module.

    validate : :class:`~Collections.abc.Callable`, optional
        A callable for validating the imported object.
        Will raise an :exc:`AssertionError` if its output evaluates to ``False``.

    Returns
    -------
    :data:`~typing.Any`
        The import object

    """
    try:
        head, *tail = string.split('.')
    except ValueError as ex:
        raise ValueError("No module has been specified in the "
                         f"passed 'string' ({string!r})") from ex

    ret: T = importlib.import_module(head)  # type: ignore
    for name in tail:
        ret = getattr(ret, name)

    if validate is not None:
        msg = f'Passing {reprlib.repr(ret)} to {validate!r} failed to return True'
        assert validate(ret), msg

    return ret


def group_by_values(iterable: Iterable[Tuple[VT, KT]],
                    mapping_type: Type[Mapping] = dict) -> Dict[KT, List[VT]]:
    """Take an iterable, yielding 2-tuples, and group all first elements by the second.

    Exameple
    --------
    .. code:: python

        >>> from typing import Iterator

        >>> str_list: list = ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b']
        >>> iterable: Iterator = enumerate(str_list)
        >>> new_dict: dict = group_by_values(iterable)

        >>> print(new_dict)
        {'a': [1, 2, 3, 4, 5], 'b': [6, 7, 8]}

    Parameter
    ---------
    iterable : :class:`Iterable<collections.abc.Iterable>`
        An iterable yielding 2 elements upon iteration
        (*e.g.* :meth:`dict.items` or :func:`enumerate`).
        The second element must be a :class:`Hashable<collections.abc.Hashable>` and will be used
        as key in the to-be returned mapping.

    mapping_type : :class:`type` [:class:`MutableMapping<collections.abc.MutableMapping>`]
        The to-be returned mapping type.

    Returns
    -------
    :class:`MutableMapping<collections.abc.MutableMapping>`
    [:class:`Hashable<collections.abc.Hashable>`, :class:`list` [:data:`Any<typing.Any>`]]
        A grouped dictionary.

    """
    ret = {}
    list_append: Dict[Hashable, Callable[[VT], None]] = {}
    for value, key in iterable:
        try:
            list_append[key](value)
        except KeyError:
            ret[key] = [value]
            list_append[key] = ret[key].append

    return ret if mapping_type is dict else mapping_type(ret)


def read_rtf_file(filename: Union[AnyStr, PathLike]
                  ) -> Optional[Tuple[Sequence[str], Sequence[float]]]:
    """Return a 2-tuple with all atom types and charges."""
    def _parse_item(item: str) -> Tuple[str, float]:
        item_list = item.split()
        return item_list[j], float(item_list[k])

    i, j, k = len('ATOM'), 2, 3
    with open(filename, 'r') as f:
        ret = [_parse_item(item) for item in f if item[:i] == 'ATOM']
    return zip(*ret) if ret else None


def fill_diagonal_blocks(a: np.ndarray, i: int, j: int, val: float = np.nan) -> None:
    """Fill diagonal blocks in **a** of size :math:`i * j`.

    The blocks are filled along the last 2 axes in **ar**.
    Performs an inplace update of **a**.

    Examples
    --------
    .. code:: python

        >>> import numpy as np

        >>> a = np.zeros((10, 15), dtype=int)
        >>> i = 2
        >>> j = 3

        >>> fill_diagonal_blocks(ar, i, j, val=1)
        >>> print(ar)
        [[1 1 1 0 0 0 0 0 0 0 0 0 0 0 0]
         [1 1 1 0 0 0 0 0 0 0 0 0 0 0 0]
         [0 0 0 1 1 1 0 0 0 0 0 0 0 0 0]
         [0 0 0 1 1 1 0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 1 1 1 0 0 0 0 0 0]
         [0 0 0 0 0 0 1 1 1 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0 1 1 1 0 0 0]
         [0 0 0 0 0 0 0 0 0 1 1 1 0 0 0]
         [0 0 0 0 0 0 0 0 0 0 0 0 1 1 1]
         [0 0 0 0 0 0 0 0 0 0 0 0 1 1 1]]

    Parameters
    ----------
    a : :class:`nump.ndarray`
        A >= 2D NumPy array whose diagonal blocks are to be filled.
        Gets modified in-place.
    i : :class:`int`
        The size of the diagonal blocks along axis -2.
    j : :class:`int`
        The size of the diagonal blocks along axis -1.
    fill_value : :class:`float`
        Value to be written on the diagonal.
        Its type must be compatible with that of the array **a**.

    """
    if (j <= 0) or (i <= 0):
        raise ValueError(f"'i' and 'j' should be larger than 0; observed values: {i} & {j}")

    i0 = j0 = 0
    dim1 = a.shape[-2]
    while dim1 > i0:
        a[..., i0:i0+i, j0:j0+j] = val
        i0 += i
        j0 += j


def split_dict(dct: MutableMapping[KT, VT], keep_keys: Container[KT]) -> Dict[KT, VT]:
    """Pop all items from **dct** which are not in **keep_keys** and use them to construct a new dictionary.

    Note that, by popping its keys, the passed **dct** will also be modified inplace.

    Examples
    --------
    .. code:: python

        >>> from FOX.functions.utils import split_dict

        >>> dict1 = {1: 'a', 2: 'b', 3: 'c', 4: 'd'}
        >>> dict2 = split_dict(dict1, keep_keys=[1, 2])

        >>> print(dict1)
        {1: 'a', 2: 'b'}

        >>> print(dict2)
        {3: 'c', 4: 'd'}

    Parameters
    ----------
    dct : :class:`~collections.abc.MutableMapping`
        A mutable mapping.
    keep_keys : :class:`~collections.abc.Iterable`
        An iterable with keys that should remain in **dct**.

    Returns
    -------
    :class:`dict`
        A new dictionaries with all key/value pairs from **dct** not specified in **keep_keys**.

    """  # noqa
    # The ordering of dict elements is preserved in this manner,
    # as opposed to the use of set.difference()
    difference = [k for k in dct if k not in keep_keys]

    return {k: dct.pop(k) for k in difference}


def as_nd_array(value: Union[Scalar, Iterable[Scalar], SupportsArray], dtype: DtypeLike,
                ndmin: int = 1,
                copy: bool = False) -> np.ndarray:
    """Convert **value**, a scalar or iterable of scalars, into an array."""
    try:
        return np.array(value, dtype=dtype, ndmin=ndmin, copy=copy)

    except TypeError as ex:
        if not isinstance(value, abc.Iterable):
            raise ex

        ret = np.fromiter(value, dtype=dtype)
        ret.shape += (ndmin - ret.ndmim) * (1,)
        return ret
