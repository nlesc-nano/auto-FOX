"""A module with miscellaneous functions."""

from typing import (
    Iterable, Tuple, Callable, Hashable, Sequence, MutableSequence, Optional, List, Any
)
from os.path import join, isfile
from functools import wraps
from pkg_resources import resource_filename

import yaml
import numpy as np
import pandas as pd

from scm.plams import (Settings, add_to_class)

__all__ = ['get_template', 'template_to_df', 'get_example_xyz']


def append_docstring(item: Callable) -> Callable:
    r"""A decorator for appending the docstring of a Callable one provided by another Callable.

    Examples
    --------
    .. code:: python

        >>> def func1():
        >>>     """'func1 docstring'"""
        >>>     pass

        >>> @append_docstring(func1)
        >>> def func2():
        >>>     """'func2 docstring'"""
        >>>     pass

        >>> help(func2)
        'func2 docstring'

        'func1 docstring'

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
            func.__doc__ += '\n\n' + item.__doc__
        except TypeError:
            pass
        return func
    return decorator


def assert_error(error_msg: str = '') -> Callable:
    """Take an error message, if not ``false`` then cause a function or class
    to raise a ModuleNotFoundError upon being called.

    Indended for use as a decorater:

    Examples
    --------
    .. code:: python

        >>> @assert_error('An error was raised by {}')
        >>> def my_custom_func():
        >>>     print(True)

        >>> my_custom_func()
        ModuleNotFoundError: An error was raised by my_custom_func

    Parameters
    ----------
    error_msg : str
        A to-be printed error message.
        If available, a single set of curly brackets will be replaced
        with the function or class name.

    Returns
    -------
    |Callable|_
        A decorated callable.

    """
    type_dict = {'function': _function_error, 'type': _class_error}

    def decorator(func):
        return type_dict[func.__class__.__name__](func, error_msg)
    return decorator


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


def get_template(name: str,
                 path: str = None,
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


def template_to_df(name: str,
                   path: str = None) -> pd.DataFrame:
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


def serialize_array(array: np.ndarray,
                    items_per_row: int = 4) -> str:
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


def read_str_file(filename: str) -> Optional[zip]:
    """Read atomic charges from CHARMM-compatible stream files (.str).

    Returns a settings object with atom types and (atomic) charges.

    Parameters
    ----------
    filename : str
        the path+filename of the .str file.

    Returns
    -------
    |plams.Settings|_ [|str|_, |tuple|_ [|str|_ or |float|_]]:
        A settings object with atom types and (atomic) charges.

    """
    def inner_loop(f):
        ret = []
        for j in f:
            if j != '\n':
                j = j.split()[2:4]
                ret.append((j[0], float(j[1])))
            else:
                return ret

    with open(filename, 'r') as f:
        for i in f:
            if 'GROUP' in i:
                return zip(*inner_loop(f))


def get_shape(item: Iterable) -> Tuple[int]:
    """Try to infer the shape of an object.

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
    elif hasattr(item, '__len__'):  # Plan B: **item** has access to the __len__() magic method
        return (len(item), )
    return (1, )  # Plan C: **item** has access to neither A nor B


def flatten_dict(input_dict: dict,
                 clip: Optional[int] = None) -> dict:
    """Flatten a nested dictionary.

    The keys of the to be returned dictionary consist are tuples with the old (nested) keys
    of **input_dict**.

    Examples
    --------
    .. code-block:: python

        >>> print(input_dict)
        {'a': {'b': {'c': True}}}

        >>> output_dict = flatten_dict(input_dict)
        >>> print(output_dict)
        {('a', 'b', 'c'): True}

    Parameters
    ----------
    input_dict : dict
        A (nested) dicionary.

    clip : int
        The maximum length of the tuple keys.
        The maximum length is enforced by concatenating the first
        :math:`n` elements of a tuple key into a string (if necessary).

    Returns
    -------
    |dict|_ [|tuple|_, object]:
        A non-nested dicionary derived from **input_dict**.

    """
    def concatenate(key_ret, dict_):
        for key, value in dict_.items():
            key = key_ret + (key, )
            if isinstance(value, dict):
                concatenate(key, value)
            elif clip is None:
                ret[key] = value
            else:
                i = len(key) - clip + 1
                try:
                    key = (' '.join(key[:i]), ) + key[i:]
                except TypeError:  # Try harder
                    key = (' '.join(str(j) for j in key[:i]), ) + key[i:]
                ret[key] = value

    # Changes keys into tuples
    ret = input_dict.__class__()
    concatenate((), input_dict)
    return ret


def dict_to_pandas(input_dict: dict,
                   name: Hashable = 0,
                   object_type: str = 'pd.DataFrame') -> pd.DataFrame:
    """Turn a nested dictionary into a pandas series or dataframe.

    Keys are un-nested and used for generating multiindices (see meth:`flatten_dict`).

    Parameters
    ----------
    input_dict : dict
        A nested dictionary.

    name : |Hashable|_
        The name of the to be returned series or dataframe column.

    object_type : str
        The object type of the to be returned item.
        Accepted values are ``"Series"`` or ``"DataFrame"``.

    Returns
    -------
    |pd.Series|_ or |pd.DataFrame|_ [|pd.MultiIndex|_]:
        A pandas series or dataframe created fron **input_dict**.

    """
    # Construct a MultiIndex
    flat_dict = flatten_dict(input_dict, clip=2)
    idx = pd.MultiIndex.from_tuples(flat_dict.keys())

    # Construct a DataFrame or Series
    pd_type = object_type.split('.')[-1].lower()
    if pd_type == 'series':
        ret = pd.Series(list(flat_dict.values()), index=idx, name=name)
    elif pd_type == 'dataframe':
        ret = pd.DataFrame(list(flat_dict.values()), index=idx, columns=[name])
    else:
        raise ValueError("{} is not an accepted value for the keyword argument 'object_type'."
                         "Accepted values are 'DataFrame' or 'Series'".format(str(object_type)))

    # Sort and return
    ret.sort_index(inplace=True)
    return ret


def array_to_index(ar: np.ndarray) -> pd.Index:
    """Convert a NumPy array into a Pandas Index or MultiIndex.

    Raises a ``ValueError`` if the dimensionality of **ar** is greater than 2.

    Parameters
    ----------
    ar : |np.ndarray|_
        A 1D or 2D NumPy array.

    Results
    -------
    |pd.Index|_ or |pd.MultiIndex|_:
        A Pandas Index or MultiIndex constructed from **ar**.

    """
    if 'bytes' in ar.dtype.name:
        ar = ar.astype(str, copy=False)

    if ar.ndim == 1:
        return pd.Index(ar)
    elif ar.ndim == 2:
        return pd.MultiIndex.from_arrays(ar)
    raise ValueError('Could not construct a Pandas (Multi)Index from an \
                     {:d}-dimensional array'.format(ar.dim))


def get_example_xyz(name: str = 'Cd68Se55_26COO_MD_trajec.xyz') -> str:
    """Return the path + name of the example multi-xyz file."""
    return resource_filename('FOX', join('data', name))


def _get_move_range(start: float = 0.005,
                    stop: float = 0.1,
                    step: float = 0.005) -> np.ndarray:
    """Generate an with array of all allowed moves.

    Moves span both the positive and negative range.

    Parameters
    ----------
    start : float
        Start of the interval.
        The interval includes this value.

    stop : float
        End of the interval.
        The interval includes this value.

    step : float
        Spacing between values.

    Returns
    -------
    |np.ndarray|_ [|np.int64|_]:
        An array with allowed moves.

    """
    rng_range1 = np.arange(1 + start, 1 + stop, step, dtype=float)
    rng_range2 = np.arange(1 - stop, 1 - start + step, step, dtype=float)
    ret = np.concatenate((rng_range1, rng_range2))
    ret.sort()
    return ret


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
    return '{}.{}.{}'.format(item_module, item_class, item_name)


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
    return '{}.{}'.format(item_module, item_class)


def slice_str(str_: str,
              intervals: list,
              strip_spaces: bool = True) -> list:
    """Slice a string, **str_**, at intervals specified in **intervals**.

    Examples
    --------
    .. code:: python
        >>> my_str = '123456789'
        >>> intervals = [None, 3, 6, None]
        >>> slice_str(my_str, intervals)
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


def get_nested_value(iterable: Sequence,
                     key_tup: Iterable[Hashable]) -> Any:
    """Retrieve a value, associated with all keys in **key_tup**, from a nested iterable.

    The following two expressions are equivalent:

    .. code:: python

        >>> get_nested_value(iterable, ('a', 'b', 3))
        >>> iterable['a']['b'][3]

    The function calls the **iterable**.__getitem__() method (recursivelly) untill all keys in
    **key_tup** are exhausted. Works on any iterable container whose elements can be
    accessed via their index or key (*e.g.* lists, tuples and dictionaries).

    Parameters
    ----------
    iterable : object
        A (nested) iterable container such as a list, tuple or dictionary.

    key_tup : |Sequence|_ [|Hashable|_]
        A sequence of nested keys and/or indices.

    Returns
    -------
    object:
        The value in **iterable** associated with all keys in **key**.

    """
    iter_slice = iterable
    for i in key_tup:
        iter_slice = iter_slice[i]
    return iter_slice


def set_nested_value(iterable: MutableSequence,
                     key_tup: Sequence[Hashable],
                     value: Any) -> None:
    """Assign a value, associated with all keys and/or indices in **key_tup**,
    to a nested iterable.

    The following two expressions are equivalent:

    .. code:: python

        >>> set_nested_value(iterable, ('a', 'b', 3), True)
        >>> iterable['a']['b'][3] = True

    The function calls the **iterable**.__getitem__() method (recursivelly) untill all keys in
    **key_tup** are exhausted. Works on any mutable iterable container whose elements can be
    accessed via their index or key (*e.g.* lists and dictionaries).

    Parameters
    ----------
    iterable : |MutableSequence|_
        A mutable (nested) iterable container such as a list or dictionary.

    key_tup : |Sequence|_ [|Hashable|_]
        A sequence of nested keys and/or indices.

    value : object
        The to-be assigned value.

    """
    iter_slice = iterable
    for i in key_tup[:-1]:
        iter_slice = iter_slice[i]
    iter_slice[key_tup[-1]] = value


def get_atom_count(iterable: Iterable[Sequence[str]],
                   mol: 'FOX.MultiMolecule') -> List[int]:
    """Count the occurences of each atom/atom-pair (from **iterable**) in **mol**.

    Parameters
    ----------
    iterable : |Iterable|_ [|Sequence|_ [str]]
        A nested iterable with :math:`n` atoms and/or atom pairs.

    mol : |FOX.MultiMolecule|_
        A :class:`.MultiMolecule` instance.

    Returns
    -------
    :math:`n` |list|_ [|int|_]:
        A list of atom(-pair) counts.

    """
    def _get_atom_count(at):
        at_list = at.split()
        if len(at_list) == 2 and at_list[0] == at_list[1]:
            at1, _ = [len(mol.atoms[i]) for i in at_list]
            return (at1**2 - at1) // 2
        elif len(at_list) == 2:
            return np.product([len(mol.atoms[i]) for i in at_list])
        else:
            return len(mol.atoms[at])

    return [_get_atom_count(at) for *_, at in iterable]


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
    item = iterable
    while True:
        try:
            item = next(iter(item))
        except TypeError:
            return item
