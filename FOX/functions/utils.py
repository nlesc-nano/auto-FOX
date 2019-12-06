"""A module with miscellaneous functions."""

import warnings
from os.path import join, isfile
from functools import wraps
from pkg_resources import resource_filename
from typing import (
    Iterable, Tuple, Callable, Hashable, Sequence, Optional, List, Any, TypeVar, Dict,
    Type, MutableMapping
)

import yaml
import numpy as np
import pandas as pd

from scm.plams import (Settings, add_to_class)

__all__ = ['get_template', 'template_to_df', 'get_example_xyz']

T = TypeVar('T')


def append_docstring(item: Callable) -> Callable:
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


def assert_error(error_msg: str = '') -> Callable:
    """Take an error message, if not ``false`` then cause a function or class to raise a ModuleNotFoundError upon being called.

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
    error_msg : str
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


def read_str_file(filename: str) -> Optional[Tuple[Sequence, Sequence]]:
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


def get_example_xyz(name: str = 'Cd68Se55_26COO_MD_trajec.xyz') -> str:
    """Return the path + name of the example multi-xyz file."""
    err = "'FOX.get_example_xyz()' has been deprecated in favour of 'FOX.example_xyz'"
    warnings.warn(err, FutureWarning)
    return resource_filename('FOX', join('data', name))


def _get_move_range(start: float = 0.005, stop: float = 0.1,
                    step: float = 0.005) -> np.ndarray:
    """Generate an with array of all allowed moves.

    The move range spans a range of 1.0 +- **stop** and moves are thus intended to
    applied in a multiplicative manner (see :meth:`MonteCarlo.move_param`).

    Examples
    --------
    .. code:: python

        >>> move_range = _get_move_range(start=0.005, stop=0.1, step=0.005)
        >>> print(move_range)
        [0.9   0.905 0.91  0.915 0.92  0.925 0.93  0.935 0.94  0.945
         0.95  0.955 0.96  0.965 0.97  0.975 0.98  0.985 0.99  0.995
         1.005 1.01  1.015 1.02  1.025 1.03  1.035 1.04  1.045 1.05
         1.055 1.06  1.065 1.07  1.075 1.08  1.085 1.09  1.095 1.1  ]

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


def slice_str(str_: str, intervals: List[Optional[int]], strip_spaces: bool = True) -> list:
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


def get_atom_count(iterable: Iterable[Sequence[str]], mol: 'FOX.MultiMolecule') -> List[int]:
    """Count the occurences of each atom/atom-pair (from **iterable**) in **mol**.

    Duplicate values are removed if when evaluating atom pairs when atom-pairs consist of
    identical atom types (*e.g.* ``"Cd Cd"``).

    Examples
    --------
    .. code:: python

        >>> Cd_count = len(mol.atoms['Cd'])
        >>> print(Cd_count)
        50

        >>> Se_count = len(mol.atoms['Se'])
        >>> print(Se_count)
        25

        >>> iterable = ['Cd', 'Cd Se', 'Cd Cd']
        >>> at_count = get_atom_count(get_atom_count)
        >>> print(at_count)
        [50, 1250, 1225]

    Parameters
    ----------
    iterable : |Iterable|_ [|Sequence|_ [str]]
        A nested iterable with :math:`n` atoms and/or atom pairs.
        Atom pairs are to-be provided as space seperated strings (*e.g.* ``"Cd Cd"``).

    mol : |FOX.MultiMolecule|_
        A :class:`.MultiMolecule` instance.

    Returns
    -------
    :math:`n` |list|_ [|int|_]:
        A list of atom(-pair) counts.

    """
    if isinstance(mol, pd.Series):
        at_list = pd.Series(mol.index, index=mol)
    else:
        at_list = mol.atoms

    def _get_atom_count(at):
        atoms = at.split()
        if len(atoms) == 2 and atoms[0] == atoms[1]:
            at1, _ = [len(at_list[i]) for i in atoms]
            return (at1**2 - at1) // 2
        elif len(atoms) == 2:
            return np.product([len(at_list[i]) for i in atoms])
        else:
            return len(at_list[at])

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
    ret = iterable
    while True:
        try:
            ret = next(iter(ret))
        except TypeError:
            return ret


def str_to_callable(string: str) -> Callable:
    """Create a callable object from a string.

    Accepts string-representations of functions, classes and methods, returning the respective
    callable.

    Examples
    --------
    An example with a builtin function:

    .. code:: python

        >>> callable_ = str_to_callable('len')
        >>> print(callable_)
        <function len(obj, /)>

        >>> out = callable_([True, True, True])
        >>> print(out)
        3

    An example with a third-party function from NumPy:

    .. code:: python

        >>> callable_ = str_to_callable('numpy.add')
        >>> print(callable_)
        <ufunc 'add'>

        >>> out = callable_(10, 5)
        >>> print(out)
        15

    Another example with a third-party method from the :class:`.MultiMolecule` class in Auto-FOX:

    .. code:: python

        >>> from FOX import get_example_xyz

        >>> callable_ = str_to_callable('FOX.MultiMolecule.from_xyz')
        >>> print(callable_)
        <bound method MultiMolecule.from_xyz of <class 'FOX.classes.multi_mol.MultiMolecule'>>

        >>> out = callable_(get_example_xyz())
        >>> print(type(out))
        <class 'FOX.classes.multi_mol.MultiMolecule'>


    Parameters
    ----------
    string : str
        A string represnting a callable object.
        The path to the callable should be included in the string (see examples).

    Returns
    -------
    |Callable|_:
        A callable object (*e.g.* function, class or method).

    """
    if '.' not in string:  # Builtin function or class
        return eval(string)

    elif string.count('.') == 1:
        try:  # Builtin method
            return eval(string)
        except NameError:  # Non-builtin function or class
            package, func = string.split('.')
            exec(f'from {package} import {func}')
            return eval(func)

    else:
        try:  # Non-builtin function or class
            package, func = string.rsplit('.', 1)
            exec(f'from {package} import {func}')
            return eval(func)
        except ImportError:  # Non-builtin method
            package, class_, method = string.rsplit('.', 2)
            exec(f'from {package} import {class_}')
            return eval('.'.join([class_, method]))


def group_by_values(iterable: Iterable[Tuple[Any, Hashable]],
                    mapping_type: Type[MutableMapping] = dict
                    ) -> MutableMapping[Hashable, List[Any]]:
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
    ret: MutableMapping[Hashable, List[Any]] = mapping_type()
    list_append: Dict[Hashable, list.append] = {}
    for value, key in iterable:
        try:
            list_append[key](value)
        except KeyError:
            ret[key] = [value]
            list_append[key] = ret[key].append
    return ret


def read_rtf_file(filename: str) -> Optional[Tuple[Sequence[str], Sequence[float]]]:
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
