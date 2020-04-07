"""
FOX.io.hdf5_utils
=================

Functions for storing Monte Carlo results in hdf5 format.

Index
-----
.. currentmodule:: FOX.io.hdf5_utils
.. autosummary::
    create_hdf5
    create_xyz_hdf5
    index_to_hdf5
    hdf5_availability
    to_hdf5
    from_hdf5
    dset_to_series
    dset_to_df

API
---
.. autofunction:: create_hdf5
.. autofunction:: create_xyz_hdf5
.. autofunction:: index_to_hdf5
.. autofunction:: hdf5_availability
.. autofunction:: to_hdf5
.. autofunction:: from_hdf5
.. autofunction:: dset_to_series
.. autofunction:: set_to_df

"""

import warnings
import subprocess
from os import remove, PathLike
from time import sleep
from os.path import isfile
from collections import abc
from typing import (Dict, Iterable, Optional, Union, Hashable, List, Tuple,
                    AnyStr, TYPE_CHECKING, Mapping, Iterator, Any)

import numpy as np
import pandas as pd
from scm.plams import Settings

from ..__version__ import __version__
from ..functions.utils import get_shape, assert_error, array_to_index, group_by_values

try:
    import h5py
    __all__ = ['create_hdf5', 'create_xyz_hdf5', 'to_hdf5', 'from_hdf5']
    H5PY_ERROR: Optional[str] = None

except ImportError:
    __all__ = []
    H5PY_ERROR: Optional[str] = (  # type: ignore
        "Use of the FOX.{} function requires the 'h5py' package."
        "\n'h5py' can be installed via anaconda with the following command:"
        "\n\tconda install --name FOX -y -c conda-forge h5py"
    )

if TYPE_CHECKING:
    from pandas.core.generic import NDFrame
    from ..classes.multi_mol import MultiMolecule
    from ..armc.armc import ARMC
    if H5PY_ERROR is not None:
        from h5py import File
    else:
        from ..type_alias import File
else:
    from ..type_alias import File, NDFrame, MultiMolecule, ARMC

"""################################### Creating .hdf5 files ####################################"""


@assert_error(H5PY_ERROR)
def create_hdf5(filename: Union[AnyStr, PathLike], armc: ARMC) -> None:
    r"""Create a hdf5 file to hold all addaptive rate Mone Carlo results (:class:`FOX.ARMC`).

    Datasets are created to hold a number of results following results over the course of the
    MC optimization:

    * The value of :math:`/phi` over the course of the parameter optimization (dataset: ``"phi"``)
    * The parameters (dataset: ``"param"``)
    * The acceptance rate (dataset: ``"acceptance"``)
    * The unmodified auxiliary error (dataset: ``"aux_error"``)
    * The modifications to the auxiliary error (dataset: ``"aux_error_mod"``)
    * User-specified PES descriptors (dataset(s): user-specified name(s))

    Cartesian coordinates (``"xyz"``) collected over the course of the current super-iteration are
    stored in a seperate temporary file.

    Parameters
    ----------
    filename : str
        The path+filename of the hdf5 file.

    armc : |FOX.ARMC|_
        An :class:`.ARMC` instance.

    """
    # Create a Settings object with the shape and dtype of all datasets
    kwarg_dict = _get_kwarg_dict(armc)

    # Create a hdf5 file with *n* datasets
    with h5py.File(filename, 'w-', libver='latest') as f:
        for key, kwargs in kwarg_dict.items():
            f.create_dataset(name=key, compression='gzip', **kwargs)
        f.attrs['super-iteration'] = -1
        f.attrs['sub-iteration'] = -1
        f.attrs['__version__'] = np.fromiter(__version__.split('.'), count=3, dtype=int)

    # Store the *index*, *column* and *name* attributes of dataframes/series in the hdf5 file
    kappa = armc.iter_len // armc.sub_iter_len
    idx = armc.param['param'][0].index.append(pd.MultiIndex.from_tuples([('', 'phi', '')]))
    aux_error_idx = list(armc.pes.keys())

    pd_dict = {
        'param': armc.param['param'][0],
        'phi': pd.Series(np.nan, index=np.arange(kappa), name='phi'),
        'aux_error': pd.Series(np.nan, index=aux_error_idx, name='aux_error'),
        'aux_error_mod': pd.Series(np.nan, index=idx, name='aux_error_mod')
    }

    for key, partial in armc.pes.items():
        ref = partial.ref
        pd_dict[key] = ref
        pd_dict[key + '.ref'] = ref
    index_to_hdf5(filename, pd_dict)


@assert_error(H5PY_ERROR)
def create_xyz_hdf5(filename: Union[AnyStr, PathLike],
                    mol_list: Iterable[MultiMolecule], iter_len: int) -> None:
    """Create the ``"xyz"`` datasets for :func:`create_hdf5` in the hdf5 file ``filename+".xyz"``.

    The ``"xyz"`` dataset is to contain Cartesian coordinates collected over the course of the
    current ARMC super-iteration.

    Parameters
    ----------
    filename : str
        The path+filename of the hdf5 file.
        **filename** will be appended with ``".xyz"``.

    mol_list : :class:`Iterable<collections.abc.Iterable>` [:class:`MultiMolecule`]
        An iterable consisting of MultiMolecule instances.

    iter_len: int
        The length of an ARMC sub-iterations.
        Determines how many MD trajectories can be stored in the .hdf5 file.

    """
    # Remove previous hdf5 xyz files
    filename_xyz = _get_filename_xyz(filename)
    if isfile(filename_xyz):
        remove(filename_xyz)

    # Create a new hdf5 xyz files
    with h5py.File(filename_xyz, 'w-', libver='latest') as f:
        f.attrs['__version__'] = np.fromiter(__version__.split('.'), count=3, dtype=int)

        for i, mol in enumerate(mol_list):
            key = f'xyz.{i}'
            f.create_dataset(
                name=key,
                compression='gzip',
                shape=(iter_len, 0, mol.shape[1], 3),
                dtype=np.dtype('float16'),
                maxshape=(iter_len, None, mol.shape[1], 3),
                fillvalue=np.nan
            )
            f[key].attrs['atoms'] = mol.symbol.astype('S')
            f[key].attrs['bonds'] = mol.bonds


@assert_error(H5PY_ERROR)
def index_to_hdf5(filename: Union[AnyStr, PathLike], pd_dict: Dict[str, NDFrame]) -> None:
    """Store the ``index`` and ``columns`` / ``name`` attributes of **pd_dict** in hdf5 format.

    Attributes are exported for all dataframes/series in **pd_dict** and skipped otherwise.
    The keys in **pd_dict**, together with the attribute names, are used for naming the datasets.

    Examples
    --------
    .. code-block:: python

        >>> pd_dict = {}
        >>> pd_dict['df'] = pd.DataFrame(np.random.rand(10, 10))
        >>> pd_dict['series'] = pd.Series(np.random.rand(10))
        >>> index_to_hdf5(pd_dict, name='my_file.hdf5')

        >>> with h5py.File('my_file.hdf5', 'r') as f:
        >>>     tuple(f.keys())
        ('df.columns', 'df.index', 'series.index', 'series.name')

    Parameter
    ---------
    filename : str
        The path+name of the hdf5 file.

    pd_dict : dict
        A dictionary with dataset names as keys and matching array-like objects as values.

    """
    attr_tup = ('index', 'columns', 'name')

    with h5py.File(filename, 'r+', libver='latest') as f:
        for key, value in pd_dict.items():
            for attr_name in attr_tup:
                if not hasattr(value, attr_name):
                    continue

                attr = getattr(value, attr_name)
                i = _attr_to_array(attr).T
                f[key].attrs.create(attr_name, i)


def _get_kwarg_dict(armc: ARMC) -> Settings:
    """Create a Settings instance with keyword arguments for h5py.Group.create_dataset.

    .. _h5py.Group.create_dataset: http://docs.h5py.org/en/stable/high/group.html#Group.create_dataset  # noqa

    Examples
    --------

    The output has the following general structure:

    .. code:: python

        >>> s = _get_kwarg_dict(armc)
        >>> print(s)
        key1:
             kwarg1: ...
             kwarg2: ...
             kwarg3: ...
        key2:
             kwarg1: ...
             kwarg2: ...
        ...

    Parameters
    ----------
    armc : |FOX.ARMC|_
        An :class:`.ARMC` instance.

    Returns
    -------
    |plams.Settings|_:
        A Settings instance with keyword arguments for h5py.Group.create_dataset_.

    """
    shape = armc.iter_len // armc.sub_iter_len, armc.sub_iter_len

    ret = Settings()
    ret.phi.shape = (shape[0], ) + armc.phi.phi.shape
    ret.phi.dtype = float
    ret.phi.fillvalue = np.nan

    ret.param.shape = shape + armc.param['param'].T.shape
    ret.param.dtype = float
    ret.param.fillvalue = np.nan

    ret.acceptance.shape = shape
    ret.acceptance.dtype = bool

    ret.aux_error.shape = shape + (len(armc.molecule), len(armc.pes) // len(armc.molecule))
    ret.aux_error.dtype = float
    ret.aux_error.fillvalue = np.nan
    ret.aux_error_mod.shape = shape + (1 + len(armc.param['param']), )
    ret.aux_error_mod.dtype = float
    ret.aux_error_mod.fillvalue = np.nan

    for key, partial in armc.pes.items():
        ref = partial.ref

        ret[key].shape = shape + get_shape(ref)
        ret[key].dtype = float
        ret[key].fillvalue = np.nan

        ret[key + '.ref'].shape = get_shape(ref)
        ret[key + '.ref'].dtype = float
        ret[key + '.ref'].data = ref
        ret[key + '.ref'].fillvalue = np.nan

    return ret


"""################################### Updating .hdf5 files ####################################"""


@assert_error(H5PY_ERROR)
def hdf5_clear_status(filename: Union[AnyStr, PathLike]) -> bool:
    """Run the :code:`h5clear filename` command if **filename** refuses to open."""
    try:
        with h5py.File(filename, 'r+', libver='latest'):
            return True
    except OSError:
        subprocess.run(['h5clear', '-s', 'repr(filename)'])
        return False


@assert_error(H5PY_ERROR)
def hdf5_availability(filename: Union[AnyStr, PathLike], timeout: float = 5.0,
                      max_attempts: Optional[int] = 10) -> None:
    """Check if a .hdf5 file is opened by another process; return once it is not.

    If two processes attempt to simultaneously open a single hdf5 file then
    h5py will raise an :class:`OSError`.
    The purpose of this function is ensure that a .hdf5 is actually closed,
    thus allowing :func:`to_hdf5` to safely access **filename** without the risk of raising
    an :class:`OSError`.

    Parameters
    ----------
    filename : str
        The path+filename of the hdf5 file.

    timeout : float
        Time timeout, in seconds, between subsequent attempts of opening **filename**.

    max_attempts : int
        Optional: The maximum number attempts for opening **filename**.
        If the maximum number of attempts is exceeded, raise an ``OSError``.

    Raises
    ------
    OSError
        Raised if **max_attempts** is exceded.

    """
    warning = "WARNING: {!r} is currently unavailable; repeating attempt in {:.0f} seconds"
    i = max_attempts if max_attempts is not None else np.inf
    if i <= 0:
        raise ValueError(f"'max_attempts' must be larger than 0; observed value: {max_attempts!r}")

    while i:
        try:
            with h5py.File(filename, 'r+', libver='latest'):
                return  # the .hdf5 file can safely be opened
        except OSError as ex:  # the .hdf5 file cannot be safely opened yet
            warning_ = RuntimeWarning(warning.format(filename, timeout))
            warning_.__cause__ = ex
            warnings.warn(warning_)

            error = ex
            sleep(timeout)
        i -= 1
    raise error


@assert_error(H5PY_ERROR)
def to_hdf5(filename: Union[AnyStr, PathLike], dset_dict: Mapping[str, np.ndarray],
            kappa: int, omega: int) -> None:
    r"""Export results from **dset_dict** to the hdf5 file **filename**.

    All items in **dset_dict**, except ``"xyz"``, are exported to **filename**.
    The float or :class:`MultiMolecule` instance stored under the ``"xyz"`` key are
    exported to a seperate .hdf5 file (see :func:`._xyz_to_hdf5`).

    Parameters
    ----------
    filename : str
        The path+filename of the hdf5 file.

    dset_dict : dict [str, |np.ndarray|_]
        A dictionary with dataset names as keys and matching array-like objects as values.

    kappa : int
        The super-iteration, :math:`\kappa`, in the outer loop of :meth:`.ARMC.__call__`.

    omega : int
        The sub-iteration, :math:`\omega`, in the inner loop of :meth:`.ARMC.__call__`.

    """
    # Check if the hdf5 file is already opened. If opened: wait for 5 sec and try again.
    hdf5_availability(filename)

    # Update the hdf5 file
    with h5py.File(filename, 'r+', libver='latest') as f:
        f.attrs['super-iteration'] = kappa
        f.attrs['sub-iteration'] = omega
        try:
            for key, value in dset_dict.items():
                if key == 'xyz':
                    continue
                elif key == 'phi':
                    f[key][kappa] = value
                else:
                    f[key][kappa, omega] = value
        except Exception as ex:
            cls = type(ex)
            raise cls(f"dataset {key!r}: {ex}") from ex

    # Update the second hdf5 file with Cartesian coordinates
    filename_xyz = _get_filename_xyz(filename)
    _xyz_to_hdf5(filename_xyz, omega, dset_dict['xyz'])


@assert_error(H5PY_ERROR)
def _xyz_to_hdf5(filename: Union[AnyStr, PathLike], omega: int,
                 mol_list: Union[Iterable[MultiMolecule], Iterable[float], float]) -> None:
    r"""Export **mol** to the hdf5 file **filename**.

    Parameters
    ----------
    filename : str
        The path+filename of the hdf5 file.

    omega : int
        The sub-iteration, :math:`\omega`, in the inner loop of :meth:`.ARMC.__call__`.

    mol_list : |list|_ [|FOX.MultiMolecule|_] or |float|_
        All to-be exported :class:`MultiMolecule` instance(s) or float (*e.g.* ``np.nan``).

    """
    # Check if the hdf5 file is already opened. If opened: wait for 5 sec and try again.
    hdf5_availability(filename)

    with h5py.File(filename, 'r+', libver='latest') as f:
        if not isinstance(mol_list, abc.Iterable):  # Check if mol_list is a scalar (np.nan)
            i = 0
            while True:
                try:
                    f[f'xyz.{i}'][omega] = mol_list if mol_list is not None else np.nan
                    i += 1
                except KeyError:
                    return None

        enumerator: Iterator[Tuple[int, Union[MultiMolecule, float]]] = enumerate(mol_list)
        for i, mol in enumerator:
            dset = f[f'xyz.{i}']
            if not isinstance(mol, abc.Iterable):  # Check if mol is a scalar (np.nan)
                dset[omega] = mol if mol is not None else np.nan
                continue

            if len(mol) <= dset.shape[1]:
                dset[omega, 0:len(mol)] = mol
            else:  # Resize and try again
                dset.resize(len(mol), axis=1)
                dset[omega] = mol

    return None


"""#################################### Reading .hdf5 files ####################################"""

DataSets = Union[Hashable, Iterable[Hashable]]


@assert_error(H5PY_ERROR)
def from_hdf5(filename: Union[AnyStr, PathLike],
              datasets: Optional[DataSets] = None
              ) -> Union[NDFrame, Dict[Hashable, NDFrame]]:
    """Retrieve all user-specified datasets from **name**.

    Values are returned in dictionary of DataFrames and/or Series.

    Parameters
    ----------
    filename : str
        The path+name of the hdf5 file.
    datasets : list [str]
        A list of to be retrieved dataset names.
        All datasets will be retrieved if ``None``.

    Returns
    -------
    |dict|_ [|str|_, (|pd.DataFrame|_ and/or |pd.Series|_)]:
        A dicionary with dataset names as keys and the matching data as values.

    """
    with h5py.File(filename, 'r', libver='latest') as f:
        # Retrieve all values up to and including the current iteration
        kappa = f.attrs['super-iteration']
        omega = f.attrs['sub-iteration']
        omega_max = f['param'].shape[1]
        i = kappa * omega_max + omega

        # Identify the to-be returned datasets
        if isinstance(datasets, str):
            datasets_: Iterable[Hashable] = (datasets, )
        elif datasets is None:
            datasets_ = f.keys()
        else:
            datasets_ = datasets  # type: ignore

        # Retrieve the datasets
        try:
            ret = {key: _get_dset(f, key)[:i+1] for key in datasets_}
        except KeyError as ex:
            raise KeyError(f"No dataset {ex} in {filename!r}. The following datasets are "
                           f"available: {list(f.keys())!r}") from ex

    # Return a DataFrame/Series or dictionary of DataFrames/Series
    if len(ret) == 1:
        for i in ret.values():
            return i
    return ret


@assert_error(H5PY_ERROR)
def _get_dset(f: File, key: Hashable) -> Union[pd.Series, pd.DataFrame, List[pd.DataFrame]]:
    """Take a h5py dataset and convert it into either a Series or DataFrame.

    See :func:`FOX.dset_to_df` and :func:`FOX.dset_to_series` for more details.

    Parameters
    ----------
    f : |h5py.File|_
        An opened hdf5 file.

    key : str
        The dataset name.

    Returns
    -------
    |pd.DataFrame|_, |pd.Series|_ or |np.ndarray|_:
        A NumPy array or a Pandas DataFrame or Series retrieved from **key** in **f**.

    """
    if key == 'phi':
        return dset_to_series(f, key).T

    if key == 'aux_error':
        return _aux_err_to_df(f, key)

    elif 'columns' in f[key].attrs.keys():
        return dset_to_df(f, key)

    elif 'name' in f[key].attrs.keys():
        return dset_to_series(f, key)

    elif f[key].ndim == 2:
        return pd.Series(f[key][:].flatten(), name=key)

    elif f[key].ndim == 3:
        data = f[key][:]
        data.shape = np.product(data.shape[:-1]), -1
        columns = pd.MultiIndex.from_product([[key], np.arange(data.shape[-1])])
        return pd.DataFrame(data, columns=columns)

    raise ValueError(key, f[key].ndim)


@assert_error(H5PY_ERROR)
def _get_xyz_dset(f: File) -> Tuple[np.ndarray, Dict[str, List[int]]]:
    """Return the ``"xyz"zz dataset from **f**.

    Parameters
    ----------
    f : |h5py.File|_
        An opened hdf5 file.

    Returns
    -------
    :math:`k*m*n*3` |np.ndarray|_ and |dict|_:
        An array of :math:`k` MultiMolecule instances with :math:`m` molecules and
        :math:`n` atoms.

    """
    key = 'xyz'

    # Construct a dictionary with atomic symbols and matching atomic indices
    iterator = enumerate(f[key].attrs['atoms'])
    idx_dict = group_by_values(iterator)

    # Extract the Cartesian coordinates; sort in chronological order
    i = f.attrs['sub-iteration']
    j = f[key].shape[0] - i
    ret = np.empty_like(f[key])
    ret[:j] = f[key][i:]
    ret[j:] = f[key][:i]
    return ret, idx_dict


"""###################################### hdf5 utilities #######################################"""


def _get_filename_xyz(filename: Union[AnyStr, PathLike], **kwargs: Any) -> str:
    """Construct a filename for the xyz-containing .hdf5 file.

    Parameters
    ----------
    filename : str
        The base filename.
        If possible, ``".xyz"`` is inserted between **filename** and its extensions.
        If not, then **filename** is appended with ``".xyz"``.

    Returns
    -------
    |str|_:
        The filename of the xyz-containing .hdf5 file.

    """
    if isinstance(filename, bytes):
        filename_ = filename.decode(**kwargs)
    elif isinstance(filename, PathLike):
        filename_ = str(filename)
    else:
        filename_ = filename

    if '.hdf5' in filename_:
        return filename_.replace('.hdf5', '.xyz.hdf5')
    return f'{filename_}.xyz'


def _attr_to_array(index: Union[str, pd.Index]) -> np.ndarray:
    """Convert an attribute value, retrieved from :func:`FOX.index_to_hdf5`, into a NumPy array.

    Accepts strings and instances of pd.Index.

    Examples
    --------
    .. code-block:: python

        >>> item = 'name'
        >>> _attr_to_array(item)
        array([b'name'], dtype='|S4')

        >>> item = pd.RangeIndex(stop=4)
        >>> _attr_to_array(item)
        array([0, 1, 2, 3, 4])

    Parameters
    ----------
    item : str or |pd.Index|_
        A string or instance of pd.Index (or one of its subclasses).

    Returns
    -------
    |np.ndarray|_:
        An array created fron **item**.

    """
    # If **idx** does not belong to the pd.Index class or one of its subclass
    if not isinstance(index, pd.Index):  # **item** belongs to the *name* attribute of pd.Series
        return np.array(index, dtype='S', ndmin=1, copy=False)

    # Convert **item** into an array
    ret = np.array(index.to_list())
    if 'str' in ret.dtype.name or ret.dtype == object:  # h5py does not support unicode strings
        return ret.astype('S', copy=False)  # Convert to byte strings
    return ret


@assert_error(H5PY_ERROR)
def dset_to_series(f: File, key: Hashable) -> Union[pd.Series, pd.DataFrame]:
    """Take a h5py dataset and convert it into a Pandas Series (if 1D) or Pandas DataFrame (if 2D).

    Parameters
    ----------
    f : |h5py.File|_
        An opened hdf5 file.

    key : str
        The dataset name.

    Returns
    -------
    |pd.Series|_ or |pd.DataFrame|_:
        A Pandas Series or DataFrame retrieved from **key** in **f**.

    """
    name = f[key].attrs['name'][0].decode()
    index = array_to_index(f[key].attrs['index'][:])
    data = f[key][:]
    data.shape = np.product(data.shape[:-1], dtype=int), -1

    # Return a Series or DataFrame
    if data.ndim == 1:
        return pd.Series(f[key][:], index=index, name=name)
    else:
        columns = index
        index = pd.Index(np.arange(data.shape[0]), name=name)
        df = pd.DataFrame(data, index=index, columns=columns)
        if key == 'aux_error_mod':
            df.set_index(list(df.columns[0:-1]), inplace=True)
            df.columns = pd.Index(['phi'])
            return df['phi']
        return df


@assert_error(H5PY_ERROR)
def dset_to_df(f: File, key: Hashable) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """Take a h5py dataset and create a DataFrame (if 2D) or list of DataFrames (if 3D).

    Parameters
    ----------
    f : |h5py.File|_
        An opened hdf5 file.

    key : str
        The dataset name.

    Returns
    -------
    |pd.DataFrame|_ or |list|_ [|pd.DataFrame|_]:
        A Pandas DataFrame retrieved from **key** in **f**.

    """
    columns = array_to_index(f[key].attrs['columns'][:])
    index = array_to_index(f[key].attrs['index'][:])
    data = f[key][:]
    data.shape = np.product(data.shape[:-2], dtype=int), data.shape[-2], -1

    # Return a DataFrame or list of DataFrames
    if data.ndim == 2:
        return pd.DataFrame(data, index=index, columns=columns)
    else:
        return [pd.DataFrame(i, index=index, columns=columns) for i in data]


@assert_error(H5PY_ERROR)
def _aux_err_to_df(f: File, key: Hashable) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """Take a h5py dataset and create a DataFrame (if 2D) or list of DataFrames (if 3D).

    Parameters
    ----------
    f : |h5py.File|_
        An opened hdf5 file.

    key : str
        The dataset name.

    Returns
    -------
    |pd.DataFrame|_ or |list|_ [|pd.DataFrame|_]:
        A Pandas DataFrame retrieved from **key** in **f**.

    """
    columns = array_to_index(f[key].attrs['index'][:])
    data = f[key][:]
    data.shape = np.product(data.shape[:-2], dtype=int), -1

    ret = pd.DataFrame(data, columns=columns)
    ret.index.name = f[key].attrs['name'][0].decode()
    return ret


Tuple3 = Tuple[np.ndarray, Dict[str, List[int]], np.ndarray]


@assert_error(H5PY_ERROR)
def mol_from_hdf5(filename: Union[AnyStr, PathLike],
                  i: int = -1, j: int = 0) -> Tuple3:
    """Read a single dataset from a (multi) .xyz.hdf5 file.

    Returns values for the :class:`MultiMolecule` ``coords``, ``atoms`` and ``bonds`` parameters.

    Parameters
    ----------
    filename : :class:`str`
        The path+name of the .xyz.hdf5 file.

    i : :class:`int`
        The (sub-)iteration number of the to-be returned trajectory.

    j : :class:`int`
        The index of the to-be returned dataset.
        For example: :code:`j=0`is equivalent to the :code:`'xyz.0'` dataset.

    Returns
    -------
    :class:`numpy.ndarray`, :class:`dict` [:class:`str`, :class:`list` [:class:`int`]] and :class:`numpy.ndarray`.
        * A 3D array with Cartesian coordinates of :math:`m` molecules with :math:`n` atoms.
        * A dictionary with atomic symbols as keys and lists of matching atomic indices as values.
        * A 2D array of bonds and bond-orders.

    """  # noqa
    with h5py.File(filename, 'r', libver='latest') as f:
        dset = f[f'xyz.{j}']
        return (
            dset[i],
            group_by_values(enumerate(dset.attrs['atoms'].astype(str).tolist())),
            dset.attrs['bonds']
        )
