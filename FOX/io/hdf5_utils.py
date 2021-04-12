"""Functions for storing Monte Carlo results in hdf5 format.

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

from __future__ import annotations

import warnings
import subprocess
from os import remove, PathLike, fsdecode
from time import sleep
from os.path import isfile
from typing import (
    Dict, Iterable, Optional, Union, List, Tuple, TYPE_CHECKING, Mapping, Any, overload,
)

import h5py
import numpy as np
import pandas as pd
from scm.plams import Settings
from nanoutils import PathType, group_by_values, recursive_keys

from ..__version__ import __version__
from ..utils import get_shape, array_to_index

if TYPE_CHECKING:
    from pandas.core.generic import NDFrame
    from ..classes import MultiMolecule
    from ..armc import ARMC
    from h5py import File
else:
    from ..type_alias import File, NDFrame, MultiMolecule, ARMC

__all__ = ['create_hdf5', 'create_xyz_hdf5', 'to_hdf5', 'from_hdf5']

"""################################### Creating .hdf5 files ####################################"""


def create_hdf5(filename: PathType, armc: ARMC) -> None:
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
    kwarg_dict2 = _get_kwarg_dict2(armc)

    # Create a hdf5 file with *n* datasets
    with h5py.File(filename, 'w-', libver='latest') as f:
        for key, kwargs in sorted(kwarg_dict.items(), key=lambda tup: tup[0]):
            compression = 'gzip' if kwargs['shape'] else None
            f.create_dataset(name=key, compression=compression, **kwargs)

        group = f.create_group('validation')
        for key, kwargs in sorted(kwarg_dict2.items(), key=lambda tup: tup[0]):
            compression = 'gzip' if kwargs['shape'] else None
            group.create_dataset(name=key, compression=compression, **kwargs)

        f.attrs['super-iteration'] = -1
        f.attrs['sub-iteration'] = -1
        f.attrs['__version__'] = np.fromiter(__version__.split('.'), count=3, dtype=int)

        str_dtype = h5py.string_dtype(encoding='ascii')
        index: pd.MultiIndex = armc.param.param.index
        index_dtype = np.dtype(list((k, str_dtype) for k in index.names))

        dset = f.create_dataset(name='param_metadata', data=armc.param.to_struct_array())
        dset.attrs['index'] = index.values.astype(index_dtype)
        dset.attrs['net_charge'] = armc.param._net_charge
        dset.attrs['move_range'] = armc.param.move_range

        constraints = armc.param.constraints_to_str()
        index2: pd.MultiIndex = constraints.index
        index2_dtype = np.dtype(list((k, str_dtype) for k in index2.names))
        dset.attrs['constraints'] = constraints.values.astype(str_dtype)
        dset.attrs['constraints_index'] = index2.values.astype(index2_dtype)

    # Store the *index*, *column* and *name* attributes of dataframes/series in the hdf5 file
    kappa = armc.iter_len // armc.sub_iter_len
    idx = armc.param.param[0].index.append(pd.MultiIndex.from_tuples([('', 'phi', '')]))
    aux_error_idx = sorted(armc.pes.keys())

    pd_dict = {
        'aux_error': pd.Series(np.nan, index=aux_error_idx, name='aux_error'),
        'aux_error_mod': pd.Series(np.nan, index=idx, name='aux_error_mod'),
        'param': armc.param.param[0],
        'phi': pd.Series(np.nan, index=np.arange(kappa), name='phi'),
    }
    pd_dict2 = {
        'aux_error': pd.Series(np.nan, index=sorted(armc.pes_validation.keys()), name='aux_error'),
    }

    lst = [(armc.pes, pd_dict), (armc.pes_validation, pd_dict2)]
    for attr, dct in lst:
        for key, partial in attr.items():
            ref: pd.DataFrame = partial.ref  # type: ignore[attr-defined]
            dct[key] = ref
            dct[key + '.ref'] = ref

    with h5py.File(filename, 'r+', libver='latest') as f:
        index_to_hdf5(f, pd_dict)
        index_to_hdf5(f['validation'], pd_dict2)


def create_xyz_hdf5(filename: PathType,
                    mol_list: Iterable[MultiMolecule],
                    iter_len: int,
                    phi: Iterable[float]) -> None:
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

        iterator = (mol for mol in mol_list for _ in phi)
        for i, mol in enumerate(iterator):
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


def index_to_hdf5(f: h5py.Group, pd_dict: Dict[str, NDFrame]) -> None:
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
    param_shape = armc.param.param.T.shape

    ret = Settings()  # type: ignore[var-annotated]
    ret.phi.shape = (shape[0], ) + armc.phi.shape
    ret.phi.dtype = np.float64
    ret.phi.fillvalue = np.nan

    ret.param.shape = shape + param_shape
    ret.param.dtype = np.float64
    ret.param.fillvalue = np.nan

    ret.acceptance.shape = shape + armc.phi.shape
    ret.acceptance.dtype = np.bool_
    ret.acceptance.fillvalue = False

    if len(armc.phi) == 1:
        ret.aux_error.shape = shape + (len(armc.molecule), len(armc.pes) // len(armc.molecule))
    else:
        ret.aux_error.shape = shape + (len(armc.pes) // len(armc.phi), len(armc.phi))
    ret.aux_error.dtype = np.float64
    ret.aux_error.fillvalue = np.nan
    ret.aux_error_mod.shape = shape + (param_shape[0], 1 + param_shape[1])
    ret.aux_error_mod.dtype = np.float64
    ret.aux_error_mod.fillvalue = np.nan

    for key, partial in armc.pes.items():
        ref: pd.DataFrame = partial.ref  # type: ignore[attr-defined]

        ret[key].shape = shape + get_shape(ref)
        ret[key].dtype = np.float64
        ret[key].fillvalue = np.nan

        ret[f'{key}.ref'].shape = get_shape(ref)
        ret[f'{key}.ref'].dtype = np.float64
        ret[f'{key}.ref'].data = ref
        ret[f'{key}.ref'].fillvalue = np.nan

    return ret


def _get_kwarg_dict2(armc: ARMC) -> Settings:
    ret = Settings()  # type: ignore[var-annotated]
    shape = armc.iter_len // armc.sub_iter_len, armc.sub_iter_len

    err_shape = shape + (len(armc.molecule), len(armc.pes_validation) // len(armc.molecule))
    ret.aux_error.shape = err_shape
    ret.aux_error.dtype = np.float64
    ret.aux_error.fillvalue = np.nan

    for key, partial in armc.pes_validation.items():
        ref: pd.DataFrame = partial.ref  # type: ignore[attr-defined]

        ret[key].shape = shape + get_shape(ref)
        ret[key].dtype = np.float64
        ret[key].fillvalue = np.nan

        ret[f'{key}.ref'].shape = get_shape(ref)
        ret[f'{key}.ref'].dtype = np.float64
        ret[f'{key}.ref'].data = ref
        ret[f'{key}.ref'].fillvalue = np.nan
    return ret


"""################################### Updating .hdf5 files ####################################"""


def hdf5_clear_status(filename: PathType) -> Optional[subprocess.CompletedProcess[str]]:
    """Run the :code:`h5clear -s filename` command if **filename** refuses to open.

    Raises
    ------
    RuntimeError
        Raised if **filename** can neither be opened nor reset.

    """
    try:
        with h5py.File(filename, 'r+', libver='latest'):
            return None
    except OSError as ex:
        if not isfile(filename):
            raise

        name = fsdecode(filename)
        try:
            return subprocess.run(f'h5clear -s {name!r}', shell=True, check=True)
        except subprocess.CalledProcessError:
            raise RuntimeError(f'Unable to open or reset {name!r}') from ex


def hdf5_availability(filename: PathType, timeout: float = 5.0,
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
    msg = f"{filename!r} is currently unavailable; repeating attempt in {timeout:.0f} seconds"
    i = max_attempts if max_attempts is not None else np.inf
    if i <= 0:
        raise ValueError(f"'max_attempts' must be larger than 0; observed value: {max_attempts!r}")

    while i:
        try:
            with h5py.File(filename, 'r+', libver='latest'):
                return  # the .hdf5 file can safely be opened
        except OSError as ex:  # the .hdf5 file cannot be safely opened yet
            warning = ResourceWarning(msg)
            warning.__cause__ = ex
            warnings.warn(warning)

            error = ex
            sleep(timeout)
        i -= 1
    raise error


def to_hdf5(filename: PathType, dset_dict: Mapping[str, np.ndarray],
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
        for key, value in dset_dict.items():
            try:
                if key == 'xyz':
                    continue

                dset = f[key]
                if key == 'phi':
                    dset[kappa] = np.asarray(value, dtype=dset.dtype)
                else:
                    dset[kappa, omega] = np.asarray(value, dtype=dset.dtype)
            except Exception as ex:
                raise RuntimeError(f'Failed to write dataset {key!r}') from ex
        f.attrs['super-iteration'] = kappa
        f.attrs['sub-iteration'] = omega

    # Update the second hdf5 file with Cartesian coordinates
    filename_xyz = _get_filename_xyz(filename)
    _xyz_to_hdf5(filename_xyz, omega, dset_dict['xyz'])


def _xyz_to_hdf5(filename: PathType, omega: int,
                 mol_list: Optional[Iterable[MultiMolecule]]) -> None:
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
    if mol_list is None:
        return

    # Check if the hdf5 file is already opened. If opened: wait for 5 sec and try again.
    hdf5_availability(filename)

    with h5py.File(filename, 'r+', libver='latest') as f:
        iterator = ((f[f'xyz.{i}'], mol) for i, mol in enumerate(mol_list))
        for dset, mol in iterator:
            i = len(mol)
            if i <= dset.shape[1]:
                dset[omega, :i] = mol
            else:  # Resize and try again
                dset.resize(i, axis=1)
                dset[omega] = mol
        return


"""#################################### Reading .hdf5 files ####################################"""


@overload
def from_hdf5(filename: PathType, datasets: str) -> Union[pd.DataFrame, pd.Series]:
    ...
@overload  # noqa: E302
def from_hdf5(
    filename: PathType, datasets: Union[None, Iterable[str]] = ...
) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    ...
def from_hdf5(filename, datasets=None):  # noqa: E302
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

        # Identify the to-be returned datasets
        as_dict = True
        try:
            if isinstance(datasets, str):
                as_dict = False
                datasets_: Iterable[str] = (f[datasets].name,)
            elif datasets is None:
                datasets_ = recursive_keys(f)
            else:
                datasets_ = [f[k].name for k in datasets]  # type: ignore
        except KeyError as ex:
            raise KeyError(f"No dataset {ex} in {filename!r}") from None

        # Retrieve the datasets
        try:
            ret = {key.strip('/'): _get_dset(f, key) for key in datasets_}
        except AttributeError:
            raise ValueError('Illegal "Group" key; only "Dataset" keys are accepted') from None

    # Return a DataFrame/Series or dictionary of DataFrames/Series
    if not as_dict:
        for df in ret.values():
            return df
    return ret


def _get_dset(f: File, key: str) -> Union[pd.Series, pd.DataFrame, List[pd.DataFrame]]:
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
    if key == '/phi':
        return _phi_to_df(f, key)
    elif key == '/aux_error':
        return _aux_err_to_df(f, key)
    elif key == '/param_metadata':
        return _metadata_to_df(f, key)
    elif key == '/acceptance':
        return _acceptance_to_df(f, key)

    elif 'columns' in f[key].attrs.keys():
        return dset_to_df(f, key)

    elif 'name' in f[key].attrs.keys():
        return dset_to_series(f, key)

    elif f[key].ndim == 2:
        return pd.Series(_read_chunked(f, key), name=key)

    elif f[key].ndim == 3:
        data = _read_chunked(f, key)
        data.shape = np.product(data.shape[:-1]), -1
        columns = pd.MultiIndex.from_product([[key], np.arange(data.shape[-1])])
        return pd.DataFrame(data, columns=columns)

    raise ValueError(key, f[key].ndim)


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
    ret: np.ndarray = np.empty_like(f[key])
    ret[:j] = f[key][i:]
    ret[j:] = f[key][:i]
    return ret, idx_dict


def _metadata_to_df(f: File, key: str) -> pd.DataFrame:
    """Convert the ``param_metadata`` dataset into a :class:`~pandas.DataFrame`."""
    dset = f[key]
    index_ar = dset.attrs['index']
    index = pd.MultiIndex.from_tuples(
        zip(*(index_ar[k].astype(str) for k in index_ar.dtype.names)),
        names=index_ar.dtype.names,
    )

    columns = pd.MultiIndex(
        levels=[pd.Index([], dtype=np.int64), pd.Index([], dtype=np.object_)],
        codes=[[], []],
    )

    df = pd.DataFrame(index=index, columns=columns)
    for i, ar in enumerate(dset[:]):
        for k in dset.dtype.names:
            df[i, k] = ar[k] if k != "unit" else ar[k].astype(str)
    return df


def _phi_to_df(f: File, key: str) -> pd.DataFrame:
    """Convert the ``phi`` dataset into a :class:`~pandas.DataFrame`."""
    i = f.attrs['super-iteration'] + 1
    dset = f[key]
    index = pd.Index(dset.attrs['index'][:i], name='kappa')
    df = pd.DataFrame(dset[:i], index=index)
    df.columns.name = dset.attrs['name'].item().decode()
    return df


def _read_chunked(f: File, key: str) -> np.ndarray:
    """Read all data up to and including the current sub-iteration."""
    kappa: int = f.attrs['super-iteration']
    omega: int = f.attrs['sub-iteration'] + 1
    omega_max: int = f['param'].shape[1]
    i = kappa * omega_max

    dset = f[key]
    tail = dset.shape[2:]
    shape = (kappa * omega_max + omega,) + tail
    data = np.empty(shape, dtype=dset.dtype)
    if data.size == 0:
        return data

    if kappa != 0:
        data[:i] = dset[:kappa].reshape(-1, *tail)
    data[i:] = dset[kappa, :omega].reshape(-1, *tail)
    return data


def _acceptance_to_df(f: File, key: str) -> pd.DataFrame:
    """Convert the ``acceptance`` dataset into a :class:`~pandas.DataFrame`."""
    data = _read_chunked(f, 'acceptance')
    df = pd.DataFrame(data)
    df.index.name = 'iteration'
    df.columns.name = 'acceptance'
    return df


"""###################################### hdf5 utilities #######################################"""


def _get_filename_xyz(filename: PathType, **kwargs: Any) -> str:
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
    elif isinstance(filename, PathLike):  # type: ignore[misc]
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
    ret = np.array(index.tolist())
    if 'str' in ret.dtype.name or ret.dtype == object:  # h5py does not support unicode strings
        return ret.astype('S', copy=False)  # Convert to byte strings
    return ret


def dset_to_series(f: File, key: str) -> Union[pd.Series, pd.DataFrame]:
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
    index = array_to_index(f[key].attrs['index'])
    if not key.endswith(".ref"):
        data = _read_chunked(f, key)
    else:
        data = f[key][:]
    data.shape = np.product(data.shape[:-1], dtype=int), -1

    # Return a Series or DataFrame
    if data.ndim == 1:
        return pd.Series(data, index=index, name=name)
    else:
        columns = index
        index = pd.Index(np.arange(data.shape[0]), name=name)
        df = pd.DataFrame(data, index=index, columns=columns)
        if key == 'aux_error_mod':
            df.set_index(list(df.columns[0:-1]), inplace=True)
            df.columns = pd.Index(['phi'])
            return df['phi']
        return df


def dset_to_df(f: File, key: str) -> Union[pd.DataFrame, List[pd.DataFrame]]:
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
    if not key.endswith(".ref"):
        data = _read_chunked(f, key)
    else:
        data = f[key][:]

    # Return a DataFrame or list of DataFrames
    if data.ndim == 2:
        return pd.DataFrame(data, index=index, columns=columns)
    else:
        return [pd.DataFrame(i, index=index, columns=columns) for i in data]


def _aux_err_to_df(f: File, key: str) -> Union[pd.DataFrame, List[pd.DataFrame]]:
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
    data = _read_chunked(f, key)
    data.shape = np.product(data.shape[:-2], dtype=int), -1

    ret = pd.DataFrame(data, columns=columns)
    ret.index.name = f[key].attrs['name'][0].decode()
    return ret


Tuple3 = Tuple[np.ndarray, Dict[str, List[int]], np.ndarray]


def mol_from_hdf5(filename: PathType, i: int = -1, j: int = 0) -> Tuple3:
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
