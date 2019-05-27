"""Functions for storing Monte Carlo results in hdf5 format."""

from time import sleep
from typing import (Dict, Iterable, Optional, Union, Hashable, List, Tuple, Any)

import numpy as np
import pandas as pd
from pandas.core.generic import NDFrame

from scm.plams import Settings, Molecule

try:
    import h5py
    H5pyFile = h5py.File
    H5PY_ERROR = ''
except ImportError:
    __all__: list = []
    H5pyFile = 'h5py.File'
    H5PY_ERROR = "Use of the FOX.{} function requires the 'h5py' package.\
                  \n\t'h5py' can be installed via anaconda with the following command:\
                  \n\tconda install --name FOX -y -c conda-forge h5py"

from ..functions.utils import (get_shape, assert_error, array_to_index)

__all__ = ['create_hdf5', 'to_hdf5', 'from_hdf5']


@assert_error(H5PY_ERROR)
def create_hdf5(filename: str,
                armc: 'FOX.ARMC') -> None:
    r"""Create a hdf5 file to hold all addaptive rate Mone Carlo results (:class:`FOX.ARMC`).

    Datasets are created to hold a number of results following results over the course of the
    MC optimization:

    * The value of :math:`/phi` over the course of the parameter optimization (dataset: ``"phi"``)
    * The parameters (dataset: ``"param"``)
    * The acceptance rate (dataset: ``"acceptance"``)
    * The unmodified auxiliary error (dataset: ``"aux_error"``)
    * The modifications to the auxiliary error (dataset: ``"aux_error_mod"``)
    * User-specified PES descriptors (dataset(s): user-specified name(s))
    * User-specified reference PES descriptors (dataset(s): user-specified name(s) + ``"_ref"``)

    Cartesian coordinates (``"xyz"``) collected over the course of the current super-iteration are
    stored in a seperate file: ``filename + ".xyz"``.

    Parameters
    ----------
    filename : str
        The path+filename of the hdf5 file.

    armc : |FOX.ARMC|_
        A :class:`.ARMC` instance.

    """
    shape = armc.armc.iter_len // armc.armc.sub_iter_len, armc.armc.sub_iter_len

    # Create a Settings object with the shape and dtype of all datasets
    shape_dict = Settings()
    shape_dict.phi.shape = (shape[0], )
    shape_dict.phi.dtype = float
    shape_dict.param.shape = shape + (len(armc.param), )
    shape_dict.param.dtype = float
    shape_dict.acceptance.shape = shape
    shape_dict.acceptance.dtype = bool
    shape_dict.aux_error.shape = shape + (len(armc.pes), )
    shape_dict.aux_error.dtype = float
    shape_dict.aux_error_mod.shape = shape + (1 + len(armc.param), )
    shape_dict.aux_error_mod.dtype = float
    for key, value in armc.pes.items():
        shape_dict[key].shape = shape + get_shape(value.ref)
        shape_dict[key].dtype = float

    # Create a hdf5 file with *n* datasets
    _create_xyz_hdf5(filename, armc.job.molecule, shape)
    with h5py.File(filename, 'w-') as f:
        for key, kwarg in shape_dict.items():
            f.create_dataset(name=key, **kwarg)
            if key in armc.pes:  # Add the ab-initio reference PES descriptors to the hdf5 file
                f[key].attrs['ref'] = armc.pes[key].ref
        f.attrs['super-iteration'] = -1
        f.attrs['sub-iteration'] = -1

    # Store the *index*, *column* and *name* attributes of dataframes/series in the hdf5 file
    idx = armc.param['param'].index.append(pd.MultiIndex.from_tuples([('phi', '')]))
    pd_dict = {
        'param': armc.param['param'],
        'phi': pd.Series(np.nan, index=np.arange(shape[0]), name='phi'),
        'aux_error': pd.Series(np.nan, index=list(armc.pes), name='aux_error'),
        'aux_error_mod': pd.Series(np.nan, index=idx, name='aux_error_mod')
    }

    for key, value in armc.pes.items():
        pd_dict[key] = value.ref
    index_to_hdf5(filename, pd_dict)


@assert_error(H5PY_ERROR)
def _create_xyz_hdf5(filename: str,
                     mol: Molecule,
                     shape: Tuple[int]) -> None:
    """Create the ``"xyz"`` dataset for :func:`create_hdf5` in the hdf5 file ``filename+".xyz"``.

    The ``"xyz"`` dataset is to contain Cartesian coordinates collected over the course of the
    current ARMC super-iteration.

    Parameters
    ----------
    filename : str
        The path+filename of the hdf5 file.
        **filename** will be appended with ``".xyz"``.

    mol : |plams.Molecule|_
        A PLAMS Molecule.

    shape: tuple [int]
        A tuple containing the length of ARMC super- and sub-iterations.

    """
    xyz = Settings()
    xyz.shape = (shape[1], 1, len(mol), 3)
    xyz.dtype = float
    xyz.maxshape = (shape[1], None, len(mol), 3)
    xyz.fillvalue = np.nan

    filename_xyz = filename + '.xyz'
    with h5py.File(filename_xyz, 'w-') as f:
        f.create_dataset(name='xyz', compression='gzip', **xyz)
        f['xyz'].attrs['atoms'] = np.array([at.symbol for at in mol], dtype='S')


@assert_error(H5PY_ERROR)
def index_to_hdf5(filename: str,
                  pd_dict: Dict[str, NDFrame]) -> None:
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

    with h5py.File(filename, 'r+') as f:
        for key, value in pd_dict.items():
            for attr_name in attr_tup:
                if not hasattr(value, attr_name):
                    continue

                attr = getattr(value, attr_name)
                i = _attr_to_array(attr).T
                f[key].attrs.create(attr_name, i)


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
    if 'U' in ret.dtype.str:  # h5py does not support unicode strings
        return ret.astype('S', copy=False)  # Convert to byte strings
    return ret


@assert_error(H5PY_ERROR)
def hdf5_availability(filename: str,
                      timeout: float = 5.0,
                      max_attempts: Optional[int] = None) -> None:
    """Check if a .hdf5 file is opened by another process; return once it is not.

    If two processes attempt to simultaneously open a single hdf5 file then
    h5py will raise an ``OSError``.
    The purpose of this function is ensure that a .hdf5 is actually closed,
    thus allowing :func:`to_hdf5` to safely access **filename** without the risk of raising
    an ``OSError``.

    Parameters
    ----------
    filename : str
        The path+filename of the hdf5 file.

    timeout : float
        Time timeout, in seconds, between subsequent attempts of opening **filename**.

    max_attempts : int
        The maximum number attempts for opening **filename**.
        If the maximum number of attempts is exceeded, raise an ``OSError``.

    """
    i = max_attempts or np.inf
    while i:
        try:
            with h5py.File(filename, 'r+') as _:
                return
        except OSError as ex:
            print(("'{}' is temporary unavailable; "
                   "repeating attempt in {:f} seconds").format(filename, timeout))
            error = ex
            sleep(timeout)
        i -= 1
    raise error


@assert_error(H5PY_ERROR)
def to_hdf5(filename: str,
            dset_dict: Dict[str, np.array],
            kappa: int,
            omega: int) -> None:
    r"""Export results from **dict_** to the hdf5 file **name**.

    Parameters
    ----------
    filename : str
        The path+filename of the hdf5 file.

    dset_dict : dict [str, |np.ndarray|_]
        A dictionary with dataset names as keys and matching array-like objects as values.

    kappa : int
        The super-iteration, :math:`\kappa`, in the outer loop of :meth:`.ARMC.init_armc`.

    omega : int
        The sub-iteration, :math:`\omega`, in the inner loop of :meth:`.ARMC.init_armc`.

    """
    # Check if the hdf5 file is already opened. If opened: wait for 5 sec and try again.
    hdf5_availability(filename)

    # Update the hdf5 file
    with h5py.File(filename, 'r+') as f:
        f.attrs['super-iteration'] = kappa
        f.attrs['sub-iteration'] = omega
        for key, value in dset_dict.items():
            if key == 'xyz':
                pass
            elif key == 'phi':
                f[key][kappa] = value
            else:
                f[key][kappa, omega] = value

    # Update the second hdf5 file with Cartesian coordinates
    with h5py.File(filename+'.xyz', 'r+') as f:
        value = dset_dict['xyz']
        try:
            f['xyz'][omega] = value
        except TypeError:  # Reshape and try again
            f['xyz'].shape = (f['xyz'].shape[0],) + value.shape
            f['xyz'][omega] = value


DataSets = Optional[Union[Hashable, Iterable[Hashable]]]


@assert_error(H5PY_ERROR)
def from_hdf5(filename: str,
              datasets: DataSets = None) -> Union[NDFrame, Dict[Hashable, NDFrame]]:
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
    with h5py.File(filename, 'r') as f:
        # Retrieve all values up to and including the current iteration
        kappa = f.attrs['super-iteration']
        omega = f.attrs['sub-iteration']
        omega_max = f['param'].shape[1]
        i = kappa * omega_max + omega

        # Identify the to-be returned datasets
        if isinstance(datasets, str):
            datasets = (datasets, )
        elif datasets is None:
            datasets = (i for i in f.keys() if i != 'xyz')

        # Retrieve the datasets
        try:
            ret = {key: _get_dset(f, key)[:i+1] for key in datasets}
        except KeyError as ex:
            err = "No dataset '{}' in '{}'. The following datasets are available: {}"
            arg = str(ex).split("'")[1], str(filename), list(f.keys())
            raise KeyError(err.format(*arg))

    # Return a DataFrame/Series or dictionary of DataFrames/Series
    if len(ret) == 1:
        for i in ret.values():
            return i
    return ret


@assert_error(H5PY_ERROR)
def _get_dset(f: H5pyFile,
              key: Hashable) -> Union[pd.Series, pd.DataFrame, List[pd.DataFrame]]:
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
    if key == 'xyz':
        return _get_xyz_dset(f)

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

    raise TypeError(key, f[key].ndim)


@assert_error(H5PY_ERROR)
def _get_xyz_dset(f: H5pyFile) -> Tuple[np.ndarray, Dict[str, List[int]]]:
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
    idx_dict = {}
    for i, at in enumerate(f[key].attrs['atoms']):
        try:
            idx_dict[at].append(i)
        except KeyError:
            idx_dict[at] = [i]

    # Extract the Cartesian coordinates; sort in chronological order
    i = f.attrs['sub-iteration']
    j = f[key].shape[0] - i
    ret = np.empty_like(f[key])
    ret[:j] = f[key][i:]
    ret[j:] = f[key][:i]
    return ret, idx_dict


@assert_error(H5PY_ERROR)
def dset_to_series(f: H5pyFile,
                   key: Hashable) -> Union[pd.Series, pd.DataFrame]:
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
def dset_to_df(f: H5pyFile,
               key: Hashable) -> Union[pd.DataFrame, List[pd.DataFrame]]:
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


def restart_from_hdf5(filename: str) -> Dict[str, Any]:
    """Restart a previously started Addaptive Rate Monte Carlo procedure.

    Parameters
    ----------
    filename : str
        The path+name of an existing ARMC hdf5 file.
    """
    pass
