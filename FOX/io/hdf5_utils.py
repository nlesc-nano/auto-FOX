"""Functions for storing Monte Carlo results in hdf5 format."""

from typing import (Dict, Iterable, Optional, Union, Hashable, List, Tuple)

import numpy as np
import pandas as pd
from pandas.core.generic import NDFrame

from scm.plams import Settings

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
                mc_kwarg: Settings) -> None:
    r"""Create a hdf5 file to hold all addaptive rate Mone Carlo results (:class:`FOX.ARMC`).

    Datasets are created to hold a number of results following results over the course of the
    MC optimization:

    * The xyz coordinates
    * The value of :math:`/phi` over the course of the parameter optimization (dataset: *phi*).
    * The parameters (dataset: *param*)
    * The acceptance rate (dataset: *acceptance*)
    * The unmodified auxiliary error (dataset: *aux_error*)
    * The modifications to the auxiliary error (dataset: *aux_error_mod*)
    * User-specified PES descriptors (dataset(s): user-specified name(s))
    * User-specified reference PES descriptors (dataset(s): user-specified name(s) + *_ref*)

    :parameter str filename: The path+name of the hdf5 file.
    :parameter mc_kwarg: An ARMC object.
    :type mc_kwarg: |FOX.ARMC|_
    """
    shape = mc_kwarg.armc.iter_len // mc_kwarg.armc.sub_iter_len, mc_kwarg.armc.sub_iter_len

    # Create a Settings object with the shape and dtype of all datasets
    shape_dict = Settings()
    shape_dict.xyz.shape = (shape[1], 1, len(mc_kwarg.job.molecule), 3)
    shape_dict.xyz.dtype = float
    shape_dict.xyz.maxshape = (shape[1], None, len(mc_kwarg.job.molecule), 3)
    shape_dict.xyz.fillvalue = np.nan
    shape_dict.phi.shape = (shape[0], )
    shape_dict.phi.dtype = float
    shape_dict.param.shape = shape + (len(mc_kwarg.param), )
    shape_dict.param.dtype = float
    shape_dict.acceptance.shape = shape
    shape_dict.acceptance.dtype = bool
    shape_dict.aux_error.shape = shape + (len(mc_kwarg.pes), )
    shape_dict.aux_error.dtype = float
    shape_dict.aux_error_mod.shape = shape + (1 + len(mc_kwarg.param), )
    shape_dict.aux_error_mod.dtype = float
    for key, value in mc_kwarg.pes.items():
        shape_dict[key].shape = shape + get_shape(value.ref)
        shape_dict[key].dtype = float

    # Create a hdf5 file with *n* datasets
    with h5py.File(filename, 'w-') as f:
        for key, kwarg in shape_dict.items():
            f.create_dataset(name=key, compression='gzip', **kwarg)

            # Add the ab-initio reference PES descriptors to the hdf5 file
            if key in mc_kwarg.pes:
                f.create_dataset(
                    data=mc_kwarg.pes[key].ref,
                    name=key + '_ref',
                    compression='gzip',
                    dtype=kwarg.dtype,
                    shape=kwarg.shape[2:]
                )
        f.attrs['super-iteration'] = -1
        f.attrs['sub-iteration'] = -1
        f['xyz'].attrs['atoms'] = np.array([at.symbol for at in mc_kwarg.job.mol], dtype='S')

    # Store the *index*, *column* and *name* attributes of dataframes/series in the hdf5 file
    idx = mc_kwarg.param['param'].index.append(pd.MultiIndex.from_tuples([('phi', '')]))
    pd_dict = {
        'param': mc_kwarg.param['param'],
        'phi': pd.Series(np.nan, index=np.arange(shape[0]), name='phi'),
        'aux_error': pd.Series(np.nan, index=list(mc_kwarg.pes), name='aux_error'),
        'aux_error_mod': pd.Series(np.nan, index=idx, name='aux_error_mod')
    }

    for key, value in mc_kwarg.pes.items():
        pd_dict[key] = value.ref
    index_to_hdf5(filename, pd_dict)


@assert_error(H5PY_ERROR)
def index_to_hdf5(filename: str,
                  pd_dict: Dict[str, NDFrame]) -> None:
    """Export the *index* and *columns* / *name* attributes of a Pandas dataframe/series to a
    pre-existing hdf5 file.

    Attributes are exported for all dataframes/series in **pd_dict** and skipped otherwise.
    The keys in **pd_dict**, together with the attribute names, are used for naming the datasets:

    .. code-block:: python

        >>> pd_dict = {}
        >>> pd_dict['df'] = pd.DataFrame(np.random.rand(10, 10))
        >>> pd_dict['series'] = pd.Series(np.random.rand(10))
        >>> index_to_hdf5(pd_dict, name='my_file.hdf5')

        >>> with h5py.File('my_file.hdf5', 'r') as f:
        >>>     tuple(f.keys())
        ('df.columns', 'df.index', 'series.index', 'series.name')

    :parameter str filename: The path+name of the hdf5 file.
    :parameter pd_dict: A dictionary with dataset names as keys and matching array-like objects
        as values.
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

    .. code-block:: python

        >>> item = 'name'
        >>> _attr_to_array(item)
        array([b'name'], dtype='|S4')

        >>> item = pd.RangeIndex(stop=4)
        >>> _attr_to_array(item)
        array([0, 1, 2, 3, 4])

    :parameter item: A string or instance of pd.Index (or one of its subclasses).
    :type index: |str|_ or |pd.Index|_
    :return: An array created fron **item**.
    :rtype: |np.ndarray|_
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
def to_hdf5(filename: str,
            dset_dict: Dict[str, np.array],
            kappa: int,
            omega: int) -> None:
    r"""Export results from **dict_** to the hdf5 file **name**.

    :parameter str filename: The path+name of the hdf5 file.
    :parameter dict dset_dict: A dictionary with dataset names as keys and matching array-like
        objects as values.
    :parameter int kappa: The super-iteration, :math:`\kappa`, in the outer loop of
        :meth:`.ARMC.init_armc`.
    :parameter int omega: The sub-iteration, :math:`\omega`, in the inner loop of
        :meth:`.ARMC.init_armc`.
    n"""
    with h5py.File(filename, 'r+') as f:
        f.attrs['super-iteration'] = kappa
        f.attrs['sub-iteration'] = omega
        for key, value in dset_dict.items():
            if key == 'xyz':
                try:
                    f[key][omega] = value
                except TypeError:  # Reshape and try again
                    f[key].shape = (f[key].shape[0],) + value.shape
                    f[key][omega] = value
            elif key == 'phi':
                f[key][kappa] = value
            else:
                f[key][kappa, omega] = value


DataSets = Optional[Union[Hashable, Iterable[Hashable]]]


@assert_error(H5PY_ERROR)
def from_hdf5(filename: str,
              datasets: DataSets = None) -> Union[NDFrame, Dict[Hashable, NDFrame]]:
    """Retrieve all user-specified datasets from **name**, returning a dicionary of
    DataFrames and/or Series.

    :parameter str filename: The path+name of the hdf5 file.
    :parameter list datasets: A list of to be retrieved dataset names.
        All datasets will be retrieved if *None*.
    :return: A dicionary with dataset names as keys and the matching data as values.
    :rtype: |dict|_ (values:|pd.DataFrame|_ and/or |pd.Series|_)
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
        else:
            datasets = datasets or f.keys()

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

    :parameter f: An opened hdf5 file.
    :type f: |h5py.File|_
    :parameter str key: The dataset name.
    :return: A NumPy array or a Pandas DataFrame or Series retrieved from **key** in **f**.
    :rtype: |pd.DataFrame|_ or |pd.Series|_
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
    """ Return the *xyz* dataset from **f** as a :class:`MultiMolecule` instance.

    :parameter f: An opened hdf5 file.
    :type f: |h5py.File|_
    :return: A list of :math:`k` MultiMolecule instances with :math:`m` molecules and
        :math:`n` atoms.
    :rtype: :math:`k*m*n*3` |np.ndarray|_ and |dict|_
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

    :parameter f: An opened hdf5 file.
    :type f: |h5py.File|_
    :parameter str key: The dataset name.
    :return: A Pandas Series or DataFrame retrieved from **key** in **f**.
    :rtype: |pd.Series|_ or |pd.DataFrame|_
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
               key: Hashable) -> Union[pd.DataFrame, List[pd.DataFrame], 'xr.DataArray']:
    """Take a h5py dataset and convert it into a Pandas DataFrame (if 2D) or list of Pandas
    DataFrames (if 3D).

    :parameter f: An opened hdf5 file.
    :type f: |h5py.File|_
    :parameter str key: The dataset name.
    :return: A Pandas DataFrame retrieved from **key** in **f**.
    :rtype: |pd.DataFrame|_ or |list|_ [|pd.DataFrame|_]
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
