""" Functions for storing Monte Carlo results in hdf5 format. """

__all__ = ['create_hdf5', 'to_hdf5', 'from_hdf5']

import os
from os.path import join

import numpy as np
import pandas as pd

from scm.plams import Settings

try:
    import h5py
    H5PY_ERROR = False
except ImportError:
    __all__ = []
    H5PY_ERROR = "Use of the FOX.{} function requires the 'h5py' package.\
                  \n\t'h5py' can be installed via anaconda with the following command:\
                  \n\tconda install --name FOX -y -c conda-forge h5py"

from ..functions.utils import (get_shape, assert_error, array_to_index)


@assert_error(H5PY_ERROR)
def create_hdf5(mc_kwarg, path=None, name='MC.hdf5'):
    """ Create a hdf5 file to hold all addaptive rate Mone Carlo results (:class:`FOX.ARMC`).
    Datasets are created to hold a number of results following results over the course of the
    MC optimization:

    * The acceptance rate (dataset: *acceptance*)
    * The parameters (dataset: *param*)
    * User-specified PES descriptors (dataset(s): user-specified name(s))
    * The *index*, *columns* and/or *name* attributes above-mentioned results

    :parameter mc_kwarg: An ARMC object.
    :type mc_kwarg: |FOX.ARMC|_
    :parameter str path: The path where the the hdf5 file is stored.
    :parameter str name: The name (including extension) of the hdf5 file.
    """
    path = path or os.getcwd()
    filename = join(path, name)
    shape = mc_kwarg.armc.iter_len // mc_kwarg.armc.sub_iter_len, mc_kwarg.armc.sub_iter_len

    # Create a Settings object with the shape and dtype of all datasets
    shape_dict = Settings()
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
        f.attrs['iteration'] = -1
        f.attrs['subiteration'] = -1
        for key, value in shape_dict.items():
            f.create_dataset(name=key, compression='gzip', **value)

    # Store the *index*, *column* and *name* attributes of dataframes/series in the hdf5 file
    idx = mc_kwarg.param['param'].index.append(pd.MultiIndex.from_tuples([('phi', '')]))
    pd_dict = {'param': mc_kwarg.param['param'],
               'phi': pd.Series(np.nan, name='phi'),
               'aux_error': pd.Series(np.nan, index=list(mc_kwarg.pes), name='aux_error'),
               'aux_error_mod': pd.Series(np.nan, index=idx, name='aux_error_mod')}
    for key, value in mc_kwarg.pes.items():
        pd_dict[key] = value.ref
    index_to_hdf5(pd_dict, path)


@assert_error(H5PY_ERROR)
def index_to_hdf5(pd_dict, path=None, name='MC.hdf5'):
    """ Export the *index* and *columns* / *name* attributes of a Pandas dataframe/series to a
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

    :parameter pd_dict: A dictionary with dataset names as keys and matching array-like objects
        as values.
    :parameter str path: The path where the the hdf5 file is stored.
    :parameter str name: The name (including extension) of the hdf5 file.
    """
    path = path or os.getcwd()
    filename = join(path, name)
    attr_tup = ('index', 'columns', 'name')

    with h5py.File(filename, 'r+') as f:
        for key, value in pd_dict.items():
            for attr_name in attr_tup:
                if hasattr(value, attr_name):
                    attr = getattr(value, attr_name)
                    i = _attr_to_array(attr).T
                    f[key].attrs.create(attr_name, i)


def _attr_to_array(item):
    """ Convert an attribute value, retrieved from :func:`FOX.index_to_hdf5`, into a NumPy array.

    .. code-block:: python

        >>> item = 'name'
        >>> _attr_to_array(item)
        array([b'name'], dtype='|S4')

        >>> item = pd.Index(np.arange(5))
        >>> _attr_to_array(item)
        array([0, 1, 2, 3, 4, 5])

    :parameter object item: An object that may or may not belong to the pd.Index class.
    :return: An array created fron **item**.
    :rtype: |np.ndarray|_
    """
    # If **idx** does not belong to the pd.Index class or one of its subclass
    if not isinstance(item, pd.Index):  # **item** belongs to the *name* attribute of pd.Series
        return np.array(item, dtype='S', ndmin=1)

    # Convert **item** into an array
    ret = np.array(item.to_list())
    if 'U' in ret.dtype.str:  # h5py does not support unicode strings
        return ret.astype('S', copy=False)  # Convert to byte strings
    return ret


@assert_error(H5PY_ERROR)
def to_hdf5(dict_, i, j, phi, path=None, name='MC.hdf5'):
    """ Export results from **dict_** to the hdf5 file **name**.

    :parameter dict dict_: A dictionary with dataset names as keys and matching array-like objects
        as values.
    :parameter int i: The iteration in the outer loop of :meth:`ARMC.init_armc`.
    :parameter int j: The subiteration in the inner loop of :meth:`ARMC.init_armc`.
    :parameter str path: The path where the the hdf5 file is stored.
    :parameter str name: The name (including extension) of the hdf5 file.
    """
    path = path or os.getcwd()
    filename = join(path, name)

    with h5py.File(filename, 'r+') as f:
        f.attrs['iteration'] = i
        f.attrs['subiteration'] = j
        f['phi'][i] = phi
        for key, value in dict_.items():
            f[key][i, j] = value


@assert_error(H5PY_ERROR)
def from_hdf5(datasets=None, path=None, name='MC.hdf5'):
    """ Retrieve all user-specified datasets from **name**, returning a dicionary of
    DataFrames and/or Series.

    :parameter list datasets: A list of to be retrieved dataset names.
        All datasets will be retrieved if *None*.
    :parameter str path: The path where the the hdf5 file is stored.
    :parameter str name: The name (including extension) of the hdf5 file.
    :return: A dicionary with dataset names as keys and the matching data as values.
    :rtype: |dict|_ (values:|pd.DataFrame|_ and/or |pd.Series|_)
    """
    path = path or os.getcwd()
    filename = join(path, name)
    ret = {}

    with h5py.File(filename, 'r') as f:
        datasets = datasets or f.keys()
        if isinstance(datasets, str):
            datasets = (datasets, )
        for key in datasets:
            ret[key] = _get_dset(f, key)

    if len(ret) == 1:
        for i in ret.values():
            return i
    return ret


@assert_error(H5PY_ERROR)
def _get_dset(f, key):
    """ Take a h5py dataset and convert it into either a NumPy array or
    a Pandas DataFrame (:func:`FOX.dset_to_df`) or Series (:func:`FOX.dset_to_series`).

    :parameter f: An opened hdf5 file.
    :type f: |h5py.File|_
    :parameter str key: The dataset name.
    :return: A NumPy array or a Pandas DataFrame or Series retrieved from **key** in **f**.
    :rtype: |np.ndarray|_, |pd.DataFrame|_ or |pd.Series|_
    """
    if 'columns' in f[key].attrs.keys():
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
def dset_to_series(f, key):
    """ Take a h5py dataset and convert it into a Pandas Series (if 1D) or Pandas Series (if 2D).

    :parameter f: An opened hdf5 file.
    :type f: |h5py.File|_
    :parameter str key: The dataset name.
    :return: A Pandas Series retrieved from **key** in **f**.
    :rtype: |pd.Series|_ or |list|_ [|pd.Series|_]
    """
    name = f[key].attrs['name'][0].decode()
    index = array_to_index(f[key].attrs['index'][:])
    data = f[key][:]
    data.shape = np.product(data.shape[:-1], dtype=int), -1
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
def dset_to_df(f, key):
    """ Take a h5py dataset and convert it into a Pandas DataFrame (if 2D) or list of Pandas
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
    if data.ndim == 2:
        return pd.DataFrame(data, index=index, columns=columns)
    return [pd.DataFrame(i, index=index, columns=columns) for i in data]
