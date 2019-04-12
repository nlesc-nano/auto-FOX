""" A class for storing MC results in hdf5 format. """

__all__ = []

from os.path import join

import numpy as np

from scm.plams import Settings

try:
    import h5py
    H5PY_ERROR = False
except ImportError:
    __all__ = []
    H5PY_ERROR = "Use of the FOX.{} class requires the 'h5py' package.\
                  \n\t'h5py' can be installed via anaconda with the following command:\
                  \n\tconda install --name FOX -y -c conda-forge h5py"

from ..functions.utils import get_shape


class HDF5():
    """ A class for exporting to HDF5 files. """
    def __init__(self, mc_kwarg):
        self.path = mc_kwarg.job.path

        # Create a dictionary with the shape and dtype of all to-be stored data
        shape_dict = Settings()
        shape_dict.param.shape = mc_kwarg.armc.iter_len, len(mc_kwarg.param)
        shape_dict.param.dtype = np.float64
        shape_dict.acceptance.shape = (mc_kwarg.armc.iter_len, )
        shape_dict.acceptance.dtype = bool

        # Create *n* hdf5 files with a single dataset
        kwarg = {'chunks': True, 'compression': 'gzip'}
        for key in shape_dict:
            filename = join(self.path, key) + '.hdf5'
            with h5py.File(filename, 'a') as f:
                shape = shape_dict[key].shape
                dtype = shape_dict[key].dtype
                f.create_dataset(name=key, shape=shape,
                                 maxshape=shape, dtype=dtype, **kwarg)

        # Create a dictionary with the shape for all PES descriptors
        shape_dict = {}
        for i in mc_kwarg.pes:
            shape_dict[i] = (mc_kwarg.armc.iter_len, ) + get_shape(mc_kwarg.pes[i].ref)

        # Create a single hdf5 files with a *n* dataset
        filename = join(self.path, 'pes_descriptors') + '.hdf5'
        with h5py.File(filename, 'a') as f:
            for key in shape_dict:
                shape = shape_dict[key]
                f.create_dataset(name=key, shape=shape, maxshape=shape,
                                 dtype=np.float64, **kwarg)

    def to_acceptance(self, accept, i, j):
        """
        :parameter bool accept: Whether or not the parameters in the latest Monte Carlo
            iteration were accepted.
        :parameter int i: The iteration in the outer loop of :meth:`ARMC.init_armc`.
        :parameter int j: The subiteration in the inner loop of :meth:`ARMC.init_armc`.
        """
        k = j + i * j
        self._to_hdf5(accept, 'acceptance', k)

    def to_pes_descriptors(self, descriptor_dict, i, j):
        """
        :parameter descriptor_dict: The latest set of PES-descriptors.
        :type descriptor_dict: |dict|_ (keys: |str|_, values: |np.ndarray|_ [|np.float64|_])
        :parameter int i: The iteration in the outer loop of :meth:`ARMC.init_armc`.
        :parameter int j: The subiteration in the inner loop of :meth:`ARMC.init_armc`.
        """
        k = j + i * j
        self._to_hdf5(descriptor_dict, 'pes_descriptors', k)

    def to_param(self, param, i, j):
        """
        :parameter param: The latest set of ARMC-optimized parameters.
        :type param: |np.ndarray|_ [|np.float64|_]
        :parameter int i: The iteration in the outer loop of :meth:`ARMC.init_armc`.
        :parameter int j: The subiteration in the inner loop of :meth:`ARMC.init_armc`.
        """
        k = j + i * j
        self._to_hdf5(param, 'param', k)

    def _to_hdf5(self, item, name, k):
        """
        :parameter item: An array-like object or dictionary of array-like objects.
        :type item: |np.ndarray|_ or |dict|_ [|np.ndarray|_]
        :parameter str name: The filename (excluding path) of the hdf5 file.
        :parameter int k: The index in the hdf5 file where **item** should be placed.
        """
        filename = join(self.path, name)
        with h5py.File(filename, 'r+') as f:
            if isinstance(item, dict):
                for key in item:
                    f[key][k] = item[key]
            else:
                f[name][k] = item


# Raise an error when trying to call the ToHDF5 class without 'h5py' installed
if H5PY_ERROR:
    _doc = ToHDF5.__doc__

    class ToHDF5(ToHDF5):
        def __init__(self, mc_kwarg):
            name = str(self.__class__).rsplit("'", 1)[0].rsplit('.', 1)[1]
            raise ModuleNotFoundError(H5PY_ERROR.format(name))
    ToHDF5.__doc__ = _doc
