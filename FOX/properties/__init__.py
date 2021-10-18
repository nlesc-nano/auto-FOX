r"""Functions for calculating/extracting various properties.

Each function can be used to calculate the respective property as is,
or to extract it from a passed :class:`qmflows.Result <qmflows.packages.Result>` instance.

.. code-block:: python

    >>> from FOX.properties import get_bulk_modulus
    >>> from qmflows.packages import Result
    >>> import numpy as np

    >>> # Calculate the bulk modulus from a set of arrays
    >>> pressure: np.ndarray = ...
    >>> volume: np.ndarray = ...
    >>> get_bulk_modulus(pressure, volume)  # doctest: +SKIP
    array([[[ 0.,  1.,  2.],
            [ 3.,  4.,  5.]],
    <BLANKLINE>
           [[ 6.,  7.,  8.],
            [ 9., 10., 11.]]])

    >>> # Calculate the bulk modulus from a qmflows.Result instance
    >>> result: Result = ...
    >>> get_bulk_modulus.from_result(result)  # doctest: +SKIP
    array([[[ 0.,  1.,  2.],
            [ 3.,  4.,  5.]],
    <BLANKLINE>
           [[ 6.,  7.,  8.],
            [ 9., 10., 11.]]])

An example for how :func:`get_bulk_modulus` can be used in conjunction
with the :ref:`ARMC yaml input <monte_carlo_parameters.pes>`.
Note that additional CP2K ``print`` keys are required in order for it
to export the necessary properties.

.. code-block:: yaml

    job:
        type: FOX.armc.PackageManager
        molecule: mol.xyz

        md:
            template: qmflows.md.specific.cp2k_mm
            settings:
                cell_parameters: [50, 50, 50]
                input:
                    motion:
                        print:
                            cell on:
                                filename: ''
                            forces on:
                                filename: ''
                        md:
                            ensemble: NVE
                            thermostat:
                                print:
                                    temperature on:
                                        filename: ''

    pes:
        bulk_modulus:
            func: FOX.properties.get_bulk_modulus.from_result
            ref: [1.0]
            kwargs:
                reduce: mean

Index
-----
.. currentmodule:: FOX.properties
.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - :func:`get_pressure <get_pressure>` (forces, coords, volume[...])
     - Calculate the pressure from the passed **forces**.
   * - :func:`get_bulk_modulus <get_bulk_modulus>` (pressure, volume[...])
     - Calculate the bulk modulus via differentiation of **pressure** w.r.t. **volume**.
   * - :func:`get_attr <get_attr>` (obj, name[, default, reduce, axis])
     - :func:`getattr` with support for additional keyword argument.
   * - :func:`call_method <call_method>` (obj, name, \*args[, reduce, axis])
     - Call the **name** method of **obj**.
   * - :class:`FromResult` (func[, result_func])
     - A class for wrapping :data:`~types.FunctionType` objects.

API
---
.. autofunction:: get_pressure
.. autofunction:: get_bulk_modulus
.. autofunction:: get_attr
.. autofunction:: call_method
.. autoclass:: FromResult(func, name, module=None, doc=None)

    .. data:: REDUCTION_NAMES

        A mapping that maps :meth:`from_result` aliases to callbacks.

        In addition to the examples below, all reducable ufuncs
        from :ref:`numpy <numpy:ufuncs>` and :mod:`scipy.special` are available.

        :type: :class:`types.MappingProxyType[str, Callable[[np.ndarray], np.float64]] <types.MappingProxyType>`

"""

# flake8: noqa: E402

from .base import FromResult, get_attr, call_method
from .pressure import get_pressure
from .bulk_modulus import get_bulk_modulus

__all__ = ['FromResult', 'get_attr', 'call_method', 'get_pressure', 'get_bulk_modulus']
