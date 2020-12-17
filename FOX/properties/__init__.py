"""Functions for calculating/extracting various properties.

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
                reduction: mean

Index
-----
.. currentmodule:: FOX.properties
.. autosummary::
    FromResult
    get_attr
    call_method
    get_pressure
    get_bulk_modulus

API
---
.. autoclass:: FromResult(func, name, module=None, doc=None)
    :members: __call__, from_result

    .. data:: REDUCTION_NAMES
        :annotation: : Mapping[str, Callable[[np.ndarray], np.float64]]
        :value: ...

        A mapping that maps :meth:`from_result` aliases to callbacks.

        In addition to the examples below, all reducable ufuncs
        from :ref:`numpy <numpy:ufuncs>` and :mod:`scipy.special` are available.

        .. code-block:: python

            >>> from types import MappingProxyType
            >>> import numpy as np
            >>> import scipy.special

            >>> REDUCTION_NAMES = MappingProxyType({
            ...     'min': np.min,
            ...     'max': np.max,
            ...     'mean': np.mean,
            ...     'sum': np.sum,
            ...     'product': np.product,
            ...     'var': np.var,
            ...     'std': np.std,
            ...     'ptp': np.ptp,
            ...     'norm': np.linalg.norm,
            ...     'argmin': np.argmin,
            ...     'argmax': np.argmax,
            ...     'all': np.all,
            ...     'any': np.any,
            ...     'add': np.add.reduce,
            ...     'eval_legendre': scipy.special.eval_legendre.reduce,
            ...     ...
            ... })

.. autofunction:: get_attr
.. autofunction:: call_method
.. autofunction:: get_pressure
.. autofunction:: get_bulk_modulus

"""

# flake8: noqa: E402

from .base import FromResult, get_attr, call_method

from .pressure import get_pressure as _get_pressure, GetPressure
get_pressure = GetPressure(
    _get_pressure,
    name='get_pressure',
    module='FOX.properties',
)
del _get_pressure, GetPressure

from .bulk_modulus import get_bulk_modulus as _get_bulk_modulus, GetBulkMod
get_bulk_modulus = GetBulkMod(
    _get_bulk_modulus,
    name='get_bulk_modulus',
    module='FOX.properties',
)
del _get_bulk_modulus, GetBulkMod

__all__ = ['FromResult', 'get_attr', 'call_method', 'get_pressure', 'get_bulk_modulus']
