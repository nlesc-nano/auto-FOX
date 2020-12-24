"""Functions for calculating the bulk modulus."""

import warnings
from typing import TypeVar, Callable, Any

import numpy as np
from scm.plams import Units
from qmflows.packages.cp2k_package import CP2K_Result
from qmflows.warnings_qmflows import QMFlows_Warning

from . import FromResult, get_pressure

__all__ = ['get_bulk_modulus', 'GetBulkMod']

T = TypeVar("T")
T1 = TypeVar("T1")
FT = TypeVar("FT", bound=Callable[..., Any])


def get_bulk_modulus(
    pressure,
    volume,
    pressure_unit='ha/bohr^3',
    volume_unit='bohr',
    return_unit='ha/bohr^3',
):
    r"""Calculate the bulk modulus via differentiation of **pressure** w.r.t. **volume**.

    .. math::

        B = -V * \frac{\delta P}{\delta V}

    Parameters
    ----------
    pressure : :class:`np.ndarray[np.float64] <numpy.ndarray>`
        A 1D array of pressures used for defining :math:`\delta P`.
        Must be of equal length as **volume**.
    volume : :class:`np.ndarray[np.float64] <numpy.ndarray>`
        A 1D array of volumes used for defining :math:`\delta V`.
        Must be of equal length as **pressure**.
    pressure_unit : :class:`str`
        The unit of the **pressure**.
    volume_unit : :class:`str`
        The unit of the **volume**.
        The passed unit will automatically cubed, *e.g.* ``Angstrom -> Angstrom**3``.
    return_unit : :class:`str`
        The unit of the to-be returned pressure.

    Returns
    -------
    :class:`np.float64 <numpy.float64>` or :class:`np.ndarray[np.float64] <numpy.ndarray>`
        The bulk modulus :math:`B`.
        Returend as either a scalar or array, depending on the dimensionality **volume_ref**.

    """
    # Parse `pressure` and `volume`
    p = np.asarray(pressure, dtype=np.float64) * Units.conversion_ratio(pressure_unit, 'ha/bohr^3')
    v = np.asarray(volume, dtype=np.float64) * Units.conversion_ratio(volume_unit, 'bohr')**3

    ret = np.gradient(p, v)
    ret *= -v
    ret *= Units.conversion_ratio('ha/bohr^3', return_unit)
    return ret


class GetBulkMod(FromResult[FT, CP2K_Result]):
    """A :class:`FOX.properties.FromResult` subclass for :func:`get_bulk_modulus`."""

    @property
    def __call__(self):  # noqa: D205,D400
        """
        Note
        ----
        Using :meth:`get_bulk_modulus.from_result() <FromResult.from_result>` requires the passed
        :class:`qmflows.CP2K_Result <qmflows.packages.cp2k_package.CP2K_Result>`
        to have access to the following files for each argument:

        * **pressure**: ``cp2k-frc-1.xyz``, ``cp2k-pos-1.xyz``, ``cp2k-1.cell`` & ``cp2k-1.ener``
        * **volume**: ``cp2k-1.cell``

        Furthermore, in order to get sensible results both the pressure and
        cell volume must be variable.

        """
        return self._func

    def from_result(self, result, reduce=None, axis=None, *, return_unit='ha/bohr^3', **kwargs):
        r"""Call **self** using argument extracted from **result**.

        Parameters
        ----------
        result : :class:`qmflows.CP2K_Result <qmflows.packages.cp2k_package.CP2K_Result>`
            The Result instance that **self** should operator on.
        reduce : :class:`str` or :class:`Callable[[Any], Any] <collections.abc.Callable>`, optional
            A callback for reducing the output of **self**.
            Alternativelly, one can provide on of the string aliases from :attr:`REDUCTION_NAMES`.
        axis : :class:`int` or :class:`Sequence[int] <collections.abc.Sequence>`, optional
            The axis along which the reduction should take place.
            If :data:`None`, use all axes.
        return_unit : :class:`str`
            The unit of the to-be returned quantity.
        \**kwargs : :data:`~typing.Any`
            Further keyword arguments for :meth:`__call__`.

        Returns
        -------
        :data:`~typing.Any`
            The output of :meth:`__call__`.

        """  # noqa: E501
        # Attempt to pull from the cache
        if result.status in {'failed', 'crashed'}:
            raise RuntimeError(f"Cannot extract data from a job with status {result.status!r}")

        # Check the cache
        ret1 = self._cache_get(result, return_unit)
        if ret1 is not None:
            return self._reduce(ret1, reduce, axis)

        a_to_au = Units.conversion_ratio('angstrom', 'bohr')
        with warnings.catch_warnings():
            warnings.simplefilter('error', QMFlows_Warning)

            volume = self._pop(
                kwargs, 'volume', callback=lambda: getattr(result, 'volume') * a_to_au**3
            )
        pressure = get_pressure.from_result(result, reduce=None, volume=volume)

        ret2 = self(pressure, volume, return_unit=return_unit, **kwargs)
        self._cache[result] = (ret2, return_unit)
        return self._reduce(ret2, reduce, axis)
