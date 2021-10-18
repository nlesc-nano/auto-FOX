"""Functions for calculating the bulk modulus."""

from __future__ import annotations

from typing import Callable, Any, TYPE_CHECKING

import numpy as np
from scm.plams import Units
from qmflows.packages.cp2k_package import CP2K_Result
from qmflows.warnings_qmflows import QMFlows_Warning
from nanoutils import warning_filter

from . import FromResult

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike
    from numpy import float64 as f8

__all__ = ['get_bulk_modulus']


@FromResult
def get_bulk_modulus(
    pressure: ArrayLike,
    volume: ArrayLike,
    *,
    pressure_unit: str = 'ha/bohr^3',
    volume_unit: str = 'bohr',
    return_unit: str = 'ha/bohr^3',
) -> NDArray[f8]:
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
    :class:`np.float64 <numpy.double>` or :class:`np.ndarray[np.float64] <numpy.ndarray>`
        The bulk modulus :math:`B`.
        Returend as either a scalar or array, depending on the dimensionality **volume_ref**.


    .. automethod:: get_bulk_modulus.from_result

    """
    # Parse `pressure` and `volume`
    p = np.asarray(pressure, dtype=np.float64) * Units.conversion_ratio(pressure_unit, 'ha/bohr^3')
    v = np.asarray(volume, dtype=np.float64) * Units.conversion_ratio(volume_unit, 'bohr')**3

    ret = np.gradient(p, v)
    ret *= -v
    ret *= Units.conversion_ratio('ha/bohr^3', return_unit)
    return ret


@get_bulk_modulus._set_result_func
@warning_filter('error', category=QMFlows_Warning)
def _(
    self: FromResult[Callable[..., Any]],
    result: CP2K_Result,
    *,
    reduce: None | str | Callable[[Any], Any] = None,
    axis: None | int | tuple[int, ...] = None,
    return_unit: str = 'ha/bohr^3',
    **kwargs: Any,
) -> Any:
    r"""Call :func:`get_bulk_modulus` using argument extracted from **result**.

    Parameters
    ----------
    result : :class:`qmflows.CP2K_Result <qmflows.packages.cp2k_package.CP2K_Result>`
        The Result instance that **self** should operator on.
    reduce : :class:`str` or :class:`Callable[[Any], Any] <collections.abc.Callable>`, optional
        A callback for reducing the output of **self**.
        Alternativelly, one can provide on of the string aliases
        from :attr:`~FromResult.REDUCTION_NAMES`.
    axis : :class:`int` or :class:`Sequence[int] <collections.abc.Sequence>`, optional
        The axis along which the reduction should take place.
        If :data:`None`, use all axes.
    return_unit : :class:`str`
        The unit of the to-be returned quantity.
    \**kwargs : :data:`~typing.Any`
        Further keyword arguments for :func:`get_bulk_modulus`.

    Returns
    -------
    :data:`~typing.Any`
        The output of :func:`get_bulk_modulus`.

    """
    # Attempt to pull from the cache
    if result.status in {'failed', 'crashed'}:
        raise RuntimeError(f"Cannot extract data from a job with status {result.status!r}")

    a_to_au = Units.conversion_ratio('angstrom', 'bohr')
    bar_to_au = Units.conversion_ratio('bar', 'ha/bohr^3')
    volume = self._pop(
        kwargs, 'volume',
        callback=lambda: getattr(result, 'volume') * a_to_au**3,
    )
    pressure = self._pop(
        kwargs, 'pressure',
        callback=lambda: getattr(result, 'pressure') * bar_to_au,
    )

    ret = self(pressure, volume, return_unit=return_unit, **kwargs)
    return self._reduce(ret, reduce, axis)
