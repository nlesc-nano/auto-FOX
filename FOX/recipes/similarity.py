"""Recipes for computing the similarity between trajectories.

Examples
--------
An example where, starting from two .xyz files, the similarity
is computed between two molecular dynamics (MD) trajectories.

.. code-block:: python

    >>> import numpy as np
    >>> import FOX
    >>> from FOX.recipes import compare_trajectories

    # The relevant multi-xyz files
    >>> md_filename: str = ...
    >>> md = FOX.MultiMolecule.from_xyz(md_filename)
    >>> md_ref_filename: str = ...
    >>> md_ref = FOX.MultiMolecule.from_xyz(md_ref_filename)

    # Calculate the similarity between `md` and `md_ref`
    >>> similarity = compare_trajectories(md, md_ref, metric="cosine")

    # Identify all sufficiently dissimilar molecules (as defined via `threshold`)
    >>> threshold: float = ...
    >>> idx = np.zeros(len(md), dtype=np.bool_)
    >>> idx[similarity >= threshold] = True

The resulting indices can be used for, for example, identifying all molecules
one wants to use for further (quantum-mechanical/classical) calculations.

.. code-block:: python

    >>> import qmflows

    # Define the job settings
    >>> s = qmflows.Settings()
    >>> s.lattice = [50, 50, 50]
    >>> s.specific.cp2k.motion.print["forces on"].filename = ""
    >>> s.overlay(qmflows.templates.singlepoint)

    # Construct the job list
    >>> mol_list = md[idx].as_Molecule()
    >>> job_list = [qmflows.cp2k(s, mol) for mol in mol_list]

    # Run the jobs
    >>> result_list = [qmflows.run(job) for job in job_list]

    # Extract the forces and energies from all jobs
    >>> forces = np.array([r.forces for r in result_list])[:, 0]
    >>> energy = np.array([r.energy for r in result_list])[:, 0]


Index
-----
.. currentmodule:: FOX.recipes
.. autosummary::
    compare_trajectories
    cosine
    euclidean

API
---
.. autofunction:: compare_trajectories
.. autofunction:: cosine
.. autofunction:: euclidean

"""

from __future__ import annotations

import sys
from functools import partial
from typing import Any, Callable, Union, Optional, TYPE_CHECKING

from FOX import MultiMolecule
import numpy as np

if sys.version_info >= (3, 8):
    from typing import Protocol, Literal, TypedDict
else:
    from typing_extensions import Protocol, Literal, TypedDict

if TYPE_CHECKING:
    import numpy.typing as npt

    class CallBack(Protocol):
        def __call__(
            self, __md: MultiMolecule, __md_ref: MultiMolecule, **kwargs: Any,
        ) -> np.ndarray: ...

    class MetricDict(TypedDict):
        cosine: Callable[[np.ndarray, np.ndarray], np.ndarray]
        euclidean: Callable[[np.ndarray, np.ndarray], np.ndarray]

__all__ = ["compare_trajectories", "cosine", "euclidean"]


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r"""Compute the cosine distance between all atom-pairs in **a** and **b**.

    .. math::

        1 - \frac{\mathbf{a}_{i,j,:} \cdot \mathbf{b}_{i,j,:}}
                 {||\mathbf{a}_{i,j,:}||_2 ||\mathbf{b}_{i,j,:}||_2}

    Parameters
    ----------
    a/b : :class:`np.ndarray <numpy.ndarray>`, shape :math:`(n_{mol}, n_{atom}, 3)`
        The two to-be compared trajectories.

    Returns
    -------
    :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(n_{mol}, n_{atom})`
        The distances between all pairs in **a** and **b**.

    See Also
    --------
    :func:`scipy.spatial.distance.cosine`
        Compute the Cosine distance between 1-D arrays.

    """
    numerator = (a * b).sum(axis=-1)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return 1 - numerator / denominator


def euclidean(a: np.ndarray, b: np.ndarray, p: Any = None) -> np.ndarray:
    r"""Compute the euclidean distance between all atom-pairs in **a** and **b**.

    .. math::

        ||\mathbf{a}_{i,j,:} - \mathbf{b}_{i,j,:}||_{p}

    Parameters
    ----------
    a/b : :class:`np.ndarray <numpy.ndarray>`, shape :math:`(n_{mol}, n_{atom}, 3)`
        The two to-be compared trajectories.
    p : :class:`int`
        The order of the norm.
        See the **ord** argument in :func:`numpy.linalg.norm` for more details.

    Returns
    -------
    :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(n_{mol}, n_{atom})`
        The distances between all pairs in **a** and **b**.

    See Also
    --------
    :func:`scipy.spatial.distance.euclidean`
        Computes the Euclidean distance between two 1-D arrays.
    :func:`numpy.linalg.norm`
        Matrix or vector norm.

    """
    return np.linalg.norm(a - b, axis=-1, ord=p)


MetricAliases = Literal["cosine", "euclidean"]

METRIC_DICT: MetricDict = {
    "cosine": cosine,
    "euclidean": euclidean,
}


def compare_trajectories(
    md: npt.ArrayLike,
    md_ref: npt.ArrayLike,
    metric: Union[MetricAliases, CallBack] = "cosine",
    reduce: Optional[Callable[[np.ndarray], np.ndarray]] = partial(np.mean, axis=-1),
    reset_origin: bool = True,
    **kwargs: Any,
) -> np.ndarray:
    r"""Compute the similarity between 2 trajectories according to the specified **metric**.

    Two default presets, ``"cosine"`` and ``"euclidean"``, are available as metrics,
    aforementioned metrics respectivelly being based on the cosine and euclidean distances
    between atoms of the passed trajectories.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from FOX.recipes import compare_trajectories

        >>> md: np.ndarray = ...
        >>> md_ref: np.ndarray = ...

        # Default `metric` presets
        >>> metric1 = compare_trajectories(md, md_ref, metric="cosine")
        >>> metric2 = compare_trajectories(md, md_ref, metric="euclidean")
        >>> metric3 = compare_trajectories(md, md_ref, metric="euclidean", p=1)

        >>> def rmsd(a: np.ndarray, axis: int) -> np.ndarray:
        ...     '''Calculate the root-mean-square deviation.'''
        ...     return np.mean(a**2, axis=axis)**0.5

        # Sum over the number of atoms rather than average
        >>> metric4 = compare_trajectories(md, md_ref, reduce=lambda n: np.sum(n, axis=-1))
        >>> metric5 = compare_trajectories(md, md_ref, reduce=lambda n: rmsd(n, axis=-1))

        >>> def sqeuclidean(md: np.ndarray, md_ref: np.ndarray) -> np.ndarray:
        ...     '''Calculate the distance based on the squared eclidian norm.'''
        ...     return np.linalg.norm(md - md_ref, axis=-1)**2

        # Pass a custom metric-function
        >>> metric6 = compare_trajectories(md, md_ref, metric=sqeuclidean)

    Parameters
    ----------
    md : :term:`numpy:array_like`, shape :math:`(n_{mol}, n_{atom}, 3)` or :math:`(n_{atom}, 3)`
        An array-like object containing the trajectory of interest.
    md_ref : :term:`numpy:array_like`, shape :math:`(n_{mol}, n_{atom}, 3)` or :math:`(n_{atom}, 3)`
        An array-like object containing the reference trajectory.
    metric : :class:`str` or :class:`Callable[[np.ndarray, np.ndarray], np.ndarray] <collections.abc.Callable>`
        The type of metric used for calculating the (dis-)similarity.
        Accepts either a callback or predefined alias: ``"cosine"`` or ``"euclidean"``.
        If a callback is provided then it should take two arrays, of shape
        :math:`(n_{mol}, n_{atom}, 3)`, as arguments and return a new array of
        shape :math:`(n_{mol}, n_{atom})`.
    reduce : :class:`Callable[[np.ndarray], np.ndarray] <collections.abc.Callable>`, optional
        A callable for performing a dimensional reduction.
        Used for transforming the shape :math:`(n_{mol}, n_{atom})` array, returned
        by **metric**, into the final shape :math:`(n_{mol},)` array.
        Setting this value to :data:`None` will disable the reduction and return the
        **metric** output in unaltered form.
    reset_origin : :class:`bool`
        Reset the origin by removing translations and rotations
        from the passed trajectories.
    \**kwargs : :data:`~typing.Any`
        Further keyword arguments for **metric**.

    Returns
    -------
    :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(n_{mol},)`
        An array with the (dis-)similarity between all molecules in **md** and **md_ref**.

    See Also
    --------
    :func:`FOX.recipes.cosine`
        Compute the cosine distance between all atom-pairs in **a** and **b**.
    :func:`FOX.recipes.euclidean`
        Compute the euclidean distance between all atom-pairs in **a** and **b**.

    """  # noqa: E501
    # Parse `md` and ensure that it is a 3D array
    md_ar = np.array(md, dtype=np.float64, ndmin=3, copy=False, subok=True)
    if not isinstance(md_ar, MultiMolecule):
        md_ar = md_ar.view(MultiMolecule)

    # Parse `md_ref` and ensure that it is a 3D array
    md_ref_ar = np.array(md_ref, dtype=np.float64, ndmin=3, copy=False, subok=True)
    if not isinstance(md_ref_ar, MultiMolecule):
        md_ref_ar = md_ref_ar.view(MultiMolecule)

    # Remove translations and rotations
    if reset_origin:
        md_ref_ar = md_ref_ar.reset_origin(inplace=False)
        md_ar = md_ar.reset_origin(inplace=False, rot_ref=md_ref_ar[0])

    # Parse the metric
    if callable(metric):
        func = metric
    else:
        try:
            func = METRIC_DICT[metric]  # type: ignore[assignment]
        except (TypeError, KeyError):
            if not isinstance(metric, str):
                raise TypeError("`metric` expected a string; observed type: "
                                f"{metric.__class__.__name__}") from None
            else:
                raise ValueError(f"Invalid `metric` value: {metric!r}") from None

    # Compute the (dis-)similarity
    ret = func(md_ar, md_ref_ar, **kwargs)
    return reduce(ret) if reduce is not None else ret
