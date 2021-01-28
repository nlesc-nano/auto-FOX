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
    fps_reduce

API
---
.. autofunction:: compare_trajectories
.. autofunction:: fps_reduce

"""

from typing import Optional
from itertools import islice
from functools import partial

import numpy as np
from FOX import MultiMolecule
from scipy.spatial.distance import cdist

try:
    from CAT.distribution import uniform_idx
    CAT_EX: Optional[ImportError] = None
except ImportError as ex:
    CAT_EX = ex

__all__ = ["compare_trajectories", "fps_reduce"]


def _parse_md(md, name, dtype=np.float64):
    md_ar = np.array(md, dtype=dtype, ndmin=3, copy=False, subok=True)
    if md_ar.ndim != 3:
        raise ValueError(f"`{name}` expected a <= 3D array; observed dimensionality: {md_ar.ndim}")
    elif not isinstance(md_ar, MultiMolecule):
        return md_ar.view(MultiMolecule)
    else:
        return md_ar


def compare_trajectories(md, md_ref, *, metric='cosine', reduce=np.mean,
                         reset_origin=True, **kwargs):
    r"""Compute the similarity between 2 trajectories according to the specified **metric**.

    The default **metric** aliases :func:`scipy.spatial.distance.cdist` for defining
    the (dis-)similarity between the passed **md** and its reference.
    This (dis-)similarity array is subsequently reduced to a vector of size
    :math:`(N_{mol},)` by taking its mean (along the relevant axes).

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
        >>> metric3 = compare_trajectories(md, md_ref, metric="minkowski", p=1)

        >>> def rmsd(a: np.ndarray) -> np.float64:
        ...     '''Calculate the root-mean-square deviation.'''
        ...     return np.mean(a**2)**0.5

        # Sum over the number of atoms rather than average
        >>> metric4 = compare_trajectories(md, md_ref, reduce=np.sum)
        >>> metric5 = compare_trajectories(md, md_ref, reduce=rmsd)

        >>> def sqeuclidean(md: np.ndarray, md_ref: np.ndarray) -> np.ndarray:
        ...     '''Calculate the distance based on the squared eclidian norm.'''
        ...     delta = md[..., None] - md_ref[..., None, :]
        ...     return np.linalg.norm(delta, axis=-1)**2

        # Pass a custom metric-function
        >>> metric6 = compare_trajectories(md, md_ref, metric=sqeuclidean)

    Parameters
    ----------
    md : :term:`numpy:array_like`, shape :math:`(N_{mol}, N_{atom1}, 3)` or :math:`(N_{atom1}, 3)`
        An array-like object containing the trajectory of interest.
    md_ref : :term:`numpy:array_like`, shape :math:`(N_{mol}, N_{atom2}, 3)` or :math:`(N_{atom2}, 3)`
        An array-like object containing the reference trajectory.
    metric : :class:`str` or :class:`Callable[[FOX.MultiMolecule, FOX.MultiMolecule], np.ndarray] <collections.abc.Callable>`
        The type of metric used for calculating the (dis-)similarity.
        Accepts either a callback or predefined alias. See **metric** parameter
        in :func:`scipy.spatial.distance.cdist` for a comprehensive overview of all aliases.
        If a callback is provided then it should take a array of shape :math:`(n_{atom1}, 3)`
        and :math:`(N_{atom2}, 3)` as arguments and return a new array of
        shape :math:`(N_{atom1}, N_{atom2})`.
    reduce : :class:`Callable[[np.ndarray], np.number] <collections.abc.Callable>`, optional
        A callable for performing a dimensional reduction.
        Used for transforming the shape :math:`(N_{atom1}, N_{atom2})` array,
        returned by **metric**, into a scalar.
        Setting this value to :data:`None` will disable the reduction and return the
        **metric** output in unaltered form.
    reset_origin : :class:`bool`
        Reset the origin by removing translations and rotations
        from the passed trajectories.
    \**kwargs : :data:`~typing.Any`
        Further keyword arguments for **metric**.

    Returns
    -------
    :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(N_{mol},)`
        An array with the (dis-)similarity between all molecules in **md** and **md_ref**.

    See Also
    --------
    :func:`scipy.spatial.distance.cdist`
        Compute distance between each pair of the two collections of inputs.

    """  # noqa: E501
    # Parse the inputs; ensure that they are a 3D array
    md_ar = _parse_md(md, "md")
    md_ref_ar = _parse_md(md_ref, "md_ref")
    if len(md_ar) != len(md_ref_ar):
        raise ValueError("`md` and `md_ref` should be of the same length")

    # Remove translations and rotations
    if reset_origin:
        md_ref_ar = md_ref_ar.reset_origin(inplace=False)
        md_ar = md_ar.reset_origin(inplace=False, rot_ref=md_ref_ar[0])

    # Parse the metric
    if callable(metric):
        func = metric
    elif isinstance(metric, str):
        func = partial(cdist, metric=metric)
    else:
        raise TypeError("`metric` expected a string or a callable; observed type: "
                        f"{metric.__class__.__name__}")

    # Compute the (dis-)similarity
    if reduce is None:
        return np.array([func(a, b, **kwargs) for a, b in zip(md_ar, md_ref_ar)])
    else:
        return np.array([reduce(func(a, b, **kwargs)) for a, b in zip(md_ar, md_ref_ar)])


def fps_reduce(dist_mat, n=1, **kwargs):
    r"""Return the indices that yield a uniform distribution of **n** points.

    Examples
    --------
    .. code-block:: python

        >>> from functools import partial
        >>> import numpy as np
        >>> from FOX.recipes import compare_trajectories, fps_reduce

        >>> md: np.ndarray = ...
        >>> md_ref: np.ndarray = ...

        >>> reduce_func = partial(fps_reduce, n=10)
        >>> out = compare_trajectories(md, md_ref, reduce=reduce_func)

    Note
    ----
    This function requires the Compound Attachment Tools package:
    `CAT <https://github.com/nlesc-nano/CAT>`_.

    Parameters
    ----------
    dist_mat : :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(m_a, m_b)`
        A distance matrix.
    n : :class:`int`, optional
        The number of to-be returned indices.
    \**kwargs : :data:`~typing.Any`
        Further keyword arguments for :func:`CAT.distribution.uniform_idx`.

    Returns
    -------
    :class:`np.ndarray[np.int64] <numpy.ndarray>`, shape :math:`(n,)`
        An array of indices.

    See Also
    --------
    :func:`CAT.distribution.uniform_idx`
        Yield the column-indices that result in a uniform or clustered distribution.
    :func:`FOX.recipes.compare_trajectories`
        Compute the similarity between 2 trajectories according to the specified **metric**.

    """
    if CAT_EX is not None:
        raise CAT_EX
    elif np.ndim(dist_mat) != 2:
        raise ValueError("`dist_mat` expected a 2D array")

    count = -1 if n is None else n
    iterator = islice(uniform_idx(dist_mat, **kwargs), None, n)
    return np.fromiter(iterator, dtype=np.intp, count=count)
