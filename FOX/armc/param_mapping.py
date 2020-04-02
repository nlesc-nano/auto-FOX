import textwrap
from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import (Any, TypeVar, Optional, Hashable, Tuple, Mapping, Iterable, ClassVar, Union,
                    Iterator, KeysView, ItemsView, ValuesView, overload, Callable, FrozenSet, Dict)
from logging import Logger
from functools import wraps, partial

import numpy as np
import pandas as pd
from assertionlib.dataclass import AbstractDataClass

from ..type_hints import Literal, TypedDict, NDArray
from ..functions.charge_utils import update_charge, get_net_charge, ChargeError

__all__ = ['ParamMapping']

# A generic typevar
T = TypeVar('T')

# MultiIndex keys
_KT1 = TypeVar('_KT1', bound=Hashable)
_KT2 = TypeVar('_KT2', bound=Hashable)
_KT3 = TypeVar('_KT2', bound=Hashable)

# All dict keys in ParamMappingABC
ValidKeys = Literal['param', 'param_old', 'min', 'max', 'constraints', 'count']

# A function for moving parameters
MoveFunc = Callable[[float, float], float]

# A dictionary of constraints
ConstrainDict = Mapping[_KT2, partial]

# MultiIndex keys as a 2-tuple
Tup3 = Tuple[_KT1, _KT2, _KT3]


class _InputMapping(TypedDict):
    # Variables
    param: pd.Series
    param_old: pd.Series


class InputMapping(_InputMapping, total=False):
    """A :class:`~typing.TypedDict` representing the :class:`ParamMappingABC` **df** parameter."""  # noqa

    # Constants
    min: pd.Series
    max: pd.Series
    constraints: pd.Series
    count: pd.Series


class ParamMappingABC(AbstractDataClass, ABC, Mapping[ValidKeys, pd.Series]):
    r"""A :class:`~collections.abc.Mapping` for storing and updating forcefield parameters.

    Besides the implementation of the :class:`~collections.abc.Mapping` protocol,
    this class has access to four main methods:

    * :meth:`__call__` or :meth:`move` move a random parameter by a random step size.
    * :meth:`identify_move` identify the parameter and move step size.
    * :meth:`clip_move` clip the move.
    * :meth:`apply_constraints` apply further constraints to the move.

    Note that  :meth:`__call__` /  :meth:`move` will internally call all other three methods.

    Examples
    --------
    .. code:: python
        >>> import pandas as pd

        >>> df = pd.DataFrame(..., index=pd.MultiIndex(...))
        >>> param = ParamMapping(df, ...)

        >>> idx = param.move()


    Attributes
    ----------
    move_range : :class:`numpy.ndarray` [:class:`float`], shape :math:`(n,)`
        An 1D array with all allowed move steps.

    func : :class:`~Collections.abc.Callable`
        The callable used for applying :math:`\phi` to the auxiliary error.
        The callable should take an two floats as arguments and return a new float.

    _data : :class:`dict` [:class:`str`, :class:`pandas.Series`], private
        A dictionary of Series containing the forcefield parameters.
        The index should be a 3-level :class:`~pandas.MultiIndex`,
        the first level containg the key-alias, the second the parameter name and
        the third the atom (pair).

        Has access to the the following keys:

        * ``"param"`` (:class:`float`): The current forcefield paramters.
        * ``"param_old"`` (:class:`float`): Old paramters from a previous iteration.
        * ``"min"`` (:class:`float`): The minimum value for each parameter.
        * ``"max"`` (:class:`float`): The maximum value for each parameter.
        * ``"constraints"`` (:class:`object`): A dictionary of constraints for each parameter.
          Setting a value to ``None`` means that there are no constraints for that parameter.
        * ``"count"`` (:class:`int`): The number of unique atom (pairs) assigned
          to a given parameter.
          Currently only important when updating the charge,
          as the latter requires a charge renormalization after every move to ensure
          the total molecular charge remains constant.

    _net_charge : :class:`float`, optional
        The net charge of the molecular system.
        Only applicable if the ``"charge"`` is among the passed parameters.

    """

    _net_charge: Optional[float]
    _move_range: np.ndarray
    __data: Mapping[ValidKeys, pd.Series]
    # func: Callable[[float, float], float]

    #: Fill values for when optional keys are absent.
    FILL_VALUE: ClassVar[Mapping[ValidKeys, Any]] = MappingProxyType({
        'param_old': np.nan,
        'min': -np.inf,
        'max': np.inf,
        'constraints': None,  # Dict[str, functools.partial]
        'count': -1
    })

    def __init__(self, data: Union[InputMapping, pd.DataFrame],
                 move_range: Iterable[float],
                 func: MoveFunc, **kwargs: Any) -> None:
        r"""Initialize an :class:`ParamMappingABC` instance.

        Parameters
        ----------
        _data : :class:`pandas.DataFrame` or :class:`~collections.abc.Mapping` [:class:`str`, :class:`pandas.Series`]
            A DataFrame (or dict of Series) containing the ``"param"`` key
            with the forcefield parameters.
            The index should be a 2-level :class:`~pandas.MultiIndex`,
            the first level containg the parameter name and the second the atom (pair).
            Optionally it can also contain one or more of the following keys:

            * ``"param_old"`` (:class:`float`): Old paramters from a previous iteration.
            * ``"min"`` (:class:`float`): The minimum value for each parameter.
            * ``"max"`` (:class:`float`): The maximum value for each parameter.
            * ``"constraints"`` (:class:`object`): A dictionary of constraints for each parameter.
              Setting a value to ``None`` means that there are no constraints for that parameter.
            * ``"count"`` (:class:`int`): The number of unique atom (pairs) assigned
              to a given parameter.
              Currently only important when updating the charge,
              as the latter requires a charge renormalization after every move to ensure
              the total molecular charge remains constant.

            See :attr:`ParamMappingABC._data`.

        move_range : :class:`~collections.abc.Iterable` [:class:`float`]
            An iterable with all allowed move steps.
            See :attr:`ParamMappingABC.move_range`.

        func : :class:`~Collections.abc.Callable`
            The callable used for applying :math:`\phi` to the auxiliary error.
            The callable should take an two floats as arguments and return a new float.
            See :attr:`ParamMappingABC.func`.

        \**kwargs : :data:`~typing.Any`
            Further keyword arguments for **func**.

        """  # noqa
        super().__init__()
        self.move_range = move_range
        self.func = wraps(func)(partial(func, **kwargs))
        self._data = data

    # Properties

    @property
    def move_range(self) -> NDArray[float]:
        return self._move_range

    @move_range.setter
    def move_range(self, value: Iterable[float]) -> None:
        try:
            count = len(value)  # type: ignore
        except TypeError:
            count = -1
        self._move_range = np.fromiter(value, count=count, dtype=float)

    @property
    def _data(self) -> Dict[ValidKeys, pd.Series]:
        return self.__data

    @_data.setter
    def _data(self, value: Union[InputMapping, pd.DataFrame]) -> None:
        dct = dict(value)

        # Check that the 'param' key is present
        try:
            param = dct['param']
        except KeyError as ex:
            raise KeyError("The {'param'!r} key is absent from the passed mapping") from ex

        # Check that it has a 3-level MultiIndex
        n_level = 3
        if not isinstance(param.index, pd.MultiIndex):
            raise TypeError(f"Series.index expected a {n_level}-level MultiIndex; "
                            f"observed type: {param.index.__class__.__name!r}")
        elif len(param.index.levels) != n_level:
            raise ValueError(f"Series.index expected a {n_level}-level MultiIndex; "
                             f"observed number levels: {len(param.index.levels)}")

        # Fill in the defaults
        for name, fill_value in self.FILL_VALUE.items():
            if name in dct:
                continue
            dtype = type(fill_value) if fill_value is not None else object
            dct[name] = pd.Series(fill_value, index=param.index, name=name, dtype=dtype)

        if 'charge' in dct['param']:
            self._net_charge = get_net_charge(dct['param']['charge'], dct['count']['charge'])
        else:
            self._net_charge = None
        self.__data = dct

    # Magic methods and Mapping implementation

    def __eq__(self, value: Any) -> bool:
        return object.__eq__(self, value)

    def __repr__(self) -> str:
        indent = 4 * ' '
        data = repr(pd.DataFrame(self._data))
        return f'{self.__class__.__name__}(\n{textwrap.indent(data, indent)}\n)'

    def __getitem__(self, key: ValidKeys) -> pd.Series:
        """Implement :code:`self[key]`."""
        return self._data[key]

    def __iter__(self) -> Iterator[ValidKeys]:
        """Implement :code:`iter(self)`."""
        return iter(self._data)

    def __len__(self) -> int:
        """Implement :code:`len(self)`."""
        return len(self._data)

    def __contains__(self, key: Any) -> bool:
        """Implement :code:`key in self`."""
        return key in self._data

    def keys(self) -> KeysView[ValidKeys]:
        """Return a set-like object providing a view of this instance's keys."""
        return self._data.keys()

    def items(self) -> ItemsView[ValidKeys, pd.Series]:
        """Return a set-like object providing a view of this instance's key/value pairs."""
        return self._data.items()

    def values(self) -> ValuesView[pd.Series]:
        """Return an object providing a view of this instance's values."""
        return self._data.values()

    @overload  # type: ignore
    def get(self, key: ValidKeys, default: T) -> pd.Series: ...

    @overload
    def get(self, key: Hashable, default: T) -> T: ...

    def get(self, key, default=None):
        """Return the value for **key** if it's available; return **default** otherwise."""
        return self._data.get(key, default)

    # The actual meat of the class

    def __call__(self, logger: Optional[Logger] = None) -> Union[ChargeError, Tup3]:
        """Update a random parameter in **self.param** by a random value from **self.move.range**.

        Performs in inplace update of the ``'param'`` column in **self.param**.
        By default the move is applied in a multiplicative manner.
        **self.job.md_settings** and **self.job.preopt_settings** are updated to reflect the
        change in parameters.

        Parameters
        ----------
        logger : :class:`logging.Logger`, optional
            A logger for reporting the updated value.

        Returns
        -------
        :class:`tuple` [:class:`~Collections.abc.Hashable`, :class:`~Collections.abc.Hashable`]
            The index of the updated parameter.

        """
        # Prepare arguments a move
        idx, x1, x2 = self.identify_move()
        _value = self.func(x1, x2)
        value = self.clip_move(idx, _value)

        # Create a call to the logger
        if logger is not None:
            _, prm_type, atoms = idx
            logger.info(f"Moving {prm_type} ({atoms}): {x1:.4f} -> {value:.4f}")

        constraints = self['constraints'][idx]
        ex = self.apply_constraints(idx, value, constraints)
        if ex is not None:
            return ex

        self['param'][idx] = value
        return idx

    @abstractmethod
    def identify_move(self) -> Tuple[Tup3, float, float]:
        """Identify the to-be moved parameter and the size of the move.

        Returns
        -------
        :class:`tuple` [:class:`~Collections.abc.Hashable`, :class:`~Collections.abc.Hashable`], :class:`float` and :class:`float`
            The index of the to-be moved parameter, it's value and the size of the move.

        """  # noqa
        raise NotImplementedError('Trying to call an abstract method')

    def clip_move(self, idx: Tup3, value: float) -> float:
        """An optional function for clipping the value of **value**.

        Parameters
        ----------
        idx : :class:`tuple` [:class:`~Collections.abc.Hashable`, :class:`~Collections.abc.Hashable`]
            The index of the moved parameter.

        value : :class:`float`
            The value of the moved parameter.

        Returns
        -------
        :class:`float`
            The newly clipped value of the moved parameter.

        """  # noqa
        return value

    def apply_constraints(self, idx: Tup3, value: float, constraints: ConstrainDict) -> None:
        """An optional function for applying further constraints based on **idx** and **value**.

        Should perform an inplace update of this instance.

        Parameters
        ----------
        idx : :class:`tuple` [:class:`~Collections.abc.Hashable`, :class:`~Collections.abc.Hashable`]
            The index of the moved parameter.

        value : :class:`float`
            The value of the moved parameter.

        idx : :class:`~collections.abc.Mapping`
            A Mapping with the to-be applied constraints per atom (pair).

        """  # noqa
        pass


MOVE_RANGE = np.array([
    0.900, 0.905, 0.910, 0.915, 0.920, 0.925, 0.930, 0.935, 0.940,
    0.945, 0.950, 0.955, 0.960, 0.965, 0.970, 0.975, 0.980, 0.985,
    0.990, 0.995, 1.005, 1.010, 1.015, 1.020, 1.025, 1.030, 1.035,
    1.040, 1.045, 1.050, 1.055, 1.060, 1.065, 1.070, 1.075, 1.080,
    1.085, 1.090, 1.095, 1.100
], dtype=float)
MOVE_RANGE.setflags(write=False)


class ParamMapping(ParamMappingABC):

    #: A set of charge-like parameters which require a parameter re-normalization after every move.
    CHARGE_LIKE: ClassVar[FrozenSet[str]] = frozenset({
        'charge'
    })

    @ParamMappingABC.inherit_annotations()
    def __init__(self, data, move_range=MOVE_RANGE, func=np.multiply, **kwargs):
        super().__init__(data, move_range, func=func, **kwargs)

    def identify_move(self) -> Tuple[Tup3, float, float]:
        """Identify and return a random parameter and move size.

        Returns
        -------
        :class:`tuple` [:class:`~Collections.abc.Hashable`, :class:`~Collections.abc.Hashable`], :class:`float` and :class:`float`
            The index of the to-be moved parameter, it's value and the size of the move.

        """  # noqa
        # Define a random parameter
        random_prm = self['param'].sample()
        idx, x1 = next(random_prm.items())

        # Define a random move size
        x2 = np.random.choice(self.move_range, 1)[0]
        return idx, x1, x2

    def clip_move(self, idx: Tup3, value: float) -> float:
        """Ensure that **value** falls within a user-specified range.

        Parameters
        ----------
        idx : :class:`tuple` [:class:`~Collections.abc.Hashable`, :class:`~Collections.abc.Hashable`]
            The index of the moved parameter.

        value : :class:`float`
            The value of the moved parameter.

        Returns
        -------
        :class:`float`
            The newly clipped value of the moved parameter.

        """  # noqa
        prm_min = self['min'][idx]
        prm_max = self['max'][idx]
        return np.clip(value, prm_min, prm_max)

    def apply_constraints(self, idx: Tup3, value: float,
                          constraints: ConstrainDict) -> Optional[ChargeError]:
        """Apply further constraints based on **idx** and **value**.

        Performs an inplace update of this instance.

        Parameters
        ----------
        idx : :class:`tuple` [:class:`~Collections.abc.Hashable`, :class:`~Collections.abc.Hashable`]
            The index of the moved parameter.

        value : :class:`float`
            The value of the moved parameter.

        constraints : :class:`~collections.abc.Mapping`
            A Mapping with the to-be applied constraints per atom (pair).

        """  # noqa
        _, prm_type, _ = idx
        charge = self._net_charge if prm_type in self.CHARGE_LIKE else None

        constraints_ = None if pd.isnull(constraints) else constraints
        return update_charge(idx, value, self['param'],
                             count=self['count'],
                             prm_min=self['min'],
                             prm_max=self['max'],
                             constrain_dict=constraints_,
                             net_charge=charge)


ParamMapping.__doc__ = ParamMappingABC.__doc__
