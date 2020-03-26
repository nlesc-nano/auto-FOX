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
from ..functions.charge_utils import update_charge

__all__ = ['ParamMapping']

# A generic typevar
T = TypeVar('T')

# MultiIndex keys
_KT1 = TypeVar('_KT1', bound=Hashable, covariant=True)
_KT2 = TypeVar('_KT2', bound=Hashable, covariant=True)

# All dict keys in ParamMappingABC
ValidKeys = Literal['param', 'param_old', 'unit', 'key_path', 'min', 'max', 'constraints', 'count']

# A function for moving parameters
MoveFunc = Callable[[float, float], float]

# A dictionary of constraints
ConstrainDict = Mapping[_KT2, partial]

# MultiIndex keys as a 2-tuple
Tup2 = Tuple[_KT1, _KT2]


class InputMapping(TypedDict, total=False):
    """A :class:`~typing.TypedDict` representing the :class:`ParamMappingABC` **df** parameter."""  # noqa

    param: pd.Series
    param_old: pd.Series
    unit: pd.Series
    key_path: pd.Series
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
        The index should be a 2-level :class:`~pandas.MultiIndex`,
        the first level containg the parameter name and the second the atom (pair).

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

    """

    # move_range: np.ndarray
    # df: Mapping[ValidKeys, pd.Series]
    # apply_move: Callable[[float, float], float]

    #: Fill values for when optional keys are absent.
    FILL_VALUE: ClassVar[Mapping[ValidKeys, Any]] = MappingProxyType({
        'param_old': np.nan,
        'unit': '{}',
        'min': -np.inf,
        'max': np.inf,
        'constraints': None,  # Dict[str, functools.partial]
        'count': -1
    })

    def __init__(self, df: Union[InputMapping, pd.DataFrame],
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
        self._data = df

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
        if 'key_path' not in dct:
            raise KeyError("The {'key_path'!r} key is absent from the passed mapping")

        # Check that it has a 2-level MultiIndex
        if not isinstance(param.index, pd.MultiIndex):
            raise TypeError("Series.index expected a 2-level MultiIndex; "
                            f"observed type: {param.index.__class__.__name!r}")
        elif param.index.levels != 2:
            raise ValueError("Series.index expected a 2-level MultiIndex; "
                             f"observed number levels: {param.index.levels!r}")

        if 'unit' in dct:
            dct['unit'] = [f'[{u}] {{}}' for u in dct['unit']]

        # Fill in the defaults
        for name, fill_value in self.FILL_VALUE.items():
            if name in dct:
                continue
            dtype = type(fill_value) if fill_value is not None else object
            dct[name] = pd.Series(fill_value, index=param.index, name=name, dtype=dtype)

        self.__data = dct

    # Magic methods and Mapping implementation

    def __eq__(self, value: Any) -> bool:
        return object.__eq__(self, value)

    def __repr__(self) -> str:
        indent = 4 * ' '
        data = repr(self._data)
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

    def __call__(self, logger: Optional[Logger] = None) -> Tup2:
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
        self['param_old'][:] = self['param']

        # Prepare arguments a move
        idx, x1, x2 = self.identify_move()
        _value = self.func(x1, x2)
        value = self.clip_move(idx, _value)

        # Create a call to the logger
        if logger is not None:
            prm_type, atoms = idx
            logger.info(f"Moving {prm_type} ({atoms}): {x1:.4f} -> {value:.4f}")

        constraints = self['constraints'].at[idx]
        if constraints is not None:
            self.apply_constraints(idx, value, constraints)

        return idx

    move = __call__  # An alias

    @abstractmethod
    def identify_move(self) -> Tuple[Tup2, float, float]:
        """Identify the to-be moved parameter and the size of the move.

        Returns
        -------
        :class:`tuple` [:class:`~Collections.abc.Hashable`, :class:`~Collections.abc.Hashable`], :class:`float` and :class:`float`
            The index of the to-be moved parameter, it's value and the size of the move.

        """  # noqa
        raise NotImplementedError('Trying to call an abstract method')

    def clip_move(self, idx: Tup2, value: float) -> float:
        """An optional function for clipping the value of **value**.

        Parameters
        ----------
        :class:`tuple` [:class:`~Collections.abc.Hashable`, :class:`~Collections.abc.Hashable`]
            The index of the moved parameter.
        :class:`float`
            The value of the moved parameter.

        Returns
        -------
        :class:`float`
            The newly clipped value of the moved parameter.

        """
        return value

    def apply_constraints(self, idx: Tup2, value: float, constraints: ConstrainDict) -> None:
        """An optional function for applying further constraints based on **idx** and **value**.

        Should perform an inplace update of this instance.

        Parameters
        ----------
        :class:`tuple` [:class:`~Collections.abc.Hashable`, :class:`~Collections.abc.Hashable`]
            The index of the moved parameter.
        :class:`float`
            The value of the moved parameter.
        :class:`~collections.abc.Mapping`
            A Mapping with the to-be applied constraints per atom (pair).

        """
        pass


class ParamMapping(ParamMappingABC):

    #: A set of charge-like parameters which require a parameter re-normalization after every move.
    CHARGE_LIKE: ClassVar[FrozenSet[str]] = frozenset({
        'charge'
    })

    @ParamMappingABC.inherit_annotations()
    def __init__(self, df, move_range, func=np.multiply, **kwargs):
        super().__init__(df, move_range, func=func, **kwargs)

    def identify_move(self) -> Tuple[Tup2, float, float]:
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

    def clip_move(self, idx: Tup2, value: float) -> float:
        """Ensure that **value** falls within a user-specified range.

        Parameters
        ----------
        :class:`tuple` [:class:`~Collections.abc.Hashable`, :class:`~Collections.abc.Hashable`]
            The index of the moved parameter.
        :class:`float`
            The value of the moved parameter.

        Returns
        -------
        :class:`float`
            The newly clipped value of the moved parameter.

        """
        prm_min = self['min'].at[idx]
        prm_max = self['max'].at[idx]
        return np.clip(value, prm_min, prm_max)

    def apply_constraints(self, idx: Tup2, value: float, constraints: ConstrainDict) -> None:
        """Apply further constraints based on **idx** and **value**.

        Performs an inplace update of this instance.

        Parameters
        ----------
        :class:`tuple` [:class:`~Collections.abc.Hashable`, :class:`~Collections.abc.Hashable`]
            The index of the moved parameter.
        :class:`float`
            The value of the moved parameter.
        :class:`~collections.abc.Mapping`
            A Mapping with the to-be applied constraints per atom (pair).

        """
        prm_type, atoms = idx
        charge = prm_type in self.CHARGE_LIKE
        update_charge(atoms, value, self['param'], self['count'],
                      constrain_dict=constraints, charge=charge)


ParamMapping.__doc__ = ParamMappingABC.__doc__