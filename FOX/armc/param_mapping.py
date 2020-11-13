"""A module containing the :class:`ParamMapping` class.

Index
-----
.. currentmodule:: FOX.armc
.. autosummary::
    ParamMappingABC
    ParamMapping

API
---
.. autoclass:: ParamMappingABC
    :members:
.. autoclass:: ParamMapping
    :members:

"""

import warnings
from copy import deepcopy
from abc import ABC, abstractmethod
from types import MappingProxyType
from logging import Logger
from functools import wraps, partial
from typing import (Any, TypeVar, Optional, Tuple, Mapping, Iterable, ClassVar, Union,
                    Iterator, KeysView, ItemsView, ValuesView, overload, Callable, FrozenSet, cast,
                    MutableMapping, TYPE_CHECKING, Dict)

import numpy as np
import pandas as pd
from assertionlib.dataclass import AbstractDataClass
from nanoutils import Literal, TypedDict, set_docstring

from ..type_hints import ArrayLike
from ..functions.charge_utils import update_charge, get_net_charge, ChargeError

if TYPE_CHECKING:
    from pandas.core.generic import NDFrame
else:
    from ..type_alias import NDFrame

__all__ = ['ParamMappingABC', 'ParamMapping']

# A generic typevar
T = TypeVar('T')

# MultiIndex keys
Tup3 = Tuple[Any, Any, Any]
Tup2 = Tuple[Any, Any]

# All dict keys in ParamMappingABC
SeriesKeys = Literal['min', 'max', 'count', 'frozen', 'guess']
DFKeys = Literal['param', 'param_old']
ValidKeys = Union[SeriesKeys, DFKeys]

# A function for moving parameters
MoveFunc = Callable[[float, float], float]

# A Mapping representing the conent of ParamMappingABC
_ParamMappingABC = Mapping[ValidKeys, NDFrame]

_KeysView = KeysView[ValidKeys]  # type: ignore[misc]
_ItemsView = ItemsView[ValidKeys, NDFrame]


class _InputMapping(TypedDict):
    param: Union[NDFrame, Mapping[str, pd.Series]]


class InputMapping(_InputMapping, total=False):
    """A :class:`~typing.TypedDict` representing the :class:`ParamMappingABC` input."""

    min: pd.Series
    max: pd.Series
    count: pd.Series
    frozen: pd.Series
    guess: pd.Series


class Data(TypedDict):
    """A :class:`~typing.TypedDict` representing :attr:`ParamMappingABC._data`."""

    param: pd.DataFrame
    param_old: pd.DataFrame
    min: pd.Series
    max: pd.Series
    count: pd.Series
    frozen: pd.Series
    guess: pd.Series


def _parse_param(dct: MutableMapping[str, Any]) -> pd.DataFrame:
    # Check that the 'param' key is present
    try:
        _param = dct['param']
    except KeyError as ex:
        raise KeyError(f"The {'param'!r} key is absent from the passed mapping") from ex

    # Cast the data into the correct shape
    if isinstance(_param, pd.Series):
        dct['param'] = param = _param.to_frame(name=0)
    elif isinstance(_param, pd.DataFrame):
        param = _param
    else:
        try:
            dct['param'] = param = pd.DataFrame(_param)
        except TypeError as ex:
            raise TypeError(f"the 'param' value expected a Series, DataFrame or dict of Series; "
                            f"observed type: {_param.__class__.__name__!r}") from ex

    # Check that it has a 3-level MultiIndex
    n_level = 3
    if not isinstance(param.index, pd.MultiIndex):
        raise TypeError(f"Series.index expected a {n_level}-level MultiIndex; "
                        f"observed type: {param.index.__class__.__name!r}")
    elif len(param.index.levels) != n_level:
        raise ValueError(f"Series.index expected a {n_level}-level MultiIndex; "
                         f"observed number levels: {len(param.index.levels)}")
    return param


class ParamMappingABC(AbstractDataClass, ABC, _ParamMappingABC):
    r"""A :class:`~collections.abc.Mapping` for storing and updating forcefield parameters.

    Besides the implementation of the :class:`~collections.abc.Mapping` protocol,
    this class has access to four main methods:

    * :meth:`__call__` or :meth:`move` move a random parameter by a random step size.
    * :meth:`identify_move` identify the parameter and move step size.
    * :meth:`clip_move` clip the move.
    * :meth:`apply_constraints` apply further constraints to the move.

    Note that  :meth:`__call__` will internally call all other three methods.

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
    _data_dict: Data

    #: Fill values for when optional keys are absent.
    FILL_VALUE: ClassVar[Mapping[ValidKeys, np.generic]] = MappingProxyType({
        'min': np.float64(-np.inf),
        'max': np.float64(np.inf),
        'count': np.int64(-1),
        'frozen': np.False_,
        'guess': np.False_,
    })

    _PRIVATE_ATTR = frozenset({'_net_charge'})  # type: ignore

    def __init__(
        self,
        data: Union[InputMapping, pd.DataFrame],
        move_range: Iterable[float],
        func: MoveFunc,
        constraints: Optional[Mapping[Tup2, Optional[Iterable[Mapping[str, float]]]]] = None,
        **kwargs: Any,
    ) -> None:
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
        self._data = cast(Data, data)
        self.move_range = cast(np.ndarray, move_range)
        self.func: Callable[[float, float], float] = wraps(func)(partial(func, **kwargs))
        self.constraints = cast(Dict[Tup2, Optional[Tuple[pd.Series, ...]]], constraints)

    # Properties

    @property
    def constraints(self) -> Dict[Tup2, Optional[Tuple[pd.Series, ...]]]:
        return self._constraints

    @constraints.setter
    def constraints(
        self, value: Optional[Mapping[Tup2, Optional[Iterable[Mapping[str, float]]]]]
    ) -> None:
        if value is None:
            dct: Dict[Tup2, Optional[Tuple[pd.Series, ...]]] = {}
        else:
            func = lambda v: tuple(pd.Series(i) for i in v) if v is not None else None  # noqa: E731,E501
            dct = {k: func(v) for k, v in value.items()}

        for key in self['param'].index:
            dct.setdefault(key[:2], None)
        self._constraints = dct

    @property
    def move_range(self) -> np.ndarray:
        return self._move_range

    @move_range.setter
    def move_range(self, value: ArrayLike) -> None:
        _ar = np.array(value, dtype=float, ndmin=1, copy=False)
        prm_len = len(self._data['param'].columns)

        if _ar.ndim == 2:
            if len(_ar) != prm_len:
                if prm_len == 1:
                    for i in range(1, len(_ar)):
                        self['param'][i] = self['param'][0].copy()
                        self['param_old'][i] = self['param_old'][0].copy()
                else:
                    raise ValueError(f"Expected 'move_range' length: {prm_len}; "
                                     f"observed length: {len(_ar)}")
            ar = _ar
        elif _ar.ndim == 1:
            ar = np.tile(_ar, prm_len)
            ar.shape = prm_len, -1
        else:
            raise ValueError("'move_range' expected a 1D or 2D array; "
                             f"observed dimensionality: {_ar.ndim}")
        self._move_range: np.ndarray = ar

    @property
    def _data(self) -> Data:
        return self._data_dict

    @_data.setter
    def _data(self, value: Union[InputMapping, pd.DataFrame]) -> None:
        dct = dict(value)

        # Check that the 'param' key is present
        param = _parse_param(dct)
        _, prm = next(iter(param.items()))

        # Fill in the defaults
        for name, fill_value in self.FILL_VALUE.items():
            if name in dct:
                continue
            dtype = type(fill_value) if fill_value is not None else object
            dct[name] = pd.Series(fill_value, index=prm.index, name=name, dtype=dtype)

        # Construct a dictionary to contain the old parameter
        dct['param_old'] = param.copy()
        self._data_dict = cast(Data, dct)

        # Cache the total charge of the system
        self._set_net_charge()

    # Magic methods and Mapping implementation

    def __eq__(self, value: Any) -> bool:
        """Implement :code:`self == value`."""
        if type(self) is not type(value):
            return False

        ret = True
        ret &= np.all(self.move_range == value.move_range)
        ret &= cast(bool, self._net_charge == value._net_charge)

        names = ("func", "args", "keywords")
        v1, v2 = self.func, value.func
        ret &= all([getattr(v1, n, np.nan) == getattr(v2, n, np.nan) for n in names])

        ret &= cast(bool, self.keys() == value.keys())
        iterator = ((v, value[k]) for k, v in self.items())
        for v1, v2 in iterator:
            ret &= np.all(v1 == v2)
        return ret

    @AbstractDataClass.inherit_annotations()
    def _str_iterator(self):
        return ((k.strip('_'), v) for k, v in super()._str_iterator())

    def _set_net_charge(self) -> None:
        """Set the total charge of the system."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
            if 'charge' in self['param'].index:
                self._net_charge = get_net_charge(self['param'].loc['charge', 0],
                                                  self['count'].loc['charge'])
            else:
                self._net_charge = None

    @overload
    def __getitem__(self, key: SeriesKeys) -> pd.Series: ...
    @overload
    def __getitem__(self, key: DFKeys) -> pd.DataFrame: ...
    @overload
    def __getitem__(self, key: Any) -> Any: ...
    def __getitem__(self, key):  # noqa: E301
        """Implement :code:`self[key]`."""
        return self._data[key]

    def __iter__(self) -> Iterator[ValidKeys]:
        """Implement :code:`iter(self)`."""
        return cast(Iterator[ValidKeys], iter(self._data))

    def __len__(self) -> int:
        """Implement :code:`len(self)`."""
        return len(self._data)

    @overload
    def __contains__(self, key: ValidKeys) -> Literal[True]: ...
    @overload
    def __contains__(self, key: Any) -> bool: ...
    def __contains__(self, key):  # noqa: E301
        """Implement :code:`key in self`."""
        return key in self._data

    def keys(self) -> _KeysView:
        """Return a set-like object providing a view of this instance's keys."""
        return cast(_KeysView, self._data.keys())

    def items(self) -> _ItemsView:
        """Return a set-like object providing a view of this instance's key/value pairs."""
        return cast(_ItemsView, self._data.items())

    def values(self) -> ValuesView[NDFrame]:
        """Return an object providing a view of this instance's values."""
        return self._data.values()

    @overload
    def get(self, key: ValidKeys) -> NDFrame: ...
    @overload
    def get(self, key: ValidKeys, default: Any) -> NDFrame: ...
    @overload
    def get(self, key: Any) -> Optional[NDFrame]: ...
    @overload
    def get(self, key: Any, default: T) -> Union[NDFrame, T]: ...
    def get(self, key, default=None):  # noqa: E301
        """Return the value for **key** if it's available; return **default** otherwise."""
        return self._data.get(key, default)

    # The actual meat of the class

    def __call__(self, logger: Optional[Logger] = None,
                 param_idx: int = 0) -> Union[Exception, Tup3]:
        """Update a random parameter in **self.param** by a random value from **self.move.range**.

        Performs in inplace update of the ``'param'`` column in **self.param**.
        By default the move is applied in a multiplicative manner.
        **self.job.md_settings** and **self.job.preopt_settings** are updated to reflect the
        change in parameters.

        Parameters
        ----------
        logger : :class:`logging.Logger`, optional
            A logger for reporting the updated value.

        param_idx : :class:`int`, optional
            The index of the parameter.
            Only relevant when multiple parameter sets have to be stored
            (see :class:`MultiParamMaping`).

        Returns
        -------
        :class:`tuple` [:class:`~Collections.abc.Hashable`, :class:`~Collections.abc.Hashable`]
            The index of the updated parameter.

        """
        # Prepare arguments a move
        idx, x1, x2 = self.identify_move(param_idx)
        _value = self.func(x1, x2)
        value = self.clip_move(idx, _value)

        # Create a call to the logger
        if logger is not None:
            _, prm_type, atoms = idx
            logger.info(f"Moving {prm_type} ({atoms}): {x1:.4f} -> {value:.4f}")

        ex = self.apply_constraints(idx, value, param_idx)
        if ex is not None:
            return ex

        self['param'].loc[idx, param_idx] = value
        return idx

    @abstractmethod
    def identify_move(self, param: int) -> Tuple[Tup3, float, float]:
        """Identify the to-be moved parameter and the size of the move.

        Parameters
        ----------
        param : :class:`~collections.abc.Hashable`
            The name of the parameter-containg column.

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

    def apply_constraints(self, idx: Tup3, value: float, param: int) -> Optional[Exception]:
        """An optional function for applying further constraints based on **idx** and **value**.

        Should perform an inplace update of this instance.

        Parameters
        ----------
        idx : :class:`tuple` [:class:`~Collections.abc.Hashable`, :class:`~Collections.abc.Hashable`]
            The index of the moved parameter.

        value : :class:`float`
            The value of the moved parameter.

        param : :class:`~collections.abc.Hashable`
            The name of the parameter-containg column.

        idx : :class:`~collections.abc.Mapping`
            A Mapping with the to-be applied constraints per atom (pair).

        """  # noqa
        pass

    def to_struct_array(self) -> np.ndarray:
        """Stack all :class:`~pandas.Series` in this instance into a single structured array."""
        cls = type(self)
        dtype = np.dtype(list((k, type(v)) for k, v in cls.FILL_VALUE.items()))
        data = [self[k].values for k in cls.FILL_VALUE]
        return np.fromiter(zip(*data), dtype=dtype)

    def constraints_to_str(self) -> pd.Series:
        """Convert the constraints into a human-readably :class:`pandas.Series`."""
        dct = {k: '' for k in self.constraints}
        for key, tup in self.constraints.items():
            if tup is None:
                continue
            dct[key] += ' == '.join(
                ' + '.join(f'{v}*{k}' for k, v in series.items()) for series in tup
            )
        ret = pd.Series(dct)
        ret.name = 'constraints'
        ret.index.names = self['param'].index.names[:2]
        return ret

    def to_yaml_dict(self) -> Dict[str, Any]:
        cls = type(self)
        func = cast('partial[float]', self.func)
        try:
            if isinstance(func.func, np.ufunc):
                module = 'numpy'
            else:
                module = func.func.__module__
            name = getattr(func.func, '__qualname__', func.func.__name__)
        except AttributeError as ex:
            raise TypeError(f"Failed to parse {cls.__name__}.func.func: {func.func!r}") from ex

        ret = {
            'type': f'{cls.__module__}.{cls.__name__}',
            'move_range': self.move_range.tolist(),
            'func': f'{module}.{name}',
            'kwargs': func.keywords,
            'validation': {
                'allow_non_existent': True,
                'charge_tolerance': 'inf'
            }
        }

        _template = {
            'param': '',
            'constraints': [],
            'frozen': {},
        }
        idx_dict = {}

        index = self['param'].index
        constraints = self.constraints_to_str()
        for key, param, _ in index:
            if (key, param) in idx_dict:
                continue

            template = deepcopy(_template)
            template['param'] = param
            if constraints.at[key, param]:
                template['constraints'].append(constraints.at[key, param])

            lst = ret.setdefault(key, [])
            lst.append(template)
            idx_dict[key, param] = len(lst) - 1

        # Set the extremites
        iterator1 = ((k, self['min'].at[k], self['max'].at[k]) for k in index)
        for (key, param, atom), min_, max_ in iterator1:
            i = idx_dict[key, param]
            ret[key][i]['constraints'].append(f'{min_} < {atom} < {max_}')

        # Set the parameters
        iterator2 = ((k, self['param'].at[k, 0], self['frozen'].at[k]) for k in index)
        for (key, param, atom), value, frozen in iterator2:
            i = idx_dict[key, param]
            if frozen:
                ret[key][i]['frozen'][atom] = value.item()
            else:
                ret[key][i][atom] = value.item()
        return ret


MOVE_RANGE = np.array([[
    0.900, 0.905, 0.910, 0.915, 0.920, 0.925, 0.930, 0.935, 0.940,
    0.945, 0.950, 0.955, 0.960, 0.965, 0.970, 0.975, 0.980, 0.985,
    0.990, 0.995, 1.005, 1.010, 1.015, 1.020, 1.025, 1.030, 1.035,
    1.040, 1.045, 1.050, 1.055, 1.060, 1.065, 1.070, 1.075, 1.080,
    1.085, 1.090, 1.095, 1.100
]], dtype=float)
MOVE_RANGE.setflags(write=False)


@set_docstring(ParamMappingABC.__doc__)
class ParamMapping(ParamMappingABC):

    #: A set of charge-like parameters which require a parameter re-normalization after every move.
    CHARGE_LIKE: ClassVar[FrozenSet[str]] = frozenset({
        'charge'
    })

    @ParamMappingABC.inherit_annotations()
    def __init__(self, data, move_range=MOVE_RANGE, func=np.multiply, **kwargs):
        super().__init__(data, move_range, func=func, **kwargs)

    def identify_move(self, param_idx: int) -> Tuple[Tup3, float, float]:
        """Identify and return a random parameter and move size.

        Parameters
        ----------
        param_idx : :class:`~collections.abc.Hashable`
            The name of the parameter-containg column.

        Returns
        -------
        :class:`tuple` [:class:`~Collections.abc.Hashable`, :class:`~Collections.abc.Hashable`], :class:`float` and :class:`float`
            The index of the to-be moved parameter, it's value and the size of the move.

        """  # noqa
        # Define a random parameter
        variable = ~self['frozen']
        random_prm: pd.Series = self['param'].loc[variable, param_idx].sample()
        idx, x1 = next(random_prm.items())  # Type: Tup3, float

        # Define a random move size
        x2: float = np.random.choice(self.move_range[param_idx], 1)[0]
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
        prm_min: float = self['min'][idx]
        prm_max: float = self['max'][idx]
        return np.clip(value, prm_min, prm_max)

    def apply_constraints(self, idx: Tup3, value: float, param_idx: int) -> Optional[ChargeError]:
        """Apply further constraints based on **idx** and **value**.

        Performs an inplace update of this instance.

        Parameters
        ----------
        idx : :class:`tuple` [:class:`~Collections.abc.Hashable`, :class:`~Collections.abc.Hashable`]
            The index of the moved parameter.

        value : :class:`float`
            The value of the moved parameter.

        param_idx : :class:`~collections.abc.Hashable`
            The name of the parameter-containg column.

        constraints : :class:`~collections.abc.Mapping`
            A Mapping with the to-be applied constraints per atom (pair).

        """  # noqa
        key = idx[:2]
        atom = idx[2]
        charge = self._net_charge if key[1] in self.CHARGE_LIKE else None

        frozen_idx = self['frozen'].loc[key]
        frozen = frozen_idx.index[frozen_idx] if frozen_idx.any() else None

        return update_charge(
            atom, value,
            param=self['param'].loc[key, param_idx],
            count=self['count'].loc[key],
            atom_coefs=self.constraints[key],
            prm_min=self['min'].loc[key],
            prm_max=self['max'].loc[key],
            net_charge=charge,
            exclude=frozen,
        )
