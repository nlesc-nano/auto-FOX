"""A module with functions related to the parsing of parameter constraints.

Index
-----
.. currentmodule:: FOX.functions.charge_parser
.. autosummary::
    assign_constraints

API
---
.. autofunction:: assign_constraints

"""

from types import MappingProxyType
from typing import Optional, Dict, Union, Iterable, Tuple, Generator, List, cast

__all__ = ["assign_constraints"]

ExtremiteDict = Dict[Tuple[str, str], float]
ConstrainList = List[Dict[str, float]]

#: Map ``"min"`` to ``"max"`` and *vice versa*.
_INVERT = MappingProxyType({
    'max': 'min',
    'min': 'max'
})

#: Map :math:`>`, :math:`<`, :math:`\ge` and :math:`\le` to either ``"min"`` or ``"max"``.
_OPPERATOR_MAPPING = MappingProxyType({
    '<': 'min',
    '<=': 'min',
    '>': 'max',
    '>=': 'max'
})


def assign_constraints(constraints: Union[str, Iterable[str]]
                       ) -> Tuple[ExtremiteDict, Optional[ConstrainList]]:
    operator_set = {'>', '<', '*', '==', '+', '-'}

    # Parse integers and floats
    if isinstance(constraints, str):
        constraints = [constraints]

    constrain_list = []
    for item in constraints:
        for i in operator_set:  # Sanitize all operators; ensure they are surrounded by spaces
            item = item.replace(i, f'~{i}~')

        item_list: List[Union[str, float]] = [i.strip().rstrip() for i in item.split('~')]
        if len(item_list) == 1:
            continue

        for j, k in enumerate(item_list):  # Convert strings to floats where possible
            try:
                float_k = float(k)
            except ValueError:
                pass
            else:
                item_list[j] = float_k

        constrain_list.append(item_list)

    # Set values in **param**
    extremite_dict: ExtremiteDict = {}
    constraints_: Optional[ConstrainList] = None
    for constrain in constrain_list:
        if '==' in constrain:
            constraints_ = _eq_constraints(constrain)
        else:
            extremite_dict.update(_gt_lt_constraints(constrain))
    return extremite_dict, constraints_


def _gt_lt_constraints(constrain: List[Union[str, float]]
                       ) -> Generator[Tuple[Tuple[str, str], float], None, None]:
    r"""Parse :math:`>`, :math:`<`, :math:`\ge` and :math:`\le`-type constraints."""
    minus = False
    for i, j in enumerate(constrain):  # type: int, str  # type: ignore[assignment]
        if j == '-':
            minus = True
            continue
        if j not in _OPPERATOR_MAPPING:
            continue

        operator, value, atom = _OPPERATOR_MAPPING[j], constrain[i-1], constrain[i+1]
        if isinstance(atom, float):
            atom, value = value, atom
            operator = _INVERT[operator]
        if minus:
            value *= -1
            minus = False
        yield (atom, operator), value  # type: ignore[misc]


def _find_float(iterable: Union[Tuple[str], Tuple[str, str]]) -> Tuple[str, float]:
    """Take an iterable of 2 strings and identify which element can be converted into a float."""
    try:
        i, j = iterable  # type: ignore[misc]
    except ValueError:
        return iterable[0], 1.0

    if i == "-":
        return j, -1.0
    elif j == "-":
        return i, -1.0

    try:
        return j, float(i)
    except ValueError:
        return i, float(j)


def _eq_constraints(constrain_: List[Union[str, float]]) -> ConstrainList:
    """Parse :math:`a = i * b`-type constraints."""
    constrain = ''.join(str(i) for i in constrain_).split('==')
    # import pdb; pdb.set_trace()

    # Assign all other constraints
    ret = []
    for _item in constrain:
        dct: Dict[str, float] = {}
        if ")" in _item or "(" in _item:
            raise NotImplementedError(f"Parenthesized constraints are not supported: {_item!r}")

        item_split = _split_operator(_item)
        for item in item_split:
            item_list = cast(Tuple[str, str], item.split('*'))
            atom, i = _find_float(item_list)
            dct[atom] = i

        ret.append(dct)
    return ret


def _split_operator(constrain: str) -> List[str]:
    plus_split = constrain.split("+")
    if '-' not in constrain:
        return plus_split

    ret = []
    for item in plus_split:
        if '-' not in item:
            ret.append(item)
            continue

        item_split = item.split("-")
        if item_split[0]:
            ret.append(item_split[0])
        ret += [f'-{i}' for i in item_split[1:]]
    return ret
